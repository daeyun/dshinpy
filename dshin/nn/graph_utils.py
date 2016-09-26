import tensorflow as tf

from dshin import log
import tempfile

import contextlib
import os
from os import path
import collections
import numpy as np
import textwrap

from IPython import display
from google.protobuf import text_format


class IPythonTfGraph(object):
    """
    Display TensorFlow graphs in Jupyter.
    http://stackoverflow.com/a/38192374
    https://github.com/tensorflow/tensorflow/issues/1978
    """

    def __init__(self, graph):
        if isinstance(graph, str) and 'pbtxt' in graph:
            self.graph_def = self.pbtxt_to_graph_def(graph)
        elif isinstance(graph, tf.GraphDef):
            self.graph_def = graph
        elif isinstance(graph, tf.Graph):
            self.graph_def = graph.as_graph_def()
        else:
            raise NotImplementedError()

    def pbtxt_to_graph_def(self, filename):
        """
        http://stackoverflow.com/a/38192374/6020752
        """
        gdef = tf.GraphDef()
        with open(filename) as f:
            text_format.Merge(f.read(), gdef)
        return gdef

    def strip_consts(self, graph_def, max_const_size=32):
        """
        Strip large constant values from graph_def.
        """
        strip_def = tf.GraphDef()
        for n0 in graph_def.node:
            n = strip_def.node.add()
            n.MergeFrom(n0)
            if n.op == 'Const':
                tensor = n.attr['value'].tensor
                size = len(tensor.tensor_content)
                if size > max_const_size:
                    tensor.tensor_content = "<stripped %d bytes>" % size
        return strip_def

    def rename_nodes(self, graph_def, rename_func):
        res_def = tf.GraphDef()
        for n0 in graph_def.node:
            n = res_def.node.add()
            n.MergeFrom(n0)
            n.name = rename_func(n.name)
            for i, s in enumerate(n.input):
                n.input[i] = rename_func(s) if s[0] != '^' else '^' + rename_func(s[1:])
        return res_def

    def display_full_width(self):
        display.display(display.HTML("<style>.container { width:100% !important; }</style>"))

    def show(self, graph_or_graph_def=None, max_const_size=32, height=800, expand_page_width=True):
        """
        Visualize TensorFlow graph.
        """
        if graph_or_graph_def is None:
            graph_or_graph_def = self.graph_def

        if hasattr(graph_or_graph_def, 'as_graph_def'):
            graph_or_graph_def = graph_or_graph_def.as_graph_def()

        strip_def = self.strip_consts(graph_or_graph_def, max_const_size=max_const_size)
        code = """
            <script>
              function load() {{
                document.getElementById("{id}").pbtxt = {data};
              }}
            </script>
            <link rel="import" href="https://tensorboard.appspot.com/tf-graph-basic.build.html" onload=load()>
            <div style="height:{height}px">
              <tf-graph-basic id="{id}"></tf-graph-basic>
            </div>
        """.format(data=repr(str(strip_def)), id='graph' + str(np.random.rand()), height=height)
        code = textwrap.dedent(code)

        iframe = """<iframe seamless style="width:100%;height:{}px;border:0" srcdoc="{}"></iframe>""".format(
            height + 20, code.replace('"', '&quot;'))

        display.display(display.HTML(iframe))
        if expand_page_width:
            self.display_full_width()


def backtrace_invalid(tensor: tf.Tensor, feed_dict, maxdepth=2):
    if maxdepth < 0:
        return

    sess = tf.get_default_session()
    assert sess is not None

    def count_invalid_single(tensor):
        eval_values = sess.run(tensor, feed_dict=feed_dict)
        values, counts = np.unique(eval_values[~np.isfinite(eval_values)], return_counts=True)
        total = eval_values.size
        return values, counts, total

    def count_invalid(tensors):
        if isinstance(tensors, (list, tuple)):
            return [count_invalid_single(t) for t in tensors]
        else:
            return count_invalid_single(tensors)

    queue = collections.deque()
    invalid_counts = count_invalid(tensor)
    current_depth = 0
    if invalid_counts[1].sum() > 0:
        queue.append([tensor, current_depth, invalid_counts])

    results = []

    while len(queue) > 0:
        parent, d, counts = queue.popleft()
        parent_invalid_values, parent_invalid_counts, parent_total_count = counts
        parent_invalid_count = parent_invalid_counts.sum()

        children = []
        has_all_valid_inputs = True
        child_invalid_counts = count_invalid(list(parent.op.inputs))
        for child, (invalid_values, invalid_counts, total_count) in zip(list(parent.op.inputs), child_invalid_counts):
            invalid_count = invalid_counts.sum()
            children.append([child, invalid_count])
            if invalid_count > 0:
                has_all_valid_inputs = False
                if d < maxdepth:
                    if d != current_depth or current_depth == 0:
                        current_depth = d
                        if current_depth != 0:
                            print()
                        print('Searching level {} {}'.format(d, parent.name), end='')
                    else:
                        print('.', end='')
                    queue.append([child, d + 1, (invalid_values, invalid_counts, total_count)])
        print(' ', end='')

        if has_all_valid_inputs:
            results.append(parent)
            print()
            print('{} invalid values out of {} in {} at depth {}'.format(parent_invalid_count, parent_total_count, parent.name, d))
            if len(parent_invalid_values) > 1:
                for i in range(len(parent_invalid_values)):
                    print('{}: {}'.format(parent_invalid_values[i], parent_invalid_counts[i]))
            print('Op: {}'.format(parent.op.type))
            print('Inputs:')
            if children:
                for child, invalid_count in children:
                    print('{} - {} of shape {} and dtype {}'.format(child.name, type(child).__name__, child.get_shape(), child.dtype.name))
            else:
                print('No input values.')
            print()

    return results


@contextlib.contextmanager
def collect_values(key, graph=None):
    if graph is None:
        graph = tf.get_default_graph()

    def collect():
        values = {value for value in graph.get_collection(key)}
        return values

    existing_values = collect()

    new_values = []

    # Expose the reference to `new_ops`.
    yield new_values

    assert not new_values
    for v in collect() - existing_values:
        new_values.append(v)


@contextlib.contextmanager
def abs_name_scope(name):
    with tf.name_scope(name + '/') as scope:
        yield scope


def save_graph_text_summary(graph: tf.Graph, dirname=None, random_dirname=False, basename='graph_summary.txt', verbose=False):
    if dirname is None:
        dirname = '/tmp/nn_logs'

    if random_dirname:
        if not path.isdir(dirname):
            os.makedirs(dirname)
        dirname = tempfile.mkdtemp(prefix='', dir=dirname)

    def device(value):
        def func():
            d = value.device
            if not d:
                return '-'
            return value.device

        return func

    def summary_items(value):
        eval_funcs = [
            device(value),
            lambda: value.dtype.name,
            lambda: type(value).__name__,
            lambda: value.name,
            lambda: value.get_shape(),
            lambda: '',
        ]
        items = []
        for fn in eval_funcs:
            try:
                items.append(fn())
            except:
                items.append('-')
        return items

    def summary_items_io(value, label):
        eval_funcs = [
            device(value),
            lambda: label,
            lambda: value.dtype.name,
            lambda: type(value).__name__[0],
            lambda: value.name,
            lambda: value.get_shape(),
        ]
        assert type(value).__name__ in ['Tensor', 'Variable', 'Operation']
        items = []
        for fn in eval_funcs:
            try:
                items.append(fn())
            except:
                items.append('-')
        return items

    def summary_items_op(value):
        eval_funcs = [
            device(value),
            lambda: value.type,
            lambda: value.name,
            lambda: str(','.join([item.decode('utf-8') for item in value.colocation_groups()])),
            lambda: '',
            lambda: '',
        ]
        assert type(value).__name__ == 'Operation'
        items = []
        for fn in eval_funcs:
            try:
                items.append(fn())
            except:
                items.append('-')
        return items

    def items_to_line(items, widths=(50, 20, 20, 1)):
        return '{{:<{0}}} {{:<{1}}} {{:<{2}}} {{:<{3}}} {{}} {{}}'.format(*widths).format(*items)

    by_graph_key = []
    for key in graph.get_all_collection_keys():
        values = graph.get_collection(key)
        items_list = []
        for value in values:
            items_list.append(summary_items(value))
        items_list = sorted(items_list)
        lines = [items_to_line(items, widths=[20, 20, 20, 1]) for items in items_list]

        section = textwrap.dedent('''
        # {key}
        {items}
        ''').format(key=key, items='\n'.join(lines)).strip() + '\n'
        by_graph_key.append(section)
    by_graph_key.sort()

    by_operation = []
    for op in graph.get_operations():
        assert isinstance(op, tf.Operation)
        opline = items_to_line(summary_items_op(op), widths=[20, 29, 50, 1])
        outputlines = []
        for value in op.inputs:
            outputlines.append(items_to_line(summary_items_io(value, '(in)'), widths=[22, 6, 18, 1]))
        for value in op.outputs:
            outputlines.append(items_to_line(summary_items_io(value, '(out)'), widths=[22, 6, 18, 1]))
        for i, value in enumerate(op.control_inputs):
            if i < 3:
                outputlines.append(items_to_line(summary_items_io(value, '(dep)'), widths=[22, 6, 18, 1]))
            else:
                outputlines.append(items_to_line(['', '(dep)', '... ({} total)'.format(len(op.control_inputs)), '', '', ''], widths=[22, 6, 18, 1]))
                break

        section = textwrap.dedent('''
        {op}
        {outputs}
        ''').format(op=opline, outputs='\n'.join(outputlines)).strip() + '\n'

        by_operation.append(section)

    content = textwrap.dedent('''
    By graph collection keys:
    -----------------------------------
    {by_graph_key}

    By operations and outputs:
    -----------------------------------
    {by_operation}
    ''').format(by_graph_key='\n'.join(by_graph_key),
                by_operation='\n'.join(by_operation)).strip()

    filename = path.join(dirname, basename)
    with open(filename, 'w') as f:
        f.write(content)

    if verbose:
        log.info('Wrote graph text summary: {}'.format(filename))
    return filename
