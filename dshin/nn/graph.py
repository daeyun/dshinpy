import tensorflow as tf

import collections
import numpy as np
import textwrap

from IPython.display import display, HTML


class IPythonTfGraph(object):
    """
    Display TensorFlow graphs in Jupyter. http://stackoverflow.com/a/38192374
    """

    def __init__(self, graph):
        self.graph = graph

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

    def show(self, graph_or_graph_def=None, max_const_size=32):
        """
        Visualize TensorFlow graph.
        """
        if graph_or_graph_def is None:
            graph_or_graph_def = self.graph

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
            <div style="height:600px">
              <tf-graph-basic id="{id}"></tf-graph-basic>
            </div>
        """.format(data=repr(str(strip_def)), id='graph' + str(np.random.rand()))
        code = textwrap.dedent(code)

        iframe = """<iframe seamless style="width:800px;height:620px;border:0" srcdoc="{}"></iframe>""".format(code.replace('"', '&quot;'))

        display(HTML(iframe))


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
