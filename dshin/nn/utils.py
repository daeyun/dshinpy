"""
Helpers for managing TensorFlow neural net models.
"""
import abc
import functools
import numbers
import os
import re
from os import path

import tensorflow as tf
import toposort
import typecheck as tc
import typing

from dshin import log
from dshin.nn import ops as nn_ops
from dshin.nn import types as nn_types

memoize = functools.lru_cache(maxsize=2048, typed=True)


@tc.typecheck
def sort_tensors(ops: nn_types.Operations) -> tc.seq_of(str):
    """
    Sorts the inputs and outputs of TensorFlow Operations in topological order.

    :param ops: List of Operations.
    :return: Sorted list of Tensor names.
    """
    # http://stackoverflow.com/a/33851308
    deps = {}
    for op in ops:
        # op node
        op_inputs = set()
        op_inputs.update([t.name for t in op.inputs])
        deps[op.name] = op_inputs
        # tensor output node
        for t in op.outputs:
            deps[t.name] = {op.name}

    out = []
    tensors = toposort.toposort(deps)
    for tensor in tensors:
        out.extend(list(tensor))

    return out


@tc.typecheck
def match_names(values: nn_types.ValuesOrOperations,
                pattern: tc.optional(str) = None,
                prefix: tc.optional(str) = None,
                suffix: tc.optional(str) = None) -> nn_types.ValuesOrOperations:
    """
    Filters TensorFlow graph objects by regular expression.

    :param values: A list of Variables, Tensors, or Operations.
    :param pattern: A regular expression pattern.
    :param prefix: A string prepended to `pattern`. Defaults to ``.*?``.
    :param suffix: A string appended to `pattern`. Defaults to ``.*?``.
    :return:
    """
    if pattern is None:
        pattern = r'.*?'
    if prefix is None:
        prefix = r'.*?'
    if suffix is None:
        suffix = r'.*?'

    final_pattern = ''.join([prefix, pattern, suffix])
    matched = []
    for item in values:
        if re.match(final_pattern, item.name):
            matched.append(item)
    return matched


@tc.typecheck
def default_sess_config(mem: float = 0.95,
                        log_device_placement: bool = False) -> tf.ConfigProto:
    conf = tf.ConfigProto()
    conf.gpu_options.per_process_gpu_memory_fraction = mem
    conf.gpu_options.allocator_type = 'BFC'
    conf.gpu_options.allow_growth = True
    conf.allow_soft_placement = True
    conf.log_device_placement = log_device_placement
    return conf


class GraphKeys(tf.GraphKeys):
    """
    Extends TensorFlow Graph collection keys.
    """
    SIMPLE_SUMMARIES = 'simple_summaries'
    IMAGE_SUMMARIES = 'image_summaries'
    TRAIN_UPDATE_SUMMARIES = 'train_update_summaries'
    EVAL_VALUES = 'eval_values'
    LAYER_TENSORS = 'layer_tensors'
    # Also used: LOSSES, SUMMARIES


# noinspection PyBroadException
class NNModel(metaclass=abc.ABCMeta):
    """
    TensorFlow neural net container.

    Implementation example:

    >>> class SampleNet(NNModel):
    ...     def _model(self):
    ...         input_placeholder = self['input']
    ...         target_placeholder = self['target']
    ... 
    ...         out = nn_ops.conv2d(input_placeholder, n_out=1, use_bias=False)
    ...         out = nn_ops.batch_norm(out, is_trainable=True, is_local=True)
    ...         self._loss = tf.reduce_mean((out - target_placeholder) ** 2, name='loss')
    ... 
    ...     def _minimize_op(self):
    ...         lr_placeholder = self['learning_rate']
    ...         return tf.train.AdamOptimizer(lr_placeholder).minimize(self._loss)
    ... 
    ...     def _placeholders(self):
    ...         return [
    ...             tf.placeholder(tf.float32, shape=[None, 5, 5, 1], name='input'),
    ...             tf.placeholder(tf.float32, shape=[None, 5, 5, 1], name='target'),
    ...         ]

    Training example:

    >>> import numpy as np
    >>> net = SampleNet(seed=42)
    >>> np.random.seed(42)
    >>> feed_dict = {'input': np.random.randn(2, 5, 5, 1),
    ...              'target': np.random.randn(2, 5, 5, 1),
    ...              'learning_rate': 0.001}
    >>> print('{0:.5f}'.format(net.eval(['loss'], feed_dict)['loss']))
    1.58611
    >>> net.train(feed_dict)
    >>> print('{0:.5f}'.format(net.eval(['loss'], feed_dict)['loss']))
    1.57963
    >>> print('{0:.5f}'.format(net.eval(['loss'], feed_dict)['loss']))
    1.57963

    Save and restore:

    >>> net.save('/tmp/sample_net/saved')
    >>> net_restored = SampleNet.from_file('/tmp/sample_net/saved')
    >>> print('{0:.5f}'.format(net_restored.eval(['loss'], feed_dict)['loss']))
    1.57963
    >>> net.train(feed_dict)
    >>> net_restored.train(feed_dict)
    >>> print('{0:.5f}'.format(net_restored.eval(['loss'], feed_dict)['loss']))
    1.57318
    >>> print('{0:.5f}'.format(net.eval(['loss'], feed_dict)['loss']))
    1.57318
    """

    _placeholder_prefix = 'placeholder'
    _saver_prefix = 'saver'
    _meta_graph_suffix = '.meta'
    _summary_modes = {
        'SIMPLE': GraphKeys.SIMPLE_SUMMARIES,
        'IMAGE': GraphKeys.IMAGE_SUMMARIES,
        # NOTE(daeyun): This also runs a training step.
        'TRAIN_UPDATE_RATIO': GraphKeys.TRAIN_UPDATE_SUMMARIES,
        'ALL': GraphKeys.SUMMARIES,
    }

    @classmethod
    @tc.typecheck
    def from_file(cls, restore_path: str, summary_dir: tc.optional(str) = None):
        """
        A factory method that returns an instance of a previously saved model.

        :param restore_path: The path used to save the model.
        :param summary_dir: Output path for summary files.
        :return: NNModel instance.
        """
        net = RestoredNNModel(build=False, summary_dir=summary_dir)
        net.restore(restore_path=restore_path)
        return net

    @staticmethod
    @tc.typecheck
    def summary_keys(modes: tc.any(tc.seq_of(str), str) = ('SIMPLE',)) -> tc.seq_of(str):
        """
        Returns a list of Graph collection keys.

        :param modes: Can be a string or sequence.
        :return: List of Tensorflow Graph collection keys.
        """
        if isinstance(modes, str):
            modes = [modes]

        keys = [NNModel._summary_modes['ALL']]
        for mode in modes:
            if mode not in NNModel._summary_modes:
                raise ValueError('Unrecognized summary mode: {}'.format(mode))
            keys.append(NNModel._summary_modes[mode])

        return sorted(list(set(keys)))

    @tc.typecheck
    def __init__(self,
                 sess: tc.optional(tc.any(tf.Session, tf.InteractiveSession)) = None,
                 seed: tc.optional(int) = None,
                 build: bool = True,
                 summary_dir: tc.optional(str) = None):
        """
        Creates a new model instance.

        :param sess: A TensorFlow Session.
        :param seed: The graph-level random seed.
        :param build: If true, builds and initializes the model.
        """
        if sess is None:
            self.graph = tf.Graph()
            self.session = tf.Session(graph=self.graph, config=default_sess_config())
        else:
            self.graph = sess.graph
            self.session = sess

        self.seed = seed
        self.needs_variable_initialization = True
        if summary_dir:
            self.summary_dir = path.expanduser(summary_dir)
            if not path.isdir(self.summary_dir):
                os.makedirs(self.summary_dir)
                log.info('Created directory: %s', self.summary_dir)
            assert path.isdir(self.summary_dir)
        else:
            self.summary_dir = None

        if self.seed is not None:
            with self.graph.as_default():
                # Graph-level random seed.
                tf.set_random_seed(self.seed)

        self._summary_ops = None
        self._summary_writers = None

        if build:
            self._build_model()  # Also initializes variables.
            self._init_summaries()  # Sets self._summary_ops

    def _init_summaries(self):
        assert not self.needs_variable_initialization

        if self.summary_dir:
            with self.graph.as_default():
                self._summary_ops = {}
                for k, v in NNModel._summary_modes.items():
                    assert k not in self._summary_ops
                    self._summary_ops[k] = tf.merge_all_summaries(key=v)

            if self._summary_writers is None:
                self._summary_writers = {}

            # Graph is only added to 'train' summary file.
            self._summary_writers['train'] = self._summary_writer('train', graph=self.session.graph)

    @tc.typecheck
    def _summary_writer(self, name: str = 'eval', graph: tc.optional(tf.Graph) = None) -> tf.train.SummaryWriter:
        """
        Creates or gets a summary writer.

        :param name: Name of the subdirectory.
        :param graph: A `tf.Graph` object saved in the summary. In most use cases, only one summary writer
        would need to save this. This is only used when the summar writer does not already exist.
        :return:
        """
        if self._summary_writers is None:
            self._summary_writers = {}

        if name not in self._summary_writers:
            summary_writer_path = path.join(self.summary_dir, name)
            log.info('Creating summary writer %s at %s', name, summary_writer_path)
            self._summary_writers[name] = tf.train.SummaryWriter(summary_writer_path, graph=graph)
        return self._summary_writers[name]

    @tc.typecheck
    def summary_ops(self, modes: tc.seq_of(str) = ('SIMPLE',)) -> tc.seq_of(tf.Tensor):
        """
        Returns a list of summary Tensors.

        :param modes: Can be a string or sequence.
        :return: A list of TensorFlow summary Tensors.
        """
        if isinstance(modes, str):
            modes = [modes]

        summaries = []
        for mode in modes:
            if mode not in NNModel._summary_modes:
                raise ValueError('Unrecognized summary mode: {}'.format(mode))
            op = self._summary_ops[mode]
            if op is not None:
                summaries.append(op)

        return list(set(summaries))

    def _build_model(self):
        """
        Populates the graph using user-defined functions.
        """
        with self.graph.as_default():
            with tf.variable_scope(NNModel._placeholder_prefix):
                self.placeholders = self.default_placeholders()
                self.placeholders.extend(self._placeholders())

            # Builds the main model.
            self._model()

            with tf.variable_scope('train'):
                with tf.device('/cpu:0'):
                    # Accessed by self['train/global_step$']
                    global_step = tf.get_variable('global_step', (), initializer=tf.constant_initializer(0), dtype=tf.int64, trainable=False)
                    tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, global_step.assign_add(1))

                minimize_op = self._minimize_op()
                assert isinstance(minimize_op, tf.Operation)

                # EMA apply ops.
                update_ops = self.graph.get_collection(tf.GraphKeys.UPDATE_OPS)
                with tf.control_dependencies([minimize_op]):
                    # Accessed by self['train/step$']
                    train_op = tf.group(*update_ops, name='step')

            # Update ratios of all trainable variables.
            # Accessible by self.summary_ops['TRAIN_UPDATE_RATIO'].
            with tf.name_scope('delta'):
                # `eps` prevents dividing by zero.
                eps = 1e-8
                # A dict with names as keys.
                var_norms = nn_ops.trainable_variable_norms(name='weight_norms')
                with tf.control_dependencies(var_norms):
                    with tf.control_dependencies([tf.group(train_op, name='train_wait')]):
                        updated_var_norms = nn_ops.trainable_variable_norms(name='updated_weight_norms')
                        for var_name, updated_norm in updated_var_norms.items():
                            prev_norm = var_norms[var_name]
                            tag = 'delta/' + var_name.replace(':0', '').replace(':', '_')
                            with tf.name_scope(tag) as subscope:
                                delta = tf.abs(tf.div((prev_norm - updated_norm), prev_norm + eps), name=subscope)
                                self.scalar_summary(tag, delta, collections=NNModel.summary_keys('TRAIN_UPDATE_RATIO'))

            with tf.name_scope('trainable_var_histograms'):
                for v in tf.trainable_variables():
                    var_name = v.name
                    with tf.get_default_graph().colocate_with(v):
                        v = tf.identity(v)
                        tag = 'trainable/' + var_name.replace(':0', '').replace(':', '_')
                        self.historgram_summary(tag=tag, values=v, collections=NNModel.summary_keys('TRAIN_UPDATE_RATIO'))

            self.saver = tf.train.Saver(
                name=self._saver_prefix,
                max_to_keep=10,
                keep_checkpoint_every_n_hours=0.5,
            )

        self.initialize()

    @tc.typecheck
    def default_placeholders(self) -> tc.seq_of(tf.Tensor):
        """
        Default placeholders required for all models. They can be accessed
        through `get` or `__getitem__`.

        For example:
        ::
            x = self.get('placeholder/learning_rate')
            x = self['placeholder/learning_rate']

            # Equivalent if no other names contain 'learning_rate'.
            x = self['learning_rate']

        :return: List of placeholder tensors.
        """
        return [
            tf.placeholder(tf.float32, name='learning_rate'),
            tf.placeholder_with_default(tf.constant(False, name='kFalse'), shape=(), name='is_training')
        ]

    @abc.abstractmethod
    def _placeholders(self) -> nn_types.Tensors:
        """
        Placeholders defined by the user.
        Side effect: Populates self.graph.

        :return: List of placeholder tensors.
        """
        return

    @abc.abstractmethod
    def _model(self):
        """
        TensorFlow model defined by the user. Return value is ignored.
        Side effect: Populates self.graph.
        """
        return

    @abc.abstractmethod
    def _minimize_op(self) -> tf.Operation:
        """
        Loss minimization operation defined by the user.
        Side effect: Populates self.graph.

        For example:
        ::
            def _minimize_op(self):
                loss = tf.reduce_mean((self._out - self['target']) ** 2, name='loss')
                train_op = tf.train.AdamOptimizer(self['learning_rate']).minimize(loss)
                return train_op

        :return: A TensorFlow Operation.
        """
        return

    def initialize(self):
        """
        Initializes all TensorFlow variables. Not needed when restoring from a file.
        """
        if self.needs_variable_initialization:
            with self.graph.as_default():
                self.session.run(tf.initialize_all_variables())
                self.needs_variable_initialization = False

    @tc.typecheck
    def restore(self, restore_path: str):
        """
        Restores a previously saved model. Should be called from `NNModel.from_file`.

        :param restore_path: The path used to save the model.
        """
        # TODO(daeyun): Make this a private method. Or refactor initialization logic.
        with self.graph.as_default():
            with self.session.as_default():
                self.saver = tf.train.import_meta_graph(restore_path + self._meta_graph_suffix)
                self.saver.restore(tf.get_default_session(), restore_path)
                log.info("Restored model from %s", restore_path)
                self.needs_variable_initialization = False
                self._init_summaries()

    @tc.typecheck
    def save(self, save_path: str):
        """
        Saves variables to a file.

        :param save_path: Path to the checkpoint file.
        """
        save_path = path.expanduser(save_path)

        assert not path.isdir(save_path) and not save_path.endswith('/'), 'save_path must be a file: {}'.format(save_path)

        dirpath = path.dirname(save_path)
        if not path.isdir(dirpath):
            log.info('mkdir %s', dirpath)
            os.makedirs(dirpath)

        with self.graph.as_default():
            self.saver.export_meta_graph(save_path + self._meta_graph_suffix, as_text=True)
            save_path_out = self.saver.save(self.session, save_path, write_meta_graph=False)
            log.info("Model saved to file: %s" % save_path_out)

    @tc.typecheck
    def train(self, feed_dict: dict, summary_modes: tc.optional(tc.any(str, tc.seq_of(str))) = None):
        """
        Runs a training step.

        :param feed_dict: A dictionary that maps graph elements to values. Keys can be regular expressions
        or placeholder objects.
        :param summary_modes: A list of summary modes, 'SIMPLE', 'ALL', 'IMAGE', etc. Can be empty (default).
        """
        self.eval([], feed_dict=feed_dict, is_training=True, summary_modes=summary_modes)

    @tc.typecheck
    def eval(self,
             values_or_patterns: tc.optional(tc.seq_of(tc.any(nn_types.Value, str))) = None,
             feed_dict: tc.optional(dict) = None,
             collection_keys: tc.optional(tc.seq_of(str)) = None,
             summary_modes: tc.optional(tc.seq_of(str)) = None,
             summary_writer_name: tc.optional(str) = None,
             is_training: bool = False) -> dict:
        """
        Evaluates TensorFlow Operations.

        :param values_or_patterns: Similar to the `fetches` argument of `tf.Session.run`. This can
        also be regex patterns. Must be a list.
        :param feed_dict: A dictionary that maps graph elements to values. Keys can be regular expressions
        or placeholder objects.
        :param collection_keys: All values in the given collections will be added to `fetches`.
        :param summary_modes: A list of summary modes, 'SIMPLE', 'ALL', 'IMAGE', etc. Can be empty (default).
        :param summary_writer_name: If None, default is 'train' if `is_training` is True, 'eval' otherwise.
        If this is a new name, a summary writer will be created.
        :param is_training: If true, executes a training step.
        :return: A dictionary that maps `tensors_or_patterns` to evaluated values.
        """
        assert not self.needs_variable_initialization, 'Variables are not initialized.'
        if values_or_patterns is None:
            values_or_patterns = []
        if collection_keys is None:
            collection_keys = []
        if feed_dict is None:
            feed_dict = {}
        if summary_modes is None:
            summary_modes = []
        else:
            if len(summary_modes) == 0:
                # TODO(daeyun): Log only once.
                log.warn('Non-default empty `summary_modes`. This was probably not expected.')

        # TODO(daeyun): If is_training is false and `fetches` contains an op that updates variables, e.g. minimizer, raise exception.

        names = []
        new_feed_dict = {}
        for k, v in feed_dict.items():
            if isinstance(k, str):
                new_feed_dict[self[k]] = v
                names.append(self[k].name)
            elif isinstance(k, tf.Tensor):
                new_feed_dict[k] = v
                names.append(k.name)
            else:
                raise ValueError('Unexpected key in feed_dict: {}'.format(k))

        fetches = []
        for item in values_or_patterns:
            if isinstance(item, str):
                try:
                    fetches.append(self[item])
                except:
                    raise ValueError('Unidentifiable pattern: {}'.format_map(item))
            else:
                fetches.append(item)

        for collection_key in collection_keys:
            fetches.extend(self.graph.get_collection(collection_key))

        if is_training:
            assert self['learning_rate'].name in names, \
                'learning_rate should be in feed_dict when is_training is True.'

            # There is a race condition here, but it does not matter in most use cases.
            # `train/step` will update any moving average variables as well.
            fetches.append(self['train/step$'])

            if self['is_training'] not in new_feed_dict:
                new_feed_dict[self['is_training']] = True
        else:
            if self['is_training'] not in new_feed_dict:
                new_feed_dict[self['is_training']] = False

        if self._summary_ops is None:
            # summary_dir is not provided.
            assert len(summary_modes) == 0

        for summary_mode in summary_modes:
            if summary_mode not in self._summary_ops:
                raise ValueError('Unrecognized summary mode: {}'.format(summary_mode))

        summary_ops = self.summary_ops(summary_modes) if self._summary_writers else []
        summary_op_fetch_indices = (len(fetches), len(fetches) + len(summary_ops))
        if summary_ops:
            fetches.extend(summary_ops)

        assert len(fetches) > 0, '`fetches` cannot be empty.'

        with self.graph.as_default():
            out_eval = self.session.run(fetches, new_feed_dict)

        if summary_ops:
            if summary_writer_name is None:
                summary_writer_name = 'train' if is_training else 'eval'
            assert isinstance(summary_writer_name, str)
            writer = self._summary_writer(summary_writer_name)
            global_step = self.global_step()

            summary_results = out_eval[summary_op_fetch_indices[0]:summary_op_fetch_indices[1]]
            for summary_result in summary_results:
                assert isinstance(summary_result, bytes)
                writer.add_summary(summary_result, global_step)

            # TODO(daeyun): Listen for a keypress event to signal flush.
            writer.flush()

        # `tensors_or_patterns` and `out_eval` may not be the same size.
        results = {}
        for name, result in zip(values_or_patterns, out_eval):
            if result is not None:
                results[name] = result
        return results

    @tc.typecheck
    def __getitem__(self, pattern: str) -> nn_types.ValueOrOperation:
        """
        Same as `get`. Returns a Variable or Tensor whose name uniquely matches the pattern.
        :param pattern: A regular expression pattern.
        :return: Matched variable or tensor.
        """
        return self.get(pattern)

    @property
    @tc.typecheck
    def variables(self) -> nn_types.Variables:
        """
        Returns all TensorFlow Variables defined in the graph.

        :return: All variables in the graph.
        """
        return self.graph.get_collection(tf.GraphKeys.VARIABLES)

    @property
    @tc.typecheck
    def activations(self) -> nn_types.Tensors:
        """
        Returns layer activation tensors.

        :return: Activation tensors.
        """
        return self.graph.get_collection(tf.GraphKeys.ACTIVATIONS)

    @property
    @tc.typecheck
    def ops(self) -> nn_types.Operations:
        """
        Returns all TensorFlow Operations in the graph that do not have an output value.

        :return: Operations that do not have an output value.
        """
        out = []
        for op in self.graph.get_operations():
            if len(op.outputs) == 0:
                out.append(op)
        return out

    @property
    @tc.typecheck
    def all_values(self) -> nn_types.ValuesOrOperations:
        """
        Returns all TensorFlow Variables or Operations defined in the graph.

        :return: All variables and operations in the graph.
        """
        unique = {}
        for value in self.ops:
            unique[value.name] = value

        # Includes placeholders.
        for op in self.graph.get_operations():
            for value in op.outputs:
                unique[value.name] = value

        for value in self.activations:
            unique[value.name] = value

        for value in self.variables:
            unique[value.name] = value

        return list(unique.values())

    @memoize
    @tc.typecheck
    def count(self, pattern: tc.optional(str) = None) -> int:
        """
        Returns the numbers of values whose name matches the pattern.

        :param pattern: A regular expression pattern.
        :return: Number of matching values.
        """
        return len(match_names(self.all_values, pattern))

    @memoize
    @tc.typecheck
    def collection(self, collection: str) -> nn_types.Values:
        """
        Retrieves values by a collection key.

        :param collection: A graph collection key. Must exist.
        :return: All values in the collection.
        """
        # TODO(daeyun): Get unique values in multiple collections.
        assert collection in self.graph.get_all_collection_keys()
        return self.graph.get_collection(collection)

    @memoize
    @tc.typecheck
    def get(self, pattern: str, prefix: tc.optional(str) = None, suffix: tc.optional(str) = None) -> nn_types.ValueOrOperation:
        """
        Returns a variable or tensor whose name uniquely matches the pattern. If `pattern` is not
        found, tries again with `pattern+':0$'`. Same as `self[pattern]`.

        :param pattern: A regular expression pattern.
        :param prefix: Prefix for `pattern`. Defaults to ``.*?``.
        :param suffix: Suffix for `pattern`. Defaults to ``.*?``.
        :return: Matched variable or tensor.
        :raises ValueError: If pattern matches more than one item.
        """
        matched = match_names(self.all_values, pattern, prefix=prefix, suffix=suffix)
        if len(matched) != 1:
            try:
                assert not pattern.endswith(r':0$')
                return self.get(pattern + r':0$', prefix=prefix, suffix='')
            except Exception as ex:
                raise ValueError('len(matched) = {}\npattern: {} {} {}\nmatched:\n{}'.format(
                    len(matched), prefix, pattern, suffix, '\n'.join([item.name for item in matched])))
        return matched[0]

    @memoize
    @tc.typecheck
    def sorted_values(self, pattern: tc.optional(str) = None) -> nn_types.ValuesOrOperations:
        """
        Topologically sorted Tensors, Operations, or Variables.

        :param pattern: A regular expression pattern. Defaults to ``.*?``.
        :return: List of Tensors in topological order.
        """
        assert not self.needs_variable_initialization

        names = sort_tensors(match_names(self.ops, pattern))
        return [self.get(name, prefix='^', suffix='$') for name in names]

    @memoize
    @tc.typecheck
    def last(self, pattern: tc.optional(str) = None):
        """
        Last item in `self.sorted_values`. Guaranteed to not have any values that depend on it.
        There may be more than one such value. Returns only one of them.

        :param pattern: A regular expression pattern. Defaults to ``.*?``.
        :return: Last item in `self.sorted_values`.
        """
        assert not self.needs_variable_initialization

        values = self.sorted_values(pattern=pattern)
        assert len(values) > 0
        return values[-1]

    def global_step(self) -> int:
        """
        Returns the global training step.

        :return: The global step value.
        """
        return tf.train.global_step(self.session, self['train/global_step'])

    @tc.typecheck
    def write_simple_summary(self, tag: str, value: numbers.Real, summary_writer_name: str) -> tf.train.SummaryWriter:
        """
        Writes a TensorFlow summary containing a numeric value.

        :param tag: A `string` label that shows up on TensorBoard.
        :param value: A number.
        :param summary_writer_name: If this is a new name, a summary writer will be created.
        Typical names are 'train', 'eval', 'test_my_experiment'.
        :return:
        """
        summary = tf.Summary(value=[
            tf.Summary.Value(tag=tag, simple_value=value),
        ])
        writer = self._summary_writer(summary_writer_name)
        global_step = self.global_step()
        writer.add_summary(summary, global_step)
        return writer

    @tc.typecheck
    def image_summary(self, tag: str, value: tf.Tensor, max_images: int = 3, collections: tc.seq_of(str) = (), name: tc.optional(str) = None) -> tf.Tensor:
        """
        A wrapper around `tf.image_summary` that prints out logs and adds to the image collection.

        :param tag: A scalar `Tensor` of type `string`. Used to build the `tag` of the summary values.
        :param value: A 4-D `uint8` or `float32` `Tensor` of shape `[batch_size, height, width, channels]` where `channels` is 1, 3, or 4.
        :param max_images: Max number of batch elements to generate images for.
        :param name: Will be the same as `tag` by default.
        :param collections: Graph collection keys to which this operation is added. 'image_summaries' is included by default.
        :return: A scalar `Tensor` of type `string`. The serialized `Summary` protocol buffer.
        """
        if name is None:
            name = tag + '_summary'
        shape = value.get_shape().as_list()
        assert len(shape) == 4
        assert shape[-1] in [1, 3, 4]
        collections = list(set(list(collections) + NNModel.summary_keys('IMAGE')))
        log.info('Adding image summary tag %s of shape %s to collections %s. max_images: %d', tag, shape, ','.join(collections), max_images)
        summary_op = tf.image_summary(tag, tensor=value, max_images=max_images, collections=collections, name=name)
        return summary_op

    @tc.typecheck
    def scalar_summary(self, tag: str, value: tf.Tensor, collections: tc.seq_of(str) = (), name: tc.optional(str) = None) -> tf.Tensor:
        """
        A wrapper around `tf.scalar_summary` that prints out logs and adds to the `all` collection.
        """
        if name is None:
            name = tag + '_summary'
        collections = list(set(list(collections) + NNModel.summary_keys('ALL')))
        log.info('Adding scalar summary tag %s to collections %s.', tag, ','.join(collections))
        summary_op = tf.scalar_summary(tag, value, collections=collections, name=name)
        return summary_op

    @tc.typecheck
    def historgram_summary(self, tag: str, values: tc.any(tf.Tensor, nn_types.Tensors), collections: tc.seq_of(str) = (), name: tc.optional(str) = None) -> tf.Tensor:
        """
        A wrapper around `tf.histogram_summary` that prints out logs and adds to the `all` collection.
        """
        if name is None:
            name = tag + '_summary'
        if isinstance(values, tf.Tensor):
            value = values
        else:
            with tf.name_scope('histogram_summary_value') as scope:
                value = tf.concat(0, [tf.reshape(v, [-1]) for v in values], name=scope)
        collections = list(set(list(collections) + NNModel.summary_keys('ALL')))
        log.info('Adding histogram summary tag %s to collection %s', tag, ','.join(collections))
        summary_op = tf.histogram_summary(tag=tag, name=name, values=value, collections=collections)
        return summary_op

    @tc.typecheck
    def collect(self, key: str, values: tc.any(nn_types.Value, nn_types.Values)):
        """
        Adds values to a graph collection.

        :param key: Graph key.
        :param values: A value or a list of values.
        """
        if not isinstance(values, typing.Sequence):
            values = [values]
        with self.graph.as_default():
            for value in values:
                log.info('Adding %s to collection %s', value, key)
                tf.add_to_collection(key, value)


class RestoredNNModel(NNModel):
    """
    Container for models restored from file. This can only be instantiated from `NNModel.from_file`.
    """

    def _model(self):
        # Restored models should not try to build a new graph.
        raise NotImplementedError()

    def _minimize_op(self):
        raise NotImplementedError()

    def _placeholders(self):
        raise NotImplementedError()
