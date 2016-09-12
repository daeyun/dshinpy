"""
Helpers for managing TensorFlow neural net models.
"""
import abc
import contextlib
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
from dshin.nn import graph
from dshin.nn import ops as nn_ops
from dshin.nn import types as nn_types

memoize = functools.lru_cache(maxsize=2048, typed=True)


class QueuePlaceholder(object):
    def __init__(self, dtype, shape, name, is_file):
        self.dtype = dtype
        self.shape = shape
        self.name = name
        self.is_file = is_file


class QueueComponents(object):
    def __init__(self, name, placeholders, tensors, enqueue_op=None, queue=None):
        self.name = name
        self.placeholders = placeholders
        self.tensors = tensors
        self.enqueue_op = enqueue_op
        self.queue = queue


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
                suffix: tc.optional(str) = None,
                return_final_pattern=False):
    """
    Filters TensorFlow graph objects by regular expression.

    :param values: A list of Variables, Tensors, or Operations.
    :param pattern: A regular expression pattern.
    :param prefix: A string prepended to `pattern`. Defaults to ``^(.*/)?``.
    :param suffix: A string appended to `pattern`. Defaults to ``(/.*)?$`.
    :param return_final_pattern: If `True`, return a tuple (list of matched values, regex pattern).
    :return: A list of matched graph objects. And optionally the regex pattern used for the search.
    """
    if pattern is None:
        pattern = r'.*?'
    if prefix is None:
        if pattern.startswith('^'):
            prefix = ''
        else:
            prefix = r'^(.*/)?'
    if suffix is None:
        if pattern.endswith('$'):
            suffix = ''
        else:
            suffix = r'(/.*)?$'

    final_pattern = ''.join([prefix, pattern, suffix])
    matched = []
    for item in values:
        if re.match(final_pattern, item.name):
            matched.append(item)

    if return_final_pattern:
        return matched, final_pattern
    return matched


def ensure_list_or_tuple(value_or_values):
    if isinstance(value_or_values, (list, tuple)):
        return value_or_values
    return tuple(value_or_values)


import functools


@functools.lru_cache()
def _check_placeholder_queue_coverage(placeholders: frozenset, return_missing=False):
    assert isinstance(placeholders, frozenset)
    queue_placeholders = {}
    for placeholder in placeholders:
        assert isinstance(placeholder, tf.Tensor)
        if 'placeholderwithdefault' in placeholder.op.type.lower():
            for queue_output in placeholder.op.inputs:
                if 'dequeue' in queue_output.op.type.lower():
                    names = []
                    for output in queue_output.op.outputs:
                        for placeholder_op in output.consumers():
                            if placeholder_op.type == placeholder.op.type:
                                for value in placeholder_op.outputs:
                                    if type(value) == type(queue_output):
                                        names.append(value.name)
                                        queue_placeholders[value.name] = value
                    if names:
                        # Sanity check. Making sure `all_names` includes input names.
                        assert placeholder.name in names

    all_names = frozenset(queue_placeholders.keys())
    names = frozenset(item.name for item in placeholders)
    missing = all_names - names

    if return_missing:
        return len(missing) == 0, missing
    return len(missing) == 0


def check_feed_queue_coverage(feed_dict):
    optional_placeholders = frozenset(filter(lambda x: 'placeholderwithdefault' in x.op.type.lower(),
                                             feed_dict.keys()))
    is_complete, missing = _check_placeholder_queue_coverage(optional_placeholders, True)
    if not is_complete:
        input_names = '\n'.join([k.name for k in feed_dict.keys()])
        missing_names = '\n'.join([name for name in missing])
        raise ValueError(('Error: Incomplete list of optional placeholders.\n'
                          'Given:\n{}\n\nMissing:\n{}').format(input_names, missing_names))


@tc.typecheck
def default_sess_config(mem: float = 0.95,
                        log_device_placement: bool = False,
                        device_filters=None) -> tf.ConfigProto:
    conf = tf.ConfigProto(device_filters=device_filters)
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

    >>> from dshin import nn
    >>> class SampleNet(nn.utils.NNModel):
    ...     def _model(self):
    ...         input_placeholder = self['input']
    ...         target_placeholder = self['target']
    ... 
    ...         out = nn.ops.conv2d(input_placeholder, n_out=1, use_bias=False)
    ...         out = nn.ops.batch_norm(out, is_trainable=True, is_local=True)
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
    1.55977
    >>> net.train(feed_dict)
    >>> print('{0:.5f}'.format(net.eval(['loss'], feed_dict)['loss']))
    1.55425
    >>> print('{0:.5f}'.format(net.eval(['loss'], feed_dict)['loss']))
    1.55425

    Save and restore:

    >>> net.save('/tmp/sample_net/saved')
    >>> net_restored = SampleNet.from_file('/tmp/sample_net/saved')
    >>> print('{0:.5f}'.format(net_restored.eval(['loss'], feed_dict)['loss']))
    1.55425
    >>> net.train(feed_dict)
    >>> net_restored.train(feed_dict)
    >>> print('{0:.5f}'.format(net_restored.eval(['loss'], feed_dict)['loss']))
    1.54875
    >>> print('{0:.5f}'.format(net.eval(['loss'], feed_dict)['loss']))
    1.54875
    """

    _placeholder_prefix = 'placeholder'
    _producer_queue_prefix = 'queue'
    _consumer_queue_prefix = 'worker_queue'
    _saver_prefix = 'saver'
    _meta_graph_suffix = '.meta'
    _worker_batch_size_placeholder_name = 'batch_size'
    _global_step_variable_name = 'global_step'
    _global_queue_buffer_size = 2000
    _worker_queue_buffer_size = 100
    _worker_queue_num_threads = 10
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
                 summary_dir: tc.optional(str) = None,
                 input_queue_device=None):
        """
        Creates a new model instance.

        :param sess: A TensorFlow Session.
        :param seed: The graph-level random seed.
        :param build: If `True`, builds and initializes the model.
        :param summary_dir: Path to the summary directory. Created if not exists.
        :param input_queue_device: Device for the input producer queue operations. Defaults to the `global_step` Variable's device
        """
        self.seed = seed
        self.needs_variable_initialization = True
        self.input_queue_device = input_queue_device

        if sess is None:
            self.graph = tf.Graph()
            self.session = tf.Session(graph=self.graph, config=default_sess_config())
        else:
            self.graph = sess.graph
            self.session = sess

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

        self.queue_runner_threads = self._run_queue_runners()

    def _run_queue_runners(self):
        self.coordinator = tf.train.Coordinator()
        return tf.train.start_queue_runners(self.session, self.coordinator)

    def shutdown(self):
        assert not self.needs_variable_initialization
        self.coordinator.request_stop()

        # Unblock blocking queue operations.
        self.session.run(self['{}/close'.format(NNModel._consumer_queue_prefix)])
        self.session.run(self['{}/close'.format(NNModel._producer_queue_prefix)])

        self.coordinator.join(stop_grace_period_secs=10)

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

    @tc.typecheck
    def _build_producer_queue(self,
                              placeholder_specs: tc.seq_of(QueuePlaceholder),
                              queue_name: str,
                              queue_device: tc.any(str, typing.Callable)):
        """
        Builds an input queue that atomically operates on the set of values specified by `QueuePlaceholder` objects.
        In order to use this as a global queue in a distributed model, all workers should use the same `queue_device`,
        e.g. `/job:ps/task:0`, when building the shared queue.

        Ideally, this should not transmit a lot of data (otherwise there could be a network IO bottleneck), and large arrays
        should be passed as a filename instead.

        Side effect: Creates new placeholders in the `placeholder/{queue_name}/` scope. They are used in the enqueue operation.

        :param placeholder_specs: A list of `QueuePlaceholder` objects. If `is_file` is `True`, the corresponding placeholder will have
        dtype `tf.string`. And the corresponding output tensor will have dtype `placeholder_spec.dtype`. All tensor shapes must have `None`
        as the first dimension.
        :param queue_name: A unique name. Used as a prefix for the placeholder names. e.g. `placeholder/{queue_name}/{tensor_name}`
        :param queue_device: Device for the queue operations.
        :return: A `QueueComponents` with the following fields:
            name: The name of the created queue. e.g. `{queue_name}`
            enqueue_op: An `enqueue_many` operation that enqueues placeholder values.
            placeholders: A list of placeholders used to insert new values to the queue. `placeholders[i]` corresponds
                          to `placeholder_specs[i]`. If `placeholder_specs[i].is_file` is `True`, the type will be a list of strings.
            tensors: A dictionary that maps placeholder names to output tensors derived from the dequeued values.
                     Shape of the tensors will be `placeholder_spec.shape[1:]`
            queue: The `tf.FIFOQueue` object. Not needed in most use cases.
        """
        assert len(placeholder_specs) > 0
        names, dtypes, shapes, placeholders = [], [], [], []
        input_specs_by_name = {item.name: item for item in placeholder_specs}
        for item in placeholder_specs:
            assert item.shape[0] is None

            if item.is_file:
                dtype = tf.string
                shape = [item.shape[0]]
            else:
                dtype = item.dtype
                shape = item.shape

            existing_placeholders = self._find_placeholders()

            # Placeholders for global enqueue ops.
            # Accessed by `self['placeholder/{queue_name}/{tensor_name}']`
            with graph.abs_name_scope(NNModel._placeholder_prefix):
                name = '{}/{}'.format(queue_name, item.name)
                assert name not in existing_placeholders
                placeholder = tf.placeholder(dtype=dtype, shape=shape, name=name)

            names.append(item.name)
            dtypes.append(dtype)
            shapes.append(shape[1:])
            placeholders.append(placeholder)

        with tf.variable_scope(queue_name):
            with tf.device(queue_device):
                queue = tf.FIFOQueue(self._global_queue_buffer_size, dtypes=dtypes, shapes=shapes,
                                     names=names, shared_name=queue_name, name=queue_name)
                enqueue_tensors = {name: placeholder for name, placeholder in zip(names, placeholders)}

                # `enqueue_op` is an atomic operation. Values for all of `placeholder/{queue_name}/*` need to be specified at runtime.
                enqueue_op = queue.enqueue_many(enqueue_tensors, name='enqueue')

                with graph.abs_name_scope(queue_name):
                    # `queue/size:0`.
                    queue.size(name='size')
                    queue.close(True, name='close')

            dequeue_tensors = queue.dequeue(name='dequeue')
            assert isinstance(dequeue_tensors, dict)

        tensors = {}
        for name, value in dequeue_tensors.items():
            input_spec = input_specs_by_name[name]
            if input_spec.is_file:
                # This should happen on the worker's device.
                value = nn_ops.npz_to_tensor(value, dtype=input_spec.dtype, shape=input_spec.shape[1:])
            else:
                # Sanity check.
                value.get_shape().assert_is_compatible_with(input_spec.shape[1:])
            tensors[name] = value

        return QueueComponents(name=queue.name, enqueue_op=enqueue_op, placeholders=placeholders, tensors=tensors, queue=queue)

    def _build_batch_tensors(self, name, batch_size, producer_queue_components: QueueComponents, placeholder_name_prefix=None):
        """
        Returns a batched tensor that pulls and concatenates values from the producer queue upon evaluation.

        Values dequeued from the producer will be buffered in a consumer queue. A `QueueRunner` for this queue will be added to the
        current `Graph`'s `QUEUE_RUNNERS` collection.

        This function also creates new placeholders in the `placeholder/` scope for directly feeding tensors instead of dequeueing
        from the queue.

        :param name: A name for the operations.
        :param batch_size: An integer, Tensor, or Variable batch size pulled from the consumer queue in the dequeue operation.
        :param producer_queue_components: The components of the queue feeding into this queue. See `QueueComponents`.
        :param placeholder_name_prefix: Optional name prefix for the placeholders created by this function.
        :return: A `QueueComponents` with the following fields:
            name: The name of the created queue. e.g. `{name}/fifo_queue`
            placeholders: A list of placeholders used to insert new values to the queue. `placeholders[i]` corresponds
                          to `placeholder_specs[i]`. If `placeholder_specs[i].is_file` is `True`, the type will be a list of strings.
            tensors: A dictionary that maps placeholder names to output tensors derived from the dequeued values.
                     Shape of the tensors will be `placeholder_spec.shape[1:]`
            queue: The `tf.FIFOQueue` object. Not needed in most use cases.
        """
        with graph.collect_values(tf.GraphKeys.QUEUE_RUNNERS) as queue_runners:
            batch_tensors = tf.train.batch(producer_queue_components.tensors,
                                           batch_size=batch_size,
                                           num_threads=self._worker_queue_num_threads,
                                           capacity=self._worker_queue_buffer_size,
                                           allow_smaller_final_batch=True,
                                           name=name)

        assert len(queue_runners) == 1
        queue = queue_runners[0].queue

        with graph.abs_name_scope(name):
            # `worker_queue/size:0`.
            queue.size('size')
            queue.close(True, name='close')

        existing_placeholders = self._find_placeholders()

        placeholders = []
        assert isinstance(batch_tensors, dict)
        for key, tensor in batch_tensors.items():
            assert isinstance(tensor, tf.Tensor)
            shape = tensor.get_shape()
            assert shape[0].value is None
            for dim in shape[1:]:
                assert dim.value is not None
            with graph.abs_name_scope(NNModel._placeholder_prefix):
                if placeholder_name_prefix is not None:
                    key = placeholder_name_prefix + key
                assert key not in existing_placeholders
                placeholder = tf.placeholder_with_default(tensor, shape=shape, name=key)
            placeholders.append(placeholder)

        return QueueComponents(name=queue.name, placeholders=placeholders, tensors=batch_tensors, queue=queue)

    @contextlib.contextmanager
    def _placeholder_scope(self):
        with tf.name_scope(NNModel._placeholder_prefix + '/') as scope:
            yield scope

    def _build_model(self):
        """
        Populates the graph using user-defined functions.
        """
        with self.graph.as_default():
            # Accessed by self['global_step']
            global_step = tf.get_variable(NNModel._global_step_variable_name, tuple(),
                                          initializer=tf.constant_initializer(0), dtype=tf.int64, trainable=False)
            if self.input_queue_device is None:
                self.input_queue_device = global_step.device

            with graph.abs_name_scope(NNModel._placeholder_prefix):
                self.default_placeholders()
                user_placeholders = self._placeholders()

            self.batch_size = self._find_placeholders()[self._worker_batch_size_placeholder_name]
            assert isinstance(self.batch_size, tf.Tensor)

            placeholder_specs = list(filter(lambda item: isinstance(item, QueuePlaceholder), user_placeholders))
            if placeholder_specs:
                queue_components = self._build_producer_queue(placeholder_specs, self._producer_queue_prefix, self.input_queue_device)
                self._build_batch_tensors(name=self._consumer_queue_prefix, batch_size=self.batch_size,
                                          producer_queue_components=queue_components)

            # Builds the main model.
            self._model()

            with tf.variable_scope('train'):
                with tf.device('/cpu:0'):
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
                name=NNModel._saver_prefix,
                max_to_keep=10,
                keep_checkpoint_every_n_hours=0.5,
            )

        self.initialize()

    def _find_placeholders(self):
        placeholders = {}
        for op in tf.get_default_graph().get_operations():
            try:
                for tensor in op.outputs:
                    name = self._parse_placeholder_name(tensor)
                    placeholders[name] = tensor
            except:
                continue
        return placeholders

    def _parse_placeholder_name(self, tensor):
        m = re.match(r'^{}/([^:]+):0$'.format(self._placeholder_prefix), tensor.name)
        if m is None:
            raise ValueError('Invalid tensor name: {}'.format(tensor.name))
        return m.group(1)

    @tc.typecheck
    def default_placeholders(self) -> tc.seq_of(tf.Tensor):
        """
        Default placeholders required for all models. They can be accessed
        through `get` or `__getitem__`.

        For example:
        ::
            x = self.get('placeholder/learning_rate')
            x = self['placeholder/learning_rate']

            # This is equivalent to the above if no other names contain 'learning_rate'.
            x = self['learning_rate']

        :return: List of placeholder tensors.
        """
        return [
            tf.placeholder(tf.float32, shape=(), name='learning_rate'),
            tf.placeholder_with_default(tf.constant(False, name='kFalse'), shape=(), name='is_training'),
            tf.placeholder(tf.int32, shape=(), name=self._worker_batch_size_placeholder_name),
        ]

    @abc.abstractmethod
    def _placeholders(self) -> typing.Sequence[typing.Union[tf.Tensor, QueuePlaceholder]]:
        """
        Placeholders defined by the user.
        Side effect: Populates self.graph.

        :return: List of placeholder tensors or QueuePlaceholder objects.
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
                self.saver = tf.train.import_meta_graph(restore_path + NNModel._meta_graph_suffix)
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
            self.saver.export_meta_graph(save_path + NNModel._meta_graph_suffix, as_text=True)
            save_path_out = self.saver.save(self.session, save_path, write_meta_graph=False)
            log.info("Model saved to file: %s" % save_path_out)

    @tc.typecheck
    def train(self,
              feed_dict: dict,
              summary_modes: tc.optional(tc.any(str, tc.seq_of(str))) = None,
              check_queue_coverage=True):
        """
        Runs a training step.

        :param feed_dict: A dictionary that maps graph elements to values. Keys can be regular expressions
        or placeholder objects.
        :param summary_modes: A list of summary modes, 'SIMPLE', 'ALL', 'IMAGE', etc. Can be empty (default).
        :param check_queue_coverage: If `True`, raise an exception when `feed` contains an optional placeholder
        and there are any remaining optional placeholders that depend on the same queue.
        """
        self.eval([], feed_dict=feed_dict, is_training=True, summary_modes=summary_modes)

    @tc.typecheck
    def enqueue(self, feed_dict: dict):
        new_feed_dict = {}
        for k, v in feed_dict.items():
            if isinstance(k, str):
                if not k.startswith(NNModel._producer_queue_prefix):
                    k = '{}/{}/{}'.format(NNModel._placeholder_prefix, NNModel._producer_queue_prefix, k)
                new_feed_dict[self[k]] = v
            elif isinstance(k, tf.Tensor):
                new_feed_dict[k] = v
            else:
                raise ValueError('Unexpected key in feed_dict: {}'.format(k))

        for k, v in new_feed_dict.items():
            assert isinstance(k, tf.Tensor)
            assert k.name.startswith('{}/{}'.format(NNModel._placeholder_prefix, NNModel._producer_queue_prefix))

        enqueue_op = self['{}/enqueue'.format(NNModel._producer_queue_prefix)]
        assert isinstance(enqueue_op, tf.Operation)
        self.session.run(enqueue_op, feed_dict=new_feed_dict)

    @tc.typecheck
    def eval(self,
             values_or_patterns: tc.optional(tc.any(tc.any(nn_types.Value, str), tc.seq_of(tc.any(nn_types.Value, str)))) = None,
             feed_dict: tc.optional(dict) = None,
             collection_keys: tc.optional(tc.seq_of(str)) = None,
             summary_modes: tc.optional(tc.seq_of(str)) = None,
             summary_writer_name: tc.optional(str) = None,
             is_training: bool = False,
             check_optional_placeholder_coverage=True):
        """
        Evaluates TensorFlow Operations.

        :param values_or_patterns: Similar to the `fetches` argument of `tf.Session.run`. This can
        also be regex patterns. Can be a list or single value.
        :param feed_dict: A dictionary that maps graph elements to values. Keys can be regular expressions
        or placeholder objects.
        :param collection_keys: All values in the given collections will be added to `fetches`.
        :param summary_modes: A list of summary modes, 'SIMPLE', 'ALL', 'IMAGE', etc. Can be empty (default).
        :param summary_writer_name: If None, default is 'train' if `is_training` is True, 'eval' otherwise.
        If this is a new name, a summary writer will be created.
        :param is_training: If `True`, executes a training step.
        :param check_queue_coverage: If `True`, raise an exception when `feed` contains an optional placeholder
        and there are any remaining optional placeholders that depend on the same queue.
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

        is_single_value = isinstance(values_or_patterns, (tf.Tensor, tf.Variable, tf.Operation, str))
        if is_single_value:
            values_or_patterns = [values_or_patterns]

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
                    raise ValueError('Unidentifiable pattern: {}'.format(item))
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

        if check_optional_placeholder_coverage:
            check_feed_queue_coverage(new_feed_dict)

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

        if is_single_value:
            return out_eval[0]
        else:
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
        Returns all TensorFlow Operations or Tensors defined in the graph.

        :return: All variables and operations in the graph.
        """
        unique = {}
        for value in self.variables:
            unique[value.name] = value

        for value in self.graph.get_operations():
            unique[value.name] = value

        # Includes placeholders.
        # NOTE: Tensors and Variables can have the same name. Tensors should have higher priority.
        for op in self.graph.get_operations():
            for value in op.outputs:
                unique[value.name] = value

        return list(zip(*sorted(list(unique.items()))))[1]

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
    def get_all(self, pattern: str, prefix: tc.optional(str) = None, suffix: tc.optional(str) = None):
        """
        Returns a list of tensors or operations whose name matches the pattern. If `pattern` is not
        found, tries again with `pattern+':0$'`. Same as `self[pattern]`.

        :param pattern: A regular expression pattern.
        :param prefix: A string prepended to `pattern`. Defaults to ``^(.*/)?``.
        :param suffix: A string appended to `pattern`. Defaults to ``(/.*)?$`.
        :return: Matched variable or tensor.
        :raises ValueError: If pattern matches more than one item.
        """
        all_values = self.all_values
        matched, final_pattern = match_names(all_values, pattern, prefix=prefix, suffix=suffix,
                                             return_final_pattern=True)

        # Base cases.
        if len(matched) == 0:
            raise ValueError('Error: pattern {} did not match any values in the graph.'.format(final_pattern))

        return matched

    @memoize
    @tc.typecheck
    def get(self, pattern: str, prefix: tc.optional(str) = None, suffix: tc.optional(str) = None) -> nn_types.ValueOrOperation:
        """
        Returns a tensor or operation whose name uniquely matches the pattern. If `pattern` is not
        found, tries again with `pattern+':0$'`. Same as `self[pattern]`.

        :param pattern: A regular expression pattern.
        :param prefix: A string prepended to `pattern`. Defaults to ``^(.*/)?``.
        :param suffix: A string appended to `pattern`. Defaults to ``(/.*)?$`.
        :return: Matched variable or tensor.
        :raises ValueError: If pattern matches more than one item.
        """
        all_values = self.all_values
        all_values_by_name = {item.name: item for item in all_values}

        if prefix is None and suffix is None:
            # If the name contains ':', i.e. a tensor or variable, check if there is an exact match.
            if ':' in pattern:
                if pattern in all_values_by_name:
                    value = all_values_by_name[pattern]
                    if isinstance(value, tf.Tensor):
                        return value

            else:
                # Check if `pattern` is a placeholder name.
                try:
                    return self.get('{}/{}:0'.format(NNModel._placeholder_prefix, pattern), prefix=None, suffix=None)
                except ValueError:
                    pass

        # Check if appending ':[0-9]+?$' works. If it does, give it priority.
        if '$' not in pattern and suffix is None:
            try:
                return self.get(pattern, prefix=prefix, suffix=':[0-9]+?$')
            except ValueError:
                pass

        matched, final_pattern = match_names(all_values, pattern,
                                             prefix=prefix, suffix=suffix, return_final_pattern=True)

        # Base cases.
        if len(matched) == 1:
            # Check if there is a value whose name uniquely matches the regex '{prefix}{pattern}{suffix}'.
            return matched[0]

        if len(matched) == 0:
            raise ValueError('Error: pattern {} did not match any values in the graph.'.format(final_pattern))

        raise ValueError('Error: pattern {} matches multiple values ({} total):\n{}'.format(
            final_pattern, len(matched), '\n'.join([item.name for item in matched])))

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
        return tf.train.global_step(self.session, self['{}:0$'.format(NNModel._global_step_variable_name)])

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
