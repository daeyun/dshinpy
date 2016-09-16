import abc
import contextlib
import functools
import numbers
import numpy as np
import os
import re
from os import path

import tensorflow as tf
import toposort
import typing

from dshin import log
from dshin.nn import graph_utils
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


class NNModel(metaclass=abc.ABCMeta):
    _producer_queue_prefix = 'queue'
    _consumer_queue_prefix = 'worker_queue'
    _saver_prefix = 'saver'
    _meta_graph_suffix = '.meta'
    _global_queue_buffer_size = 2000
    _worker_queue_buffer_size = 100
    _worker_queue_num_threads = 10

    def __init__(self, graph: tf.Graph, summary_dir=None):
        self.graph = graph
        self.needs_initialization = True

        if summary_dir:
            self.summary_dir = path.expanduser(summary_dir)
            if not path.isdir(self.summary_dir):
                os.makedirs(self.summary_dir)
                log.info('Created directory: %s', self.summary_dir)
            assert path.isdir(self.summary_dir)
        else:
            self.summary_dir = None

    @abc.abstractmethod
    def _model(self) -> tf.Tensor:
        return

    @abc.abstractmethod
    def _placeholders(self):
        return

    def _build_input_pipeline(self, server):
        assert self.needs_initialization
        assert isinstance(server, tf.train.Server)

        with tf.name_scope('placeholder/'):
            self.default_placeholders()
            user_placeholders = self._placeholders()
        variables = self.default_variables()

        assign_learning_rate = tf.assign(self.variable('learning_rate'), self.placeholder('learning_rate'),
                                         validate_shape=True, use_locking=True, name='assign_learning_rate')
        input_queue_device = self.variable('global_step').device

        placeholder_specs = list(filter(lambda item: isinstance(item, QueuePlaceholder), user_placeholders))

        if not placeholder_specs:
            return

        queue_components = self._build_producer_queue(placeholder_specs, self._producer_queue_prefix, input_queue_device)
        return queue_components

    def build(self, server, data_pipeline_only=False, seed=None, sync_replicas=True, save_model_secs=600):
        assert self.needs_initialization
        assert isinstance(server, tf.train.Server)

        self.seed = seed
        self.data_pipeline_only = data_pipeline_only

        self.task_id = graph_utils.task_id_from_server(server)
        self.job_name, self.hosts = graph_utils.job_info_from_server(server)
        self.num_replicas = len(self.hosts)
        self.cluster_def = server.server_def.cluster

        # TODO(daeyun): Prevent data processes from consuming data from the queue

        worker_device_name = "/job:{job}/task:{task}".format(job=self.job_name, task=self.task_id)
        log.info('[%s %d] Device name is %s', self.job_name, self.task_id, worker_device_name)
        device_setter = tf.train.replica_device_setter(worker_device=worker_device_name, cluster=self.cluster_def)

        with self.graph.as_default(), tf.device(device_setter):
            if seed is not None:
                # Graph-level random seed.
                tf.set_random_seed(seed)

            queue_components = self._build_input_pipeline(server)

            batch_size = self.placeholder('batch_size', fail=True)
            assert isinstance(batch_size, tf.Tensor)

            self._build_batch_tensors(name=self._consumer_queue_prefix, batch_size=batch_size,
                                      producer_queue_components=queue_components)

            # Builds the main model.
            loss = self._model()
            assert isinstance(loss, tf.Tensor), '_model() must return a Tensor for the total loss.'

            # self._init_summaries()  # Sets self._summary_ops

            optim = self._optimizer(self.variable('learning_rate'))

            with tf.variable_scope('train'):
                if sync_replicas:
                    optim = tf.train.SyncReplicasOptimizer(
                        optim,
                        replicas_to_aggregate=self.num_replicas,
                        replica_id=self.task_id,
                        total_num_replicas=self.num_replicas,
                        use_locking=False
                    )

                with tf.name_scope('minimize') as scope:
                    minimize_op = optim.minimize(loss, global_step=self.variable('global_step'))
                    # TODO(daeyun): remove this after PR is merged.
                    minimize_op = tf.group(minimize_op, name=scope)

            is_chief = self.task_id == 0

            queue_runners = []
            if is_chief:
                if sync_replicas:
                    queue_runners.append(optim.get_chief_queue_runner())
                    queue_runners.extend(self.graph.get_collection(tf.GraphKeys.QUEUE_RUNNERS))
                    init_tokens_op = optim.get_init_tokens_op()

            summary_op = tf.merge_all_summaries()
            self.saver = tf.train.Saver(
                name='saver',
                max_to_keep=10,
                keep_checkpoint_every_n_hours=0.5,
            )

            graph_utils.save_graph_text_summary(self.graph)

            init_op = tf.initialize_all_variables()

            # Create a "supervisor", which oversees the training process.
            self.supervisor = tf.train.Supervisor(is_chief=is_chief,
                                                  logdir=self.summary_dir,
                                                  init_op=init_op,
                                                  summary_op=summary_op,
                                                  saver=self.saver,
                                                  recovery_wait_secs=1,
                                                  global_step=self.variable('global_step'),
                                                  save_model_secs=save_model_secs,
                                                  )

        config = server.server_def.default_session_config

        log.info('[%s %d] Waiting for session.', self.job_name, self.task_id)
        sess = self.supervisor.prepare_or_wait_for_session(master=server.target, config=config,
                                                           start_standard_services=True)
        self.session = sess

        log.info('[%s %d] Session is ready.', self.job_name, self.task_id)

        if is_chief:
            self.supervisor.start_queue_runners(sess, queue_runners)
            assert isinstance(sess, tf.Session)
            if sync_replicas:
                sess.run(init_tokens_op)
        self.needs_initialization = False

    def _build_producer_queue(self,
                              placeholder_specs,
                              queue_name,
                              queue_device):
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
        :return: A `_QueueComponents` with the following fields:
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

            # Placeholders for global enqueue ops.
            with tf.name_scope('placeholder/'):
                name = '{}/{}'.format(queue_name, item.name)
                assert self.placeholder(name, fail=False) is None
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

                with tf.name_scope('{}/'.format(queue_name)):
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
        with graph_utils.collect_values(tf.GraphKeys.QUEUE_RUNNERS) as queue_runners:
            batch_tensors = tf.train.batch(producer_queue_components.tensors,
                                           batch_size=batch_size,
                                           num_threads=self._worker_queue_num_threads,
                                           capacity=self._worker_queue_buffer_size,
                                           allow_smaller_final_batch=True,
                                           name=name)

        assert len(queue_runners) == 1
        queue = queue_runners[0].queue

        with tf.name_scope(name + '/'):
            # `worker_queue/size:0`.
            queue.size('size')
            queue.close(True, name='close')

        placeholders = []
        assert isinstance(batch_tensors, dict)
        for key, tensor in batch_tensors.items():
            assert isinstance(tensor, tf.Tensor)
            shape = tensor.get_shape()
            assert shape[0].value is None
            for dim in shape[1:]:
                assert dim.value is not None
            with tf.name_scope('placeholder/'):
                if placeholder_name_prefix is not None:
                    key = placeholder_name_prefix + key
                assert self.placeholder(key, fail=False) is None
                placeholder = tf.placeholder_with_default(tensor, shape=shape, name=key)
            placeholders.append(placeholder)

        return QueueComponents(name=queue.name, placeholders=placeholders, tensors=batch_tensors, queue=queue)

    def _optimizer(self, learning_rate):
        return tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=1e-4)

    def _build_model(self) -> tf.Tensor:
        with self.graph.as_default():
            with tf.name_scope('placeholder/'):
                self.default_placeholders()
                user_placeholders = self._placeholders()
            variables = self.default_variables()

            assign_learning_rate = tf.assign(self.variable('learning_rate'), self.placeholder('learning_rate'),
                                             validate_shape=True, use_locking=True, name='assign_learning_rate')
            input_queue_device = self.variable('global_step').device

            batch_size = self.placeholder('batch_size', fail=True)
            assert isinstance(batch_size, tf.Tensor)

            placeholder_specs = list(filter(lambda item: isinstance(item, QueuePlaceholder), user_placeholders))
            if placeholder_specs:
                queue_components = self._build_producer_queue(placeholder_specs, self._producer_queue_prefix, input_queue_device)
                self._build_batch_tensors(name=self._consumer_queue_prefix, batch_size=batch_size,
                                          producer_queue_components=queue_components)

            # Builds the main model.
            loss = self._model()
            assert isinstance(loss, tf.Tensor), '_model() must return a Tensor for the total loss.'

        return loss

    def default_variables(self):
        return [
            tf.get_variable('global_step', shape=(), initializer=tf.constant_initializer(0, dtype=tf.int64), dtype=tf.int64, trainable=False),
            tf.get_variable('learning_rate', shape=(), initializer=tf.constant_initializer(0.001, dtype=tf.float32), dtype=tf.float32, trainable=False),
        ]

    def default_placeholders(self) -> typing.Sequence[tf.Tensor]:
        return [
            tf.placeholder(tf.float32, shape=(), name='learning_rate'),
            # tf.placeholder_with_default(tf.constant(0.001), shape=(), name='learning_rate'),
            tf.placeholder_with_default(tf.constant(False, name='kFalse'), shape=(), name='is_training'),
            tf.placeholder(tf.int32, shape=(), name='batch_size'),
        ]

    def shutdown(self):
        assert not self.needs_initialization
        self.supervisor.request_stop()

        # Unblock blocking queue operations.
        self.session.run(self.operation('{}/close'.format(NNModel._consumer_queue_prefix)))
        self.session.run(self.operation('{}/close'.format(NNModel._producer_queue_prefix)))

        self.session.close()
        log.info('[%s %d] Closed session.', self.job_name, self.task_id)
        self.supervisor.coord.join(stop_grace_period_secs=5)

    def operation(self, names, fail=True) -> tf.Operation:
        is_single = not isinstance(names, (list, tuple))
        if is_single:
            names = [names]

        results = []
        for name in names:
            if isinstance(name, tf.Operation):
                results.append(name)
                continue
            assert isinstance(name, str)

            try:
                results.append(self.graph.get_operation_by_name(name))
            except Exception as ex:
                if fail:
                    raise ex
                results.append(None)

        if is_single:
            assert len(results) == 1
            return results[0]
        return results

    def placeholder(self, name, fail=True) -> tf.Tensor:
        if ':' not in name:
            name = '{}:0'.format(name)
        if not name.startswith('placeholder/'):
            name = 'placeholder/{}'.format(name)
        try:
            value = self.graph.get_tensor_by_name(name)
            assert value.op.type in ('Placeholder', 'PlaceholderWithDefault')
            return value
        except Exception as ex:
            if fail:
                raise ex
            return None

    def variable(self, names, fail=True) -> tf.Variable:
        is_single = not isinstance(names, (list, tuple))
        if is_single:
            names = [names]

        results = []

        for name in names:
            if isinstance(name, tf.Variable):
                results.append(name)
                continue
            assert isinstance(name, str)

            if ':' not in name:
                name = '{}:0'.format(name)
            try:
                for variable in self.graph.get_collection(tf.GraphKeys.VARIABLES):
                    if variable.name == name:
                        results.append(variable)
            except Exception as ex:
                if fail:
                    raise ex
                results.append(None)

        if is_single:
            assert len(results) == 1
            return results[0]
        return results

    def placeholders(self, return_dict=True):
        placeholders = {}
        for op in self.graph.get_operations():
            if op.type in ('Placeholder', 'PlaceholderWithDefault'):
                assert len(op.values()) == 1
                tensor = op.values()[0]
                placeholders[tensor.name] = tensor

        if return_dict:
            return placeholders
        return list(placeholders.values())

    def tensor(self, names, fail=True) -> tf.Tensor:
        is_single = not isinstance(names, (list, tuple))
        if is_single:
            names = [names]

        results = []
        for name in names:
            if isinstance(name, tf.Tensor):
                results.append(name)
                continue
            assert isinstance(name, str)

            if ':' not in name:
                name = '{}:0'.format(name)
            try:
                results.append(self.graph.get_tensor_by_name(name))
            except Exception as ex:
                if fail:
                    raise ex
                results.append(None)

        if is_single:
            assert len(results) == 1
            return results[0]
        return results

    def tensor_or_operation(self, names, fail=True):
        is_single = not isinstance(names, (list, tuple))
        if is_single:
            names = [names]

        results = []
        for name in names:
            val = self.tensor(name, fail=False)
            if val is None:
                val = self.operation(name, fail=fail)
            # val can still be None.
            results.append(val)

        if is_single:
            assert len(results) == 1
            return results[0]
        return results

    def train(self,
              feed_dict: dict,
              summary_modes=None,
              check_placeholder_coverage=True):
        """
        Runs a training step.

        :param feed_dict: A dictionary that maps g elements to values. Keys can be regular expressions
        or placeholder objects.
        :param summary_modes: A list of summary modes, 'SIMPLE', 'ALL', 'IMAGE', etc. Can be empty (default).
        :param check_placeholder_coverage: If `True`, raise an exception when `feed` contains an optional placeholder
        and there are any remaining optional placeholders that depend on the same queue.
        """
        self.eval([], feed_dict=feed_dict, is_training=True, summary_modes=summary_modes, check_placeholder_coverage=check_placeholder_coverage)

    def enqueue(self, feed_dict: dict):
        assert not self.needs_initialization, 'Variables are not initialized.'
        new_feed_dict = {}
        for k, v in feed_dict.items():
            if isinstance(k, str):
                if not k.startswith(NNModel._producer_queue_prefix):
                    k = 'placeholder/{}/{}'.format(NNModel._producer_queue_prefix, k)
                new_feed_dict[self[k]] = v
            elif isinstance(k, tf.Tensor):
                new_feed_dict[k] = v
            else:
                raise ValueError('Unexpected key in feed_dict: {}'.format(k))

        for k, v in new_feed_dict.items():
            assert isinstance(k, tf.Tensor)
            assert k.name.startswith('placeholder/{}'.format(NNModel._producer_queue_prefix))

        enqueue_op = self.operation('{}/enqueue'.format(NNModel._producer_queue_prefix))
        self.session.run(enqueue_op, feed_dict=new_feed_dict)

    def set_learning_rate(self, learning_rate):
        assert not self.needs_initialization, 'Variables are not initialized.'
        assert isinstance(learning_rate, float)
        assert np.isfinite(learning_rate)
        new_learning_rate = self.session.run(self.tensor('assign_learning_rate'), feed_dict={
            self.placeholder('learning_rate'): learning_rate
        })
        assert np.isclose(learning_rate, new_learning_rate)

    def eval(self,
             values=None,
             feed_dict=None,
             collection_keys=None,
             summary_modes=None,
             summary_writer_name=None,
             is_training=False,
             check_placeholder_coverage=True):
        """
        Evaluates TensorFlow Operations.

        :param values: Similar to the `fetches` argument of `tf.Session.run`. This can
        also be regex patterns. Can be a list or single value.
        :param feed_dict: A dictionary that maps g elements to values. Keys can be regular expressions
        or placeholder objects.
        :param collection_keys: All values in the given collections will be added to `fetches`.
        :param summary_modes: A list of summary modes, 'SIMPLE', 'ALL', 'IMAGE', etc. Can be empty (default).
        :param summary_writer_name: If None, default is 'train' if `is_training` is True, 'eval' otherwise.
        If this is a new name, a summary writer will be created.
        :param is_training: If `True`, executes a training step.
        :param check_placeholder_coverage: If `True`, raise an exception when `feed` contains an optional placeholder
        and there are any remaining optional placeholders that depend on the same queue.
        :return: A dictionary that maps `tensors_or_patterns` to evaluated values.
        """
        assert not self.needs_initialization, 'Variables are not initialized.'
        if values is None:
            values = []
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

        is_single_value = isinstance(values, (tf.Tensor, tf.Operation, str))
        if is_single_value:
            values = [values]

        # TODO(daeyun): If is_training is false and `fetches` contains an op that updates variables, e.g. minimizer, raise exception.

        names = []
        new_feed_dict = {}
        for k, v in feed_dict.items():
            ph = self.placeholder(k)
            if ph is None:
                raise ValueError('Unexpected key in feed_dict: {}'.format(k))
            new_feed_dict[ph] = v
            names.append(ph.name)

        fetches = self.tensor_or_operation(values)
        if None in fetches:
            raise ValueError('Unidentifiable pattern: {}'.format(values[fetches.index(None)]))

        for collection_key in collection_keys:
            fetches.extend(self.graph.get_collection(collection_key))

        assert self.placeholder('is_training') is not None
        assert self.placeholder('learning_rate') is not None
        if is_training:
            assert self.placeholder('learning_rate').name not in names, \
                ('learning_rate placeholder is only used when changing the learning rate variable. '
                 'It should NOT be in `feed_dict` for training.')

            # There is a race condition here, but it does not matter in most use cases.
            # `train/step` will update any moving average variables as well.
            minimize = self.operation('train/minimize')
            if minimize not in fetches:
                fetches.append(minimize)

            if self.placeholder('is_training') not in new_feed_dict:
                new_feed_dict[self.placeholder('is_training')] = True
        else:
            if self.placeholder('is_training') not in new_feed_dict:
                new_feed_dict[self.placeholder('is_training')] = False

        # Do not allow using enqueue placeholders e.g. placeholder/queue/input to avoid confusion because they are not needed
        # here in most use cases.
        enqueue_prefix = 'placeholder/{}'.format(NNModel._producer_queue_prefix)
        for placeholder, _ in new_feed_dict.items():
            if enqueue_prefix in placeholder.name:
                raise ValueError('{}/* is should only be used from NNModel.enqueue()'.format(placeholder.name))

        # if self._summary_ops is None:
        #     # summary_dir is not provided.
        #     assert len(summary_modes) == 0
        #
        # for summary_mode in summary_modes:
        #     if summary_mode not in self._summary_ops:
        #         raise ValueError('Unrecognized summary mode: {}'.format(summary_mode))
        #
        # summary_ops = self.summary_ops(summary_modes) if self._summary_writers else []
        # summary_op_fetch_indices = (len(fetches), len(fetches) + len(summary_ops))
        # if summary_ops:
        #     fetches.extend(summary_ops)
        #
        # assert len(fetches) > 0, '`fetches` cannot be empty.'
        #
        # if check_placeholder_coverage:
        #     check_feed_queue_coverage(new_feed_dict)
        #
        with self.graph.as_default():
            out_eval = self.session.run(fetches, new_feed_dict)
        #
        # if summary_ops:
        #     if summary_writer_name is None:
        #         summary_writer_name = 'train' if is_training else 'eval'
        #     assert isinstance(summary_writer_name, str)
        #     writer = self._summary_writer(summary_writer_name)
        #     global_step = self.global_step()
        #
        #     summary_results = out_eval[summary_op_fetch_indices[0]:summary_op_fetch_indices[1]]
        #     for summary_result in summary_results:
        #         assert isinstance(summary_result, bytes)
        #         writer.add_summary(summary_result, global_step)
        #
        #     # TODO(daeyun): Listen for a keypress event to signal flush.
        #     writer.flush()

        if is_single_value:
            return out_eval[0]

        # `tensors_or_patterns` and `out_eval` may not be the same size.
        results = {}
        for name, result in zip(values, out_eval):
            if result is not None:
                results[name] = result
        return results

    def __getitem__(self, pattern):
        """
        Same as `get`. Returns a Variable or Tensor whose name uniquely matches the pattern.
        :param pattern: A regular expression pattern.
        :return: Matched variable or tensor.
        """
        assert isinstance(pattern, str)
        return self.tensor(pattern)


class SampleModel(NNModel):
    def _model(self):
        out = nn_ops.batch_norm(self.placeholder('input'), is_trainable=True, is_local=True)
        loss = tf.reduce_mean((out - self.placeholder('target')) ** 2, name='loss')
        return loss

    def _placeholders(self):
        return [
            QueuePlaceholder(tf.float32, shape=[None, 5, 5, 1], name='input', is_file=False),
            QueuePlaceholder(tf.float32, shape=[None, 5, 5, 1], name='target', is_file=True),
        ]
