import abc
import contextlib
import functools
import threading
import numbers
import numpy as np
import os
import time
import re
from os import path

import tensorflow as tf
import toposort
import typing

from dshin import log
from dshin.nn import graph_utils
from dshin.nn import ops as nn_ops
from dshin.nn import types as nn_types
import collections
import socket

memoize = functools.lru_cache(maxsize=2048, typed=True)


class QueuePlaceholder(object):
    def __init__(self, dtype, shape, name, is_file):
        self.dtype = dtype
        self.shape = shape
        self.name = name
        self.is_file = is_file


class QueueComponents(object):
    def __init__(self, name, placeholders=None, tensors=None, enqueue_op=None, queue=None):
        self.name = name
        self.placeholders = placeholders
        self.tensors = tensors
        self.enqueue_op = enqueue_op
        self.queue = queue


class CountingQueueRunner(object):
    def __init__(self, queue, enqueue_ops, timeout_ms=None, name=None):
        self.queue = queue
        self.enqueue_ops = enqueue_ops
        self._count = 0
        self._eval_limit = 0
        self.should_stop = False
        self.name = name

        if timeout_ms is None:
            self._timeout_ms = 0
            self._run_options = None
        else:
            self._timeout_ms = timeout_ms
            self._run_options = tf.RunOptions(timeout_in_ms=timeout_ms)
        self._lock = threading.RLock()
        self._threads = []

    def _run(self, sess: tf.Session, enqueue_op: tf.Operation):
        while True:
            with self._lock:
                if self._count < self._eval_limit:
                    self._count += 1
                else:
                    break
            while True:
                if self.should_stop:
                    break
                try:
                    sess.run(enqueue_op, options=self._run_options)
                except tf.errors.DeadlineExceededError:
                    pass
                else:
                    break
            if self.should_stop:
                break

    def request_stop(self):
        self.should_stop = True

    @property
    def timeout(self):
        return self._timeout_ms

    @property
    def count(self):
        return self._count

    @property
    def limit(self):
        return self._eval_limit

    def running_threads(self):
        return [t for t in self._threads if t.is_alive()]

    def join(self):
        for t in self._threads:
            t.join()

    def _reset(self):
        with self._lock:
            running = self.running_threads()
            if len(running) > 0:
                raise RuntimeError('{} threads are still running.'.format(len(running)))
            self._count = 0

    def start_threads(self, sess, eval_limit):
        assert isinstance(sess, tf.Session)
        assert isinstance(eval_limit, int)

        with self._lock:
            self._reset()
            self._eval_limit = eval_limit

            threads = []
            for op in self.enqueue_ops:
                assert isinstance(op, tf.Operation)

                t = threading.Thread(target=self._run, args=(sess, op))
                t.daemon = True
                t.start()
                threads.append(t)

            self._threads = threads

        return threads


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


def get_local_cluster_spec(num_processes):
    assert isinstance(num_processes, dict)
    jobs = collections.defaultdict(list)
    sockets = []
    for job_name, num in num_processes.items():
        for i in range(num):
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sockets.append(s)
            s.bind(('localhost', 0))
            addr, port = s.getsockname()
            host = '{}:{}'.format(addr, port)
            jobs[job_name].append(host)
            log.info('Assigned {} to /job:{}/task:{}'.format(host, job_name, i))
    for s in sockets:
        s.close()
    return tf.train.ClusterSpec(jobs)


def ensure_list_or_tuple(value_or_values, dtype=None):
    if isinstance(value_or_values, (list, tuple)):
        out = value_or_values
    else:
        out = tuple(value_or_values)
    if dtype is not None and len(out) != 0:
        assert isinstance(out[0], dtype)
    return out


def job_info_from_server_def(server_def):
    for job in server_def.cluster.job:
        if job.name == server_def.job_name:
            tasks = list(zip(*sorted([(k, v) for k, v in job.tasks.items()])))[1]
            return job.name, tasks
    raise RuntimeError('Unable to parse job info.')


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


def sort_tensors(ops):
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


class NNModel(metaclass=abc.ABCMeta):
    _source_queue_prefix = 'queue'
    _consumer_queue_prefix = 'worker_queue'
    _saver_prefix = 'saver'
    _optimizer_prefix = 'optim'
    _meta_graph_suffix = '.meta'
    _checkpoint_basename = 'model.ckpt'  # Full path will be `{log_dir}/{_checkpoint_basename}`
    _summary_key_prefix = 'summary_'
    _summary_keys = [
        'scalar',
        'image',
        'histogram',
        'train_update_ratio',  # NOTE(daeyun): This also runs a training step.
    ]

    def __init__(self, graph: tf.Graph = None, log_dir=None):
        if graph is None:
            graph = tf.Graph()
        self.graph = graph
        self.needs_initialization = True

        self._consumer_queue_runners = {}
        self._consumer_threads = {}

        self._current_local_queue_source_name = None

        self._summary_tensors = []
        self._summary_writers = {}

        if log_dir:
            self.log_dir = path.expanduser(log_dir)
            if not path.isdir(self.log_dir):
                os.makedirs(self.log_dir)
                log.info('Created directory: %s', self.log_dir)
            assert path.isdir(self.log_dir)
        else:
            self.log_dir = None

    def _init_summaries(self):
        assert not self.needs_initialization

        if self.log_dir:
            with self.graph.as_default():
                self._summary_tensors = {}
                for name in self._summary_keys:
                    fullname = self._summary_key_prefix + name
                    self._summary_tensors[name] = tf.merge_all_summaries(key=fullname)

            # Graph is only added to the 'train' summary file.
            self._summary_writers['train'] = self._summary_writer('train', graph=self.session.graph)

    def _summary_writer(self, name: str = 'eval', graph: tf.Graph = None) -> tf.train.SummaryWriter:
        """
        Creates or gets a summary writer.

        :param name: Name of the subdirectory.
        :param graph: A `tf.Graph` object saved in the summary. In most use cases, only one summary writer
        would need to save this. This is only used when the summary writer does not already exist.
        :return:
        """
        if name not in self._summary_writers:
            summary_writer_path = path.join(self.log_dir, 'summary', name)
            log.info('Creating summary writer %s at %s', name, summary_writer_path)
            self._summary_writers[name] = tf.train.SummaryWriter(summary_writer_path, graph=graph)
        return self._summary_writers[name]

    @abc.abstractmethod
    def _model(self) -> tf.Tensor:
        return

    @abc.abstractmethod
    def _placeholders(self):
        return

    def _build_input_pipeline(self, queue_names, queue_sizes, local_queue_size) -> typing.Sequence[QueueComponents]:
        assert self.needs_initialization

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

        batch_size = self.placeholder('batch_size', fail=True)
        assert isinstance(batch_size, tf.Tensor)

        def build_queue(name, size):
            source_name = '{}/{}'.format(NNModel._source_queue_prefix, name)
            queue_components = self._build_source_queue(placeholder_specs,
                                                        queue_name=source_name,
                                                        queue_size=size,
                                                        enqueue_device=input_queue_device,
                                                        dequeue_device=self.local_device_name())

            return queue_components

        all_queue_components = [build_queue(name, size) for name, size in zip(queue_names, queue_sizes)]

        consumer_name = '{}'.format(NNModel._consumer_queue_prefix)
        tensor_dict = self._build_batch_tensors(name=consumer_name,
                                                batch_size=batch_size,
                                                queue_size=local_queue_size,
                                                all_source_queue_components=all_queue_components)

        with tf.name_scope('placeholder/'):
            for i, (name, tensor) in enumerate(tensor_dict.items()):
                field_name = placeholder_specs[i].name
                shape = placeholder_specs[i].shape
                tf.placeholder_with_default(tensor, shape=shape, name=field_name)

        return tensor_dict

    def local_device_name(self, cpu_id=None, gpu_id=None):
        assert cpu_id is None or gpu_id is None
        base_device = "/job:{job}/task:{task}".format(job=self.job_name, task=self.task_id)
        if gpu_id is not None:
            device = '{}/device:GPU:{}'.format(base_device, gpu_id)
        elif cpu_id is not None:
            device = '{}/device:CPU:{}'.format(base_device, cpu_id)
        else:
            device = base_device
        return device

    def build(self,
              server,
              source_queue_names=('train',),
              source_queue_sizes=(2000,),
              local_queue_size=100,
              seed=None,
              sync_replicas=True,
              save_model_secs=600):
        assert self.needs_initialization
        assert isinstance(server, tf.train.Server)

        self.server = server

        self.queue_names = ensure_list_or_tuple(source_queue_names)
        self.queue_sizes = ensure_list_or_tuple(source_queue_sizes)
        self.local_queue_size = local_queue_size
        assert len(self.queue_names) == len(self.queue_sizes)
        self.seed = seed

        self.task_id = server.server_def.task_index
        self.is_chief = self.task_id == 0

        self.job_name, self.hosts = job_info_from_server_def(server.server_def)
        self.num_replicas = len(self.hosts)

        # TODO(daeyun): Prevent data processes from consuming data from the queue

        worker_device_name = self.local_device_name()
        log.info('[%s %d] Device name is %s', self.job_name, self.task_id, worker_device_name)
        device_setter = tf.train.replica_device_setter(worker_device=worker_device_name, cluster=server.server_def.cluster)

        with self.graph.as_default(), tf.device(device_setter):
            if seed is not None:
                # Graph-level random seed.
                tf.set_random_seed(seed)

            self._build_input_pipeline(self.queue_names, self.queue_sizes, self.local_queue_size)

            # Builds the main model.
            loss = self._model()
            assert isinstance(loss, tf.Tensor), '_model() must return a Tensor for the total loss.'

            with tf.variable_scope(self._optimizer_prefix):
                optim = self._optimizer(self.variable('learning_rate'))

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

                queue_runners = []
                if self.is_chief:
                    if sync_replicas:
                        queue_runners.append(optim.get_chief_queue_runner())
                        queue_runners.extend(self.graph.get_collection(tf.GraphKeys.QUEUE_RUNNERS))
                        init_tokens_op = optim.get_init_tokens_op()

            summary_op = tf.merge_all_summaries()

            # Exclude any queue-specific variables.
            vars_to_save = []
            for name in self.queue_names:
                for var in self.get_collections([tf.GraphKeys.TRAINABLE_VARIABLES, tf.GraphKeys.VARIABLES]):
                    if not var.name.startswith('{}/{}/'.format(self._source_queue_prefix, name)):
                        vars_to_save.append(var)

            self.saver = tf.train.Saver(
                name=self._saver_prefix,
                max_to_keep=10,
                keep_checkpoint_every_n_hours=0.5,
                var_list=vars_to_save,
            )
            # Variables created after this will not be saved.

            # TODO(daeyun)
            graph_utils.save_graph_text_summary(self.graph)

            init_op = tf.initialize_all_variables()
            local_init_op = tf.initialize_local_variables()

            if path.isfile(self.checkpoint_filename()):
                log.info('{} exists. Model will be restored.'.format(self.checkpoint_filename()))

            # Create a "supervisor", which oversees the training process.
            self.supervisor = tf.train.Supervisor(is_chief=self.is_chief,
                                                  logdir=self.log_dir,
                                                  init_op=init_op,
                                                  summary_op=summary_op,
                                                  saver=self.saver,
                                                  recovery_wait_secs=1,
                                                  save_summaries_secs=0,
                                                  global_step=self.variable('global_step'),
                                                  save_model_secs=save_model_secs,
                                                  checkpoint_basename=self._checkpoint_basename,
                                                  )

        config = server.server_def.default_session_config

        log.info('[%s %d] Waiting for session.', self.job_name, self.task_id)
        sess = self.supervisor.prepare_or_wait_for_session(master=server.target, config=config,
                                                           start_standard_services=True)
        assert isinstance(sess, tf.Session)
        self.session = sess

        log.info('[%s %d] Session is ready.', self.job_name, self.task_id)

        if self.is_chief:
            assert isinstance(sess, tf.Session)
            if sync_replicas:
                sess.run(init_tokens_op)
            sess.run(local_init_op)
            self.supervisor.start_queue_runners(sess, queue_runners)

        self.needs_initialization = False

        self._init_summaries()  # Sets self._summary_ops

        log.info('global_step: %d', self.eval('global_step'))
        log.info('learning_rate: %f', self.eval('learning_rate'))

    def checkpoint_filename(self):
        return path.join(self.log_dir, self._checkpoint_basename)

    def _build_source_queue(self,
                            placeholder_specs,
                            queue_name,
                            queue_size,
                            enqueue_device,
                            dequeue_device):
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
        :param queue_size: The capacity of the queue.
        :param queue_device: Device for the queue operations.
        :return: A `_QueueComponents` with the following fields:
            name: The name of the created queue. e.g. `{queue_name}`
            enqueue_op: An `enqueue_many` operation that enqueues placeholder values.
            placeholders: A list of placeholders used to insert new values to the queue. `placeholders[i]` corresponds
                          to `placeholder_specs[i]`. If `placeholder_specs[i].is_file` is `True`, the type will be a list of strings.
            tensors: A dictionary that maps placeholder names (without prefix) to the dequeued output tensors.
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

        with tf.device(enqueue_device):
            queue = tf.FIFOQueue(queue_size, dtypes=dtypes, shapes=shapes,
                                 names=names, shared_name=queue_name, name=queue_name)
            enqueue_tensors = {name: placeholder for name, placeholder in zip(names, placeholders)}

            with tf.name_scope('{}/'.format(queue_name)):
                # e.g. `queue/train/size:0`.
                queue.size(name='size')
                queue.close(True, name='close')
                # `enqueue_op` is an atomic operation. Values for all of `placeholder/{queue_name}/*` need to be specified at runtime.
                enqueue_op = queue.enqueue_many(enqueue_tensors, name='enqueue')

            with tf.variable_scope('{}/'.format(queue_name)):
                dequeue_count = tf.Variable(0, trainable=False, dtype=tf.int64, name='count')

                # `use_locking` is needed.
                increment = dequeue_count.assign_add(1, use_locking=True)

        with tf.device(dequeue_device):
            # The dequeue operation itself still happens on `enqueue_device`.
            with tf.control_dependencies([increment]):
                dequeue_tensors = queue.dequeue(name='dequeue')
            assert isinstance(dequeue_tensors, dict)

            # Processing (e.g. loading file from retrieved filename) happens on `dequeue_device`.
            tensors = {}
            for field_name, value in dequeue_tensors.items():
                input_spec = input_specs_by_name[field_name]
                if input_spec.is_file:
                    # This should happen on `dequeue_device`.
                    value = nn_ops.npz_to_tensor(value, dtype=input_spec.dtype, shape=input_spec.shape[1:])
                else:
                    # Sanity check.
                    value.get_shape().assert_is_compatible_with(input_spec.shape[1:])
                tensors[field_name] = value

        return QueueComponents(name=queue.name, enqueue_op=enqueue_op, placeholders=placeholders, queue=queue, tensors=tensors)

    def join_local_queue_runner_threads(self, name=None):
        if name is None:
            name = self._current_local_queue_source_name
        qr = self._consumer_queue_runners[name]
        assert isinstance(qr, CountingQueueRunner)
        qr.join()

    def start_local_queue_runner(self, name, num_examples, num_threads=5):
        qr = self._consumer_queue_runners[name]
        assert isinstance(qr, CountingQueueRunner)
        assert len(qr.running_threads()) == 0

        for op in qr.enqueue_ops:
            assert op.name == qr.enqueue_ops[0].name

        self._current_local_queue_source_name = name

        if num_threads is not None and num_threads > 0:
            qr.enqueue_ops = [qr.enqueue_ops[0]] * num_threads
        self._consumer_threads[name] = qr.start_threads(self.session, num_examples)
        # TODO(daeyun): check dequeue from the local queues still work after global dequeue halts.

    def _build_batch_tensors(self, name, batch_size, queue_size, all_source_queue_components: typing.Sequence[QueueComponents]):
        """
        Returns a batched tensor that pulls and concatenates values from the source queue upon evaluation.

        Values dequeued from the source will be buffered in a consumer queue. A `QueueRunner` for this queue will be added to the
        current `Graph`'s `QUEUE_RUNNERS` collection.

        This function also creates new placeholders in the `placeholder/` scope for directly feeding tensors instead of dequeueing
        from the queue.

        :param name: A name for the operations.
        :param batch_size: An integer, Tensor, or Variable batch size pulled from the consumer queue in the dequeue operation.
        :param all_source_queue_components: The components of the queue feeding into this queue. See `QueueComponents`.
        :return: A `QueueComponents` with the following fields:
            name: The name of the created queue. e.g. `{name}/fifo_queue`
            placeholders: A list of placeholders used to insert new values to the queue. `placeholders[i]` corresponds
                          to `placeholder_specs[i]`. If `placeholder_specs[i].is_file` is `True`, the type will be a list of strings.
            tensors: A dictionary that maps placeholder names to output tensors derived from the dequeued values.
                     Shape of the tensors will be `placeholder_spec.shape[1:]`
            queue: The `tf.FIFOQueue` object. Not needed in most use cases.
        """
        with tf.variable_scope('{}/'.format(name)):
            source_queue_components = all_source_queue_components[0]
            tensors = source_queue_components.tensors
            types = [t.dtype for t in tensors.values()]
            shapes = [t.get_shape() for t in tensors.values()]
            names = source_queue_components.queue.names
            queue = tf.FIFOQueue(capacity=queue_size, dtypes=types, shapes=shapes, names=names, shared_name=None, name='queue')

            # `queue_name/size:0`.
            queue.size('size')
            queue.close(True, name='close')

            batch_tensors = queue.dequeue_many(batch_size, name='dequeue')

            for source_queue_components in all_source_queue_components:
                basename = source_queue_components.name.rsplit('/', 1)[-1]

                with tf.variable_scope('{}'.format(basename)):
                    tensors = source_queue_components.tensors

                    # Enqueue to the local destination queue.
                    enqueue_op = queue.enqueue(tensors, name='enqueue')
                    enqueue_ops = [enqueue_op]

                    qr = CountingQueueRunner(queue, enqueue_ops=enqueue_ops, name=basename)
                    self._consumer_queue_runners[basename] = qr

            # Sanity check.
            assert isinstance(batch_tensors, dict)
            for key, tensor in batch_tensors.items():
                assert isinstance(tensor, tf.Tensor)
                shape = tensor.get_shape()
                assert shape[0].value is None
                for dim in shape[1:]:
                    assert dim.value is not None
                assert self.placeholder(key, fail=False) is None

            return batch_tensors

    def _optimizer(self, learning_rate):
        # TODO(daeyun): consider setting `use_locking`.
        return tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=1e-4)

    def default_variables(self):
        return [
            tf.get_variable('global_step', shape=(), initializer=tf.constant_initializer(0, dtype=tf.int64), dtype=tf.int64, trainable=False),
            tf.get_variable('learning_rate', shape=(), initializer=tf.constant_initializer(0.001, dtype=tf.float32), dtype=tf.float32, trainable=False),
        ]

    def default_placeholders(self) -> typing.Sequence[tf.Tensor]:
        return [
            tf.placeholder(tf.float32, shape=(), name='learning_rate'),
            tf.placeholder_with_default(tf.constant(False, name='kFalse'), shape=(), name='is_training'),
            tf.placeholder(tf.int32, shape=(), name='batch_size'),
            tf.placeholder(tf.int64, shape=(), name='limit'),
        ]

    def shutdown(self):
        assert not self.needs_initialization
        self.supervisor.request_stop()

        for name, runner in self._consumer_queue_runners.items():
            runner.request_stop()
            # Unblock blocking queue operations.
            self.session.run(self.operation('{}/close'.format(self._consumer_queue_prefix)))
            self.session.run(self.operation('{}/{}/close'.format(self._source_queue_prefix, name)))

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
                    graph_utils.save_graph_text_summary(self.graph, dirname=self.log_dir, basename='graph_summary.txt')
                    raise ex
                results.append(None)

        if is_single:
            assert len(results) == 1
            return results[0]
        return results

    def local_queue_runners(self):
        return self._consumer_queue_runners

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
                for variable in (self.graph.get_collection(tf.GraphKeys.VARIABLES) +
                                     self.graph.get_collection(tf.GraphKeys.LOCAL_VARIABLES) +
                                     self.graph.get_collection(tf.GraphKeys.MODEL_VARIABLES) +
                                     self.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) +
                                     self.graph.get_collection(tf.GraphKeys.MOVING_AVERAGE_VARIABLES)):
                    if variable.name == name:
                        results.append(variable)
            except Exception as ex:
                if fail:
                    graph_utils.save_graph_text_summary(self.graph, dirname=self.log_dir, basename='graph_summary.txt')
                    raise ex
                results.append(None)

        results = {item.name: item for item in results}
        results = list(results.values())

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
                    graph_utils.save_graph_text_summary(self.graph, dirname=self.log_dir, basename='graph_summary.txt')
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

    def get_collections(self, names):
        names = ensure_list_or_tuple(names)
        ret = []
        seen = set()
        for name in names:
            for item in self.graph.get_collection(name):
                if item in seen:
                    continue
                seen.add(item)
                # Preserves order.
                ret.append(item)
        return ret

    def train(self,
              feed_dict: dict,
              summary_keys=None,
              check_placeholder_coverage=True):
        """
        Runs a training step.

        :param feed_dict: A dictionary that maps g elements to values. Keys can be regular expressions
        or placeholder objects.
        :param summary_keys: A list of summary modes, 'SIMPLE', 'ALL', 'IMAGE', etc. Can be empty (default).
        :param check_placeholder_coverage: If `True`, raise an exception when `feed` contains an optional placeholder
        and there are any remaining optional placeholders that depend on the same queue.
        """
        self.eval([], feed_dict=feed_dict, is_training=True, summary_keys=summary_keys, check_placeholder_coverage=check_placeholder_coverage)

    def enqueue(self, name, feed_dict: dict):
        assert not self.needs_initialization, 'Variables are not initialized.'
        new_feed_dict = {}
        for k, v in feed_dict.items():
            if isinstance(k, str):
                if not k.startswith(self._source_queue_prefix):
                    k = 'placeholder/{}/{}/{}'.format(self._source_queue_prefix, name, k)
                new_feed_dict[self.placeholder(k)] = v
            elif isinstance(k, tf.Tensor):
                new_feed_dict[k] = v
            else:
                raise ValueError('Unexpected key in feed_dict: {}'.format(k))

        for k, v in new_feed_dict.items():
            assert isinstance(k, tf.Tensor)
            assert k.name.startswith('placeholder/{}/{}'.format(self._source_queue_prefix, name))

        enqueue_op = self.operation('{}/{}/enqueue'.format(self._source_queue_prefix, name))
        self.session.run(enqueue_op, feed_dict=new_feed_dict)

    def set_learning_rate(self, learning_rate):
        assert not self.needs_initialization, 'Variables are not initialized.'
        assert isinstance(learning_rate, float)
        assert np.isfinite(learning_rate)
        new_learning_rate = self.session.run(self.tensor('assign_learning_rate'), feed_dict={
            self.placeholder('learning_rate'): learning_rate
        })
        assert np.isclose(learning_rate, new_learning_rate)

    def learning_rate(self):
        return self.session.run(self.variable('learning_rate'))

    def eval(self,
             values=None,
             feed_dict=None,
             collection_keys=None,
             summary_keys=None,
             summary_writer_name=None,
             is_training=False,
             check_placeholder_coverage=True,
             ):
        """
        Evaluates TensorFlow Operations.

        :param values: Similar to the `fetches` argument of `tf.Session.run`. This can
        also be regex patterns. Can be a list or single value.
        :param feed_dict: A dictionary that maps g elements to values. Keys can be regular expressions
        or placeholder objects.
        :param collection_keys: All values in the given collections will be added to `fetches`.
        :param summary_keys: A list of summary modes, 'SIMPLE', 'ALL', 'IMAGE', etc. Can be empty (default).
        :param summary_writer_name: If None, default is 'train' if `is_training` is True, 'eval' otherwise.
        If this is a new name, a summary writer will be created.
        :param is_training: If `True`, executes a training step.
        :param check_placeholder_coverage: If `True`, raise an exception when `feed` contains an optional placeholder
        and there are any remaining optional placeholders that depend on the same queue.
        :return: A dictionary that maps `tensors_or_patterns` to evaluated values.
        """
        # if hasattr(self, 'threads'):
        #     print([t.is_alive() for t in self.threads])

        assert not self.needs_initialization, 'Variables are not initialized.'
        if values is None:
            values = []
        if collection_keys is None:
            collection_keys = []
        if feed_dict is None:
            feed_dict = {}
        if summary_keys is None:
            summary_keys = []
        else:
            if len(summary_keys) == 0:
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
            minimize = self.operation('{}/minimize'.format(self._optimizer_prefix))
            if minimize not in fetches:
                fetches.append(minimize)

            if self.placeholder('is_training') not in new_feed_dict:
                new_feed_dict[self.placeholder('is_training')] = True
        else:
            if self.placeholder('is_training') not in new_feed_dict:
                new_feed_dict[self.placeholder('is_training')] = False

        # Do not allow using enqueue placeholders e.g. placeholder/queue/input to avoid confusion because they are not needed
        # here in most use cases.
        enqueue_prefix = 'placeholder/{}/'.format(self._source_queue_prefix)
        for placeholder, _ in new_feed_dict.items():
            if enqueue_prefix in placeholder.name:
                raise ValueError('{}/* is should only be used from NNModel.enqueue()'.format(placeholder.name))

        if self._summary_tensors is None:
            # log_dir is not provided.
            assert len(summary_keys) == 0

        for summary_key in summary_keys:
            if summary_key not in self._summary_keys:
                raise ValueError('Unrecognized summary key: {}'.format(summary_key))

        summary_ops = self.summary_tensors(summary_keys)
        summary_op_fetch_indices = (len(fetches), len(fetches) + len(summary_ops))
        if summary_ops:
            fetches.extend(summary_ops)

        assert len(fetches) > 0, '`fetches` cannot be empty.'

        if check_placeholder_coverage:
            check_feed_queue_coverage(new_feed_dict)

        with self.graph.as_default():
            out_eval = self.session.run(fetches, new_feed_dict)

        if summary_ops:
            if summary_writer_name is None:
                summary_writer_name = 'train' if is_training else 'eval'
            assert isinstance(summary_writer_name, str)
            writer = self._summary_writer(summary_writer_name)
            global_step = self.session.run(self.variable('global_step'))

            summary_results = out_eval[summary_op_fetch_indices[0]:summary_op_fetch_indices[1]]
            for summary_result in summary_results:
                assert isinstance(summary_result, bytes)
                writer.add_summary(summary_result, global_step=global_step)

            # TODO(daeyun): Listen for a keypress event to signal flush.
            writer.flush()

        if is_single_value:
            return out_eval[0]

        # `tensors_or_patterns` and `out_eval` may not be the same size.
        results = {}
        for name, result in zip(values, out_eval):
            if result is not None:
                results[name] = result
        return results

    def summary_tensors(self, keys) -> typing.Sequence[tf.Tensor]:
        """
        Returns a list of summary Tensors.

        :param modes: Can be a string or sequence.
        :return: A list of TensorFlow summary Tensors.
        """
        keys = ensure_list_or_tuple(keys, str)

        summaries = []
        for key in keys:
            if key not in self._summary_tensors:
                raise ValueError('Unrecognized summary key: {}'.format(key))
            op = self._summary_tensors[key]
            if op is not None:
                assert isinstance(op, tf.Tensor)
                summaries.append(op)

        return list({t.name: t for t in summaries}.keys())

    def save(self, save_path: str = None):
        """
        Saves variables to a file.

        :param save_path: Path to the checkpoint file.
        """
        if save_path is None:
            save_path = self.checkpoint_filename()
        assert isinstance(save_path, str)
        assert not self.needs_initialization
        assert not path.isdir(save_path) and not save_path.endswith('/'), 'save_path must be a file: {}'.format(save_path)

        dirpath = path.dirname(save_path)
        if not path.isdir(dirpath):
            log.info('mkdir %s', dirpath)
            os.makedirs(dirpath)

        with self.graph.as_default():
            self.saver.export_meta_graph(save_path + NNModel._meta_graph_suffix, as_text=True)
            save_path_out = self.saver.save(self.session, save_path, write_meta_graph=False)
            log.info("Model saved to file: %s" % save_path_out)

    def restore(self, restore_path: str = None):
        """
        Restores a previously saved model.

        :param restore_path: The path used to save the model.
        """
        if restore_path is None:
            restore_path = self.checkpoint_filename()
        assert path.isfile(restore_path), '{} is not a file.'.format(restore_path)
        assert isinstance(restore_path, str)
        assert not self.needs_initialization
        with self.graph.as_default():
            with self.session.as_default():
                self.saver.restore(self.session, restore_path)
                log.info("Restored model from %s", restore_path)

    def add_scalar_summary(self, tag: str, value: tf.Tensor, summary_keys=(), name: str = None) -> tf.Tensor:
        """
        A wrapper around `tf.scalar_summary` that prints out logs and adds to the `all` collection.
        """
        if name is None:
            name = tag + '_summary'
        ensure_list_or_tuple(summary_keys, str)
        for summary_key in summary_keys:
            if summary_key not in self._summary_keys:
                self._summary_keys.append(summary_keys)
                log.info('Adding a new summary key: %s', summary_key)
        assert summary_keys in self._summary_keys

        with_default_summary_keys = list(summary_keys) + ['scalar']
        collections = [self._summary_key_prefix + key for key in with_default_summary_keys]
        log.info('Adding scalar summary tag %s to collections %s.', tag, ','.join(collections))
        summary_op = tf.scalar_summary(tag, value, collections=collections, name=name)
        return summary_op

    def add_historgram_summary(self, tag: str, values: tf.Tensor, summary_keys=(), name: str = None) -> tf.Tensor:
        """
        A wrapper around `tf.histogram_summary` that prints out logs and adds to the `all` collection.
        """
        if name is None:
            name = tag + '_summary'
        ensure_list_or_tuple(summary_keys, str)
        ensure_list_or_tuple(values, tf.Tensor)
        for summary_key in summary_keys:
            if summary_key not in self._summary_keys:
                self._summary_keys.append(summary_keys)
                log.info('Adding a new summary key: %s', summary_key)
        assert summary_keys in self._summary_keys

        with tf.name_scope('histogram_summary_value') as scope:
            concat_value = tf.concat(0, [tf.reshape(v, [-1]) if v.get_shape().ndims > 0 else v for v in values], name=scope)

        with_default_summary_keys = list(summary_keys) + ['histogram']
        collections = [self._summary_key_prefix + key for key in with_default_summary_keys]
        log.info('Adding histogram summary tag %s to collection %s', tag, ','.join(collections))
        summary_op = tf.histogram_summary(tag=tag, name=name, values=concat_value, collections=collections)
        return summary_op

    def add_image_summary(self, tag: str, value: tf.Tensor, max_images: int = 3, summary_keys=(), name: str = None) -> tf.Tensor:
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
        ensure_list_or_tuple(summary_keys, str)
        for summary_key in summary_keys:
            if summary_key not in self._summary_keys:
                self._summary_keys.append(summary_keys)
                log.info('Adding a new summary key: %s', summary_key)
        assert summary_keys in self._summary_keys

        shape = value.get_shape().as_list()
        assert len(shape) == 4
        assert shape[-1] in [1, 3, 4]

        with_default_summary_keys = list(summary_keys) + ['image']
        collections = [self._summary_key_prefix + key for key in with_default_summary_keys]
        log.info('Adding image summary tag %s of shape %s to collections %s. max_images: %d', tag, shape, ','.join(collections), max_images)
        summary_op = tf.image_summary(tag, tensor=value, max_images=max_images, collections=collections, name=name)
        return summary_op


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
