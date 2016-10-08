import abc
import functools
import os
import time
import threading
import numbers
import collections
from os import path

import numpy as np
import tensorflow as tf
import toposort
import typing

from dshin import log
from dshin.nn import graph_utils
from dshin.nn import distributed
from dshin.nn import ops as nn_ops


class QueuePlaceholder(object):
    def __init__(self, dtype, shape, name, is_file=None, tf_record_key=None):
        # Assume compression. gzip for "npz", zlib for "tfrecords".
        assert isinstance(dtype, tf.DType)
        assert isinstance(shape, (list, tuple))
        assert isinstance(name, str)
        assert isinstance(is_file, bool) or is_file is None
        assert isinstance(tf_record_key, str) or tf_record_key is None
        self.dtype = dtype
        self.shape = shape
        self.name = name
        if tf_record_key is not None:
            is_file = True
        self.is_file = is_file
        self.tf_record_key = tf_record_key

    def __hash__(self):
        return hash((self.name, self.tf_record_key))


class QueueComponents(object):
    def __init__(self, name, placeholders=None, tensors=None, enqueue_op=None, queue=None):
        self.name = name
        self.placeholders = placeholders
        self.tensors = tensors
        self.enqueue_op = enqueue_op
        self.queue = queue


class CountingQueueRunner(object):
    def __init__(self, queue, enqueue_op, parallel_enqueue_op, parallel_enqueue_size, name=None):
        self.queue = queue
        self._enqueue_op = enqueue_op
        self._parallel_enqueue_op = parallel_enqueue_op
        self._parallel_enqueue_size = parallel_enqueue_size
        self._count = 0
        self._eval_limit = 0
        self.should_stop = False
        self.name = name

        self._lock = threading.RLock()
        self._threads = []

    def _run(self, sess: tf.Session):
        while True:
            with self._lock:
                if self._count < self._eval_limit:
                    is_parallel_enqueue = self._count + self._parallel_enqueue_size < self._eval_limit
                    if is_parallel_enqueue:
                        self._count += self._parallel_enqueue_size
                    else:
                        self._count += 1
                else:
                    break
            if self.should_stop:
                break
            if is_parallel_enqueue:
                sess.run(self._parallel_enqueue_op)
            else:
                sess.run(self._enqueue_op)

    def request_stop(self):
        self.should_stop = True

    @property
    def count(self):
        return self._count

    @property
    def limit(self):
        return self._eval_limit

    def running_threads(self):
        return [t for t in self._threads if t.is_alive()]

    def remaining_enqueue(self):
        return self._eval_limit - self._count

    def join(self):
        for t in self._threads:
            t.join()

    def _reset(self):
        with self._lock:
            running = self.running_threads()
            if len(running) > 0:
                raise RuntimeError('{} threads are still running.'.format(len(running)))
            self._count = 0

    def start_threads(self, sess, eval_limit, num_threads):
        assert isinstance(sess, tf.Session)
        assert isinstance(eval_limit, int)

        with self._lock:
            self._reset()
            self._eval_limit = eval_limit

            threads = []
            for _ in range(num_threads):
                assert isinstance(self._enqueue_op, tf.Operation)
                t = threading.Thread(target=self._run, args=(sess,))
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
    # conf.gpu_options.allocator_type = 'BFC'
    conf.gpu_options.allow_growth = True
    conf.allow_soft_placement = True
    conf.log_device_placement = log_device_placement
    return conf


def ensure_list_or_tuple(value_or_values, dtype=None):
    if isinstance(value_or_values, (list, tuple)):
        out = value_or_values
    else:
        out = (value_or_values,)
    if dtype is not None and len(out) != 0:
        assert isinstance(out[0], dtype)
    return out


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

        self._current_source_queue_name = None

        self._summary_tensors = []
        self._summary_writers = {}

        if log_dir:
            self.log_dir = path.expanduser(log_dir)
            if not path.isdir(self.log_dir):
                try:
                    os.makedirs(self.log_dir)
                    log.info('Created directory: %s', self.log_dir)
                except FileExistsError:
                    # Race condition.
                    pass
            assert path.isdir(self.log_dir)
        else:
            self.log_dir = None

        self.is_debug_mode = False

        self.run_metadata = None

        # These values are set in `self.build()`.
        self.job_name = ''  # Can be 'worker', 'data', or 'local'. Extracted from `tf.train.Server`.
        self.task_id = 0
        self.queue_names = []  # Source queue names. e.g. 'train', 'eval'.
        self.queue_sizes = []  # Source queue capacity.
        self.local_queue_size = 0  # Ideally this should be at least `batch_size`.
        self.parallel_read_size = 1
        self.seed = None  # Optional graph-level random seed.
        self.is_chief = False
        self.num_replicas = 1  # The number of servers whose job name is 'worker'.
        self.saved_vars = []  # Variables saved and restored from checkpoints.
        self.saver = None
        self.server = None
        self.supervisor = None
        self.cluster_spec = None  # Initialized from the server.server_def.cluster
        self.session = None  # Managed session from the supervisor.
        self._local_queue_size_tensor = None
        self._source_queue_size_tensors = {}
        self._tf_record_readers = []  # Only used internally to reuse record readers.

    def _optimizer(self, learning_rate):
        # TODO(daeyun): consider setting `use_locking`.
        return tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=1e-6)

    def default_placeholders(self) -> typing.Sequence[tf.Tensor]:
        return [
            tf.placeholder(tf.float32, shape=(), name='learning_rate'),  # Used for setting learning rates.
            tf.placeholder_with_default(tf.constant(False, name='kFalse'), shape=(), name='is_training'),
            tf.placeholder(tf.int32, shape=(), name='batch_size'),
        ]

    def default_variables(self):
        return [
            tf.get_variable('global_step', shape=(), initializer=tf.constant_initializer(0, dtype=tf.int64), dtype=tf.int64, trainable=False),
            tf.get_variable('learning_rate', shape=(), initializer=tf.constant_initializer(0.001, dtype=tf.float32), dtype=tf.float32, trainable=False),
        ]

    def _init_summaries(self):
        if self.log_dir:
            self._summary_tensors = {}
            for name in self._summary_keys:
                fullname = self._summary_key_prefix + name
                self._summary_tensors[name] = tf.merge_all_summaries(key=fullname)

            # Graph is only added to the 'train' summary file.
            self._summary_writers['train'] = self._summary_writer('train', graph=tf.get_default_graph())

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

        placeholder_specs = list(filter(lambda item: 'QueuePlaceholder' in item.__class__.__name__, user_placeholders))

        if not placeholder_specs:
            return

        batch_size = self.placeholder('batch_size', fail=True)
        assert isinstance(batch_size, tf.Tensor)

        def build_queue(name, size):
            source_name = '{}/{}'.format(NNModel._source_queue_prefix, name)
            queue_components = self._build_source_queue(placeholder_specs,
                                                        queue_name=source_name,
                                                        queue_size=size,
                                                        enqueue_device=input_queue_device)

            return queue_components

        all_queue_components = [build_queue(name, size) for name, size in zip(queue_names, queue_sizes)]

        consumer_name = '{}'.format(NNModel._consumer_queue_prefix)

        tensor_dict = self._build_batch_tensors(name=consumer_name,
                                                batch_size=batch_size,
                                                queue_size=local_queue_size,
                                                placeholder_specs=placeholder_specs,
                                                all_source_queue_components=all_queue_components,
                                                local_device=self.local_device_name())
        log.info('Built local queue: %s', tensor_dict)

        with tf.name_scope('placeholder/'):
            for i, (field_name, tensor) in enumerate(tensor_dict.items()):
                tf.placeholder_with_default(tensor, shape=tensor.get_shape(), name=field_name)

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

    def build_debug_mode(self, sess, seed=None):
        self.is_debug_mode = True
        self.graph = sess.graph
        self.job_name = 'worker'
        self.task_id = 0
        with self.graph.as_default():
            if seed is not None:
                # Graph-level random seed.
                tf.set_random_seed(seed)
            self._build_input_pipeline('train', [100], 100)
            loss = self._model()
            assert isinstance(loss, tf.Tensor), '_model() must return a Tensor for the total loss.'
            optimizer = self._optimizer(self.variable('learning_rate'))

            with tf.name_scope('optim') as scope:
                minimize_op = optimizer.minimize(loss, global_step=self.variable('global_step'))

                update_ops = self.graph.get_collection(tf.GraphKeys.UPDATE_OPS)

                with tf.control_dependencies([minimize_op]):
                    minimize_op = tf.group(*update_ops, name='minimize')

            self.session = sess
            sess.run(tf.initialize_all_variables())

            self.needs_initialization = False

            self._init_summaries()  # Sets self._summary_tensors

        log.info('Initialized')

    def build(self,
              server=None,
              job_name=None,
              source_queue_names=('train',),
              source_queue_sizes=(2000,),
              local_queue_size=150,
              parallel_read_size=20,
              seed=None,
              sync_replicas=True,
              save_model_secs=600):
        assert self.needs_initialization

        if server is None:
            log.info('`tf.train.Server` was not provided. Creating a local server.')
            server = tf.train.Server.create_local_server()

        assert isinstance(server, tf.train.Server)
        self.server = server

        self.queue_names = ensure_list_or_tuple(source_queue_names)
        self.queue_sizes = ensure_list_or_tuple(source_queue_sizes)
        self.local_queue_size = local_queue_size
        assert len(self.queue_names) == len(self.queue_sizes)
        self.seed = seed
        self.parallel_read_size = parallel_read_size

        self.task_id = server.server_def.task_index

        if job_name is None:
            job_name, _ = distributed.job_info_from_server_def(server.server_def)
        self.job_name = job_name

        if self.log_dir:
            log.add_file_handler(path.join(self.log_dir, 'out.log'))

        self.cluster_spec = tf.train.ClusterSpec(server.server_def.cluster)

        if self.job_name == 'ps':
            raise ValueError('Parameter servers should call server.join() instead of using this class.')

        assert self.job_name in ('worker', 'data', 'local')

        if self.job_name == 'data':
            sync_replicas = False

        self.is_chief = self.task_id == 0 and self.job_name in ('worker', 'local')

        job_names = set(self.cluster_spec.jobs)
        assert ('worker' in job_names) != ('local' in job_names)
        trainer_job_name = 'worker' if 'worker' in job_names else 'local'

        # Here the number of replicas is the same as worker tasks, but technically that's not a requirement.
        self.num_replicas = self.cluster_spec.num_tasks(trainer_job_name)

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

                    update_ops = self.graph.get_collection(tf.GraphKeys.UPDATE_OPS)
                    with tf.control_dependencies([minimize_op]):
                        minimize_op = tf.group(*update_ops, name=scope)

                queue_runners = []
                init_tokens_op = None
                if self.is_chief:
                    if sync_replicas:
                        queue_runners.append(optim.get_chief_queue_runner())
                        queue_runners.extend(self.graph.get_collection(tf.GraphKeys.QUEUE_RUNNERS))
                        init_tokens_op = optim.get_init_tokens_op()

            with tf.device(self.variable('global_step').device):
                is_chief_ready = tf.Variable(False, trainable=False, dtype=tf.bool, name='is_chief_ready')
                set_chief_ready = tf.assign(is_chief_ready, True, use_locking=True, name='set_chief_ready')

            # Exclude any queue-specific variables.
            # TODO(daeyun): exclude local steps.
            vars_to_save = []
            for var in self.get_collections([tf.GraphKeys.TRAINABLE_VARIABLES, tf.GraphKeys.VARIABLES]):
                for name in self.queue_names:
                    if var.name.startswith('{}/{}/'.format(self._source_queue_prefix, name)) or 'is_chief_ready' in var.name:
                        break
                else:
                    vars_to_save.append(var)

            vars_to_init = {tf.is_variable_initialized(v): v for v in tf.all_variables()}

            self.saver = tf.train.Saver(
                name=self._saver_prefix,
                max_to_keep=10,
                keep_checkpoint_every_n_hours=0.5,
                var_list=vars_to_save,
            )
            self.saved_vars = vars_to_save

            # Variables created after this point will not be saved.

            init_op = tf.initialize_all_variables()
            local_init_op = tf.initialize_local_variables()

            if self.is_chief:
                self._init_summaries()  # Sets self._summary_tensors

            if path.isfile(self.checkpoint_filename()):
                log.info('{} exists. Model will be restored.'.format(self.checkpoint_filename()))

            # Create a "supervisor", which oversees the training process.
            self.supervisor = tf.train.Supervisor(is_chief=self.is_chief,
                                                  logdir=self.log_dir,
                                                  init_op=init_op,
                                                  ready_op=None,
                                                  summary_op=None,
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

        # Data processes should not be able to consume from the source queue.
        if self.job_name == 'data':
            self.session.run(self.operation('{}/close'.format(self._consumer_queue_prefix)))

        if self.is_chief:
            assert isinstance(sess, tf.Session)
            is_initialized = self.session.run([value for value in vars_to_init.keys()])
            uninitialized_vars = [var for is_init, var in zip(is_initialized, vars_to_init.values()) if not is_init]
            self.session.run([v.initializer for v in uninitialized_vars])

            log.info('Variables not restored from checkpoint were initialized from scratch: %s',
                     [v.name for v in uninitialized_vars])

            if init_tokens_op is not None:
                sess.run(init_tokens_op)

            self.supervisor.start_queue_runners(sess, queue_runners)

            log.info('Initialized chief.')

        sess.run(local_init_op)
        self.needs_initialization = False
        log.info('Initialized local variables.')

        if self.is_chief and self.log_dir:
            graph_utils.save_graph_text_summary(self.graph, dirname=self.log_dir, verbose=True)

        if self.is_chief:
            self.session.run(set_chief_ready)

        while True:
            try:
                log.info('global_step: %d', self.eval('global_step'))
                log.info('learning_rate: %f', self.eval('learning_rate'))
                self.eval('is_chief_ready')
            except tf.errors.FailedPreconditionError as ex:
                assert not self.is_chief
                time.sleep(0.2)
            except Exception as ex:
                log.error('Unexpected exception: {}.'.format(ex))
                raise ex
            else:
                break

        if self.job_name in ('worker', 'local'):
            while not self.session.run(is_chief_ready):
                print('{}:{}'.format(self.job_name, self.task_id), end=' ', flush=True)
                time.sleep(1.0)
            log.info('All variables are initialized.')

    def checkpoint_filename(self):
        return path.join(self.log_dir, self._checkpoint_basename)

    def _build_source_queue(self,
                            placeholder_specs,
                            queue_name,
                            queue_size,
                            enqueue_device):
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
        :param enqueue_device: Device for the enqueue operations. This should be a shared device. e.g. parameter server.
        :param dequeue_device: Device for receiving dequeued items and file IO operations. This should be a local device. e.g. worker.
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
        for input_spec in placeholder_specs:
            assert input_spec.shape[0] is None

            if input_spec.is_file:
                dtype = tf.string
                shape = [input_spec.shape[0]]
            else:
                dtype = input_spec.dtype
                shape = input_spec.shape

            # Placeholders for global enqueue ops.
            with tf.name_scope('placeholder/'):
                name = '{}/{}'.format(queue_name, input_spec.name)
                assert self.placeholder(name, fail=False) is None
                placeholder = tf.placeholder(dtype=dtype, shape=shape, name=name)

            names.append(input_spec.name)
            dtypes.append(dtype)
            shapes.append(shape[1:])
            placeholders.append(placeholder)

        with tf.device(enqueue_device):
            queue = tf.FIFOQueue(queue_size, dtypes=dtypes, shapes=shapes,
                                 names=names, shared_name=queue_name, name=queue_name)
            enqueue_tensors = {name: placeholder for name, placeholder in zip(names, placeholders)}

            with tf.name_scope('{}/'.format(queue_name)):
                # e.g. `queue/train/size:0`.
                self._source_queue_size_tensors[queue_name.rsplit('/', 1)[-1]] = queue.size(name='size')
                queue.close(True, name='close')
                # `enqueue_op` is an atomic operation. Values for all of `placeholder/{queue_name}/*` need to be specified at runtime.
                enqueue_op = queue.enqueue_many(enqueue_tensors, name='enqueue')

        return QueueComponents(name=queue.name, enqueue_op=enqueue_op, placeholders=placeholders, queue=queue)

    def _build_local_fetch_op(self, source_queue, target_queue, placeholder_specs, fetch_device, parallel_read_size):
        # `source_queue` must be a queue that dequeues a dictionary of tensors.
        assert isinstance(parallel_read_size, int)
        assert len(source_queue.names) > 0
        assert source_queue.names == target_queue.names

        input_specs_by_name = collections.defaultdict(list)
        for input_spec in placeholder_specs:
            assert input_spec.name in source_queue.names
            if input_spec.name in input_specs_by_name:
                assert input_spec.is_file and input_spec.tf_record_key is not None
            # Multiple TFRecord placeholders can share the same name but should have different `tf_record_key` values.
            for previous_item in input_specs_by_name[input_spec.name]:
                assert input_spec.tf_record_key != previous_item.tf_record_key
            input_specs_by_name[input_spec.name].append(input_spec)

        queue_name = source_queue.name.rsplit('/', 1)[-1]

        with tf.device(fetch_device):
            # The dequeue operation itself will still happen on `enqueue_device`.
            dequeue_tensors = source_queue.dequeue_many(parallel_read_size, name='dequeue')

            # Maps (placeholder_name -> (tf_record_key -> tf.FIFOQueue))
            options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.ZLIB)

            with tf.name_scope('{}/record_readers/'.format(self._consumer_queue_prefix)) as tf_record_scope:
                while parallel_read_size > len(self._tf_record_readers):
                    r = tf.TFRecordReader(name='reader', options=options)
                    self._tf_record_readers.append(r)

            assert isinstance(dequeue_tensors, dict)

            # File IO should happen on `dequeue_device`.
            tensors = {}
            for placeholder_name, value in dequeue_tensors.items():
                # It is a `list` because multiple TFRecord placeholders with different `tf_record_key`s can share the same name.
                input_specs = input_specs_by_name[placeholder_name]
                tf_record_fields = {spec for spec in input_specs if spec.tf_record_key is not None}
                npz_fields = {spec for spec in input_specs if spec.tf_record_key is None and spec.is_file}
                direct_feed = set(input_specs) - tf_record_fields - npz_fields
                assert len(npz_fields) <= 1 and len(direct_feed) <= 1
                assert 1 == sum([len(npz_fields) > 0, len(direct_feed) > 0, len(tf_record_fields) > 0])

                if npz_fields:
                    # Filename should end with '.npz'.
                    # For now, assume they always have only one key named 'data'.
                    assert len(npz_fields) == 1
                    input_spec = npz_fields.pop()
                    value = nn_ops.npz_to_tensor(value, dtype=input_spec.dtype, shape=[parallel_read_size] + list(input_spec.shape[1:]), key='data')
                    tensors[placeholder_name] = value
                elif tf_record_fields:
                    # TODO(daeyun): Refactor this block into a separate function.
                    # Filename should end with '.tfrecords'.
                    # One queue for each filename.
                    with tf.name_scope(tf_record_scope), tf.name_scope(placeholder_name):
                        assert value.get_shape()[0].value == parallel_read_size
                        parallel_serialized = []
                        for i, reader in enumerate(self._tf_record_readers[:parallel_read_size]):
                            q = tf.FIFOQueue(capacity=parallel_read_size, dtypes=tf.string, name='filename_queue/{}'.format(i))
                            # Enqueue and dequeue immediately. `tf.TFRecordReader.read` requires a queue as input.
                            # `value` is one of the filename values atomically dequeued from the source queue.
                            with tf.control_dependencies([q.enqueue(value[i])]):
                                _, serialized = reader.read(q, name='serialized_record')
                                parallel_serialized.append(serialized)

                        features = {}
                        needs_casting = {}
                        for input_spec in input_specs:
                            needs_casting[input_spec.tf_record_key] = (
                                input_spec.dtype.name not in ('float32', 'int64', 'string'))
                            if needs_casting[input_spec.tf_record_key]:
                                # Assume other types are stored in a ByteList.
                                shape, dtype = [1], tf.string  # There should be only one item.
                            else:
                                # dtypes matching protobuf types.
                                shape, dtype = input_spec.shape[1:], input_spec.dtype

                            # TODO(daeyun): Support variable length.
                            features[input_spec.tf_record_key] = tf.FixedLenFeature(
                                shape=shape, dtype=dtype, default_value=None)

                        parsed_tensors = tf.parse_example(parallel_serialized, features=features, name='parsed_tensors')
                        assert isinstance(parsed_tensors, dict)
                        for input_spec in input_specs:
                            with tf.name_scope(input_spec.tf_record_key):
                                # There might be some unused fields if they were not specified by the user.
                                tensor = parsed_tensors[input_spec.tf_record_key]
                                if needs_casting[input_spec.tf_record_key]:
                                    if input_spec.dtype.name == 'bool':
                                        # byte alignment.
                                        decoded = tf.decode_raw(bytes=tensor, out_type=tf.uint8)
                                        tensor = tf.cast(decoded, dtype=tf.bool)
                                    else:
                                        # Assume little-endian. NOTE: Network byte order is big-endian.
                                        tensor = tf.decode_raw(bytes=tensor, out_type=input_spec.dtype, little_endian=True)
                                # Raises error at runtime if shapes are incompatible.
                                tensor = tf.reshape(tensor, shape=[parallel_read_size] + list(input_spec.shape[1:]))
                            tensors[placeholder_name] = tensor
                else:
                    assert len(direct_feed) == 1
                    input_spec = direct_feed.pop()
                    # Sanity check.
                    value.get_shape().assert_is_compatible_with(input_spec.shape)
                    # `value` is a tensor (rather than a filename) directly fed into the queue.
                    tensors[placeholder_name] = value

        # Enqueue to the local destination queue.
        enqueue_op = target_queue.enqueue_many(tensors, name='enqueue')
        return enqueue_op

    def join_local_queue_runner_threads(self, name=None):
        if name is None:
            name = self._current_source_queue_name
        qr = self._consumer_queue_runners[name]
        # assert isinstance(qr, CountingQueueRunner)
        qr.join()

    def start_local_queue_runners(self, queue_name, request_size, num_threads=1) -> CountingQueueRunner:

        """
        Starts queue runner threads for fetching data from the global queue.

        :param queue_name: Name of the source queue. e.g. 'train' to ask for training data.
        :param request_size: Number of items needed.
        :param num_threads: Number of threads running dequeue/enqueue operations.
        :return:
        """
        qr = self._consumer_queue_runners[queue_name]
        # assert isinstance(qr, CountingQueueRunner)
        assert len(qr.running_threads()) == 0

        self._current_source_queue_name = queue_name
        current_worker_queue_size = self.current_queue_size()
        if current_worker_queue_size != 0:
            raise RuntimeError('Worker queue size is not 0: {}'.format(current_worker_queue_size))

        self._consumer_threads[queue_name] = qr.start_threads(self.session, request_size, num_threads)
        # TODO(daeyun): check dequeue from the local queues still work after global dequeue halts.

        return qr

    def _build_batch_tensors(self, name, batch_size, queue_size, placeholder_specs, all_source_queue_components: typing.Sequence[QueueComponents], local_device):
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
            # One local queue, multiple enqueue operations.

            types, shapes, names = [], [], []
            for spec in placeholder_specs:
                # assert isinstance(spec, QueuePlaceholder)
                types.append(spec.dtype)
                shapes.append(spec.shape[1:])
                names.append(spec.name)
            queue = tf.FIFOQueue(capacity=queue_size, dtypes=types, shapes=shapes, names=names, shared_name=None, name='queue')

            # `worker_queue/size:0`.
            self._local_queue_size_tensor = queue.size('size')
            queue.close(True, name='close')

            # `worker_queue/dequeue:*`.
            batch_tensors = queue.dequeue_many(batch_size, name='dequeue')

            for source_queue_components in all_source_queue_components:
                basename = source_queue_components.name.rsplit('/', 1)[-1]

                with tf.variable_scope('{}'.format(basename)):
                    # Enqueue to the local destination queue.
                    parallel_enqueue_op = self._build_local_fetch_op(source_queue_components.queue, queue, placeholder_specs, local_device,
                                                                     parallel_read_size=self.parallel_read_size)
                    enqueue_op = self._build_local_fetch_op(source_queue_components.queue, queue, placeholder_specs, local_device, parallel_read_size=1)
                    qr = CountingQueueRunner(queue, enqueue_op=enqueue_op, parallel_enqueue_op=parallel_enqueue_op,
                                             parallel_enqueue_size=self.parallel_read_size, name=basename)
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
                # Assumes all variables belong to at least one collection.
                variables = self.get_collections([tf.GraphKeys.VARIABLES,
                                                  tf.GraphKeys.LOCAL_VARIABLES,
                                                  tf.GraphKeys.MODEL_VARIABLES,
                                                  tf.GraphKeys.TRAINABLE_VARIABLES,
                                                  tf.GraphKeys.MOVING_AVERAGE_VARIABLES])
                for variable in variables:
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
            try:
                assert isinstance(name, str)
            except Exception as ex:
                if fail:
                    raise ex
                results.append(None)
                continue

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
              check_placeholder_coverage=True,
              run_options=None):
        """
        Runs a training step.

        :param feed_dict: A dictionary that maps g elements to values. Keys can be regular expressions
        or placeholder objects.
        :param summary_keys: A list of summary modes, 'SIMPLE', 'ALL', 'IMAGE', etc. Can be empty (default).
        :param check_placeholder_coverage: If `True`, raise an exception when `feed` contains an optional placeholder
        and there are any remaining optional placeholders that depend on the same queue.
        """
        assert self.job_name in ('worker', 'local')
        # summary_writer_name should be 'train' by default.
        self.eval([], feed_dict=feed_dict, is_training=True, summary_keys=summary_keys, check_placeholder_coverage=check_placeholder_coverage, run_options=run_options)

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
        assert abs((learning_rate - new_learning_rate) / learning_rate) < 1e-7

    def current_learning_rate(self):
        return self.session.run(self.variable('learning_rate'))

    def current_global_step(self):
        return self.session.run(self.variable('global_step'))

    def has_data(self):
        qr = self._consumer_queue_runners[self._current_source_queue_name]
        return qr.remaining_enqueue() > 0 or len(qr.running_threads()) > 0 or self.current_queue_size() > 0

    def eval(self,
             values=None,
             feed_dict=None,
             collection_keys=None,
             summary_keys=None,
             summary_writer_name=None,
             is_training=False,
             check_placeholder_coverage=True,
             assert_no_dequeue=False,
             run_options=None):
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

        # TODO(daeyun): warn if fetching from the queue and there is no active local queue runner or local queue size is 0.
        # TODO(daeyun): friendlier error message. e.g. wrong dtype in QueuePlaceholder. Should be easier to recognize.

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
                log.warn('Non-default empty `summary_keys`. This was probably not expected.')
        summary_keys = ensure_list_or_tuple(summary_keys, str)

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
        # here in most use cases. Use `NNModel.enqueue`.
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

        # Only the chief process computes summaries.
        if self.is_chief:
            summary_ops = self.summary_tensors(summary_keys)
        else:
            summary_ops = ()
        summary_op_fetch_indices = (len(fetches), len(fetches) + len(summary_ops))
        if summary_ops:
            fetches.extend(summary_ops)

        assert len(fetches) > 0, '`fetches` cannot be empty.'

        dequeuing_value = None
        for item in fetches:
            # This is cached but may be slow the first time.
            placeholder_names = frozenset(k.op.name for k in new_feed_dict.keys() if 'placeholder' in k.op.name)
            if self.has_dequeue_dependency(item.name, placeholder_names):
                dequeuing_value = item
                break
        will_dequeue = dequeuing_value is not None and not self.is_debug_mode

        if assert_no_dequeue and will_dequeue:
            raise ValueError('assert_no_dequeue is True and {} has a dequeue dependency.'.format(dequeuing_value.name))

        if check_placeholder_coverage and will_dequeue:
            check_feed_queue_coverage(new_feed_dict)

        if will_dequeue:
            batch_size_ph = self.placeholder('batch_size')
            if batch_size_ph not in new_feed_dict:
                raise ValueError('{} has a dequeue dependency and batch_size placeholder is missing.'.format(dequeuing_value.name))
            batch_size_value = new_feed_dict[batch_size_ph]

            if self._current_source_queue_name is None:
                raise BlockingIOError('Current source queue is not specified. Did you forget to call `.start_local_queue_runners()`?')
            i = 0
            while True:
                local_queue_size = self.current_queue_size()
                if local_queue_size < batch_size_value:
                    log.warn('IO bottleneck detected: worker queue size {} is less than batch size {}.'.format(local_queue_size, batch_size_value))

                if local_queue_size > 0:
                    break
                if i == 1:
                    for thread in self._consumer_threads[self._current_source_queue_name]:
                        assert isinstance(thread, threading.Thread)
                        if thread.is_alive():
                            break
                    else:
                        raise BlockingIOError('Worker queue is empty and queue runner threads are not running. Did you forget to call `.start_local_queue_runners()`?')
                log.info('Worker queue is empty. Waiting for data from global queue "{}" (size {}).'.format(
                    self._current_source_queue_name, self.source_queue_size(self._current_source_queue_name)))
                time.sleep(min(0.4 * (i + 1), 3))
                i += 1

        if run_options is None:
            self.run_metadata = None
        else:
            self.run_metadata = tf.RunMetadata()

        with self.graph.as_default():
            out_eval = self.session.run(fetches, new_feed_dict, options=run_options, run_metadata=self.run_metadata)

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

        if is_single_value:
            return out_eval[0]

        # `tensors_or_patterns` and `out_eval` may not be the same size.
        results = {}
        for name, result in zip(values, out_eval):
            if result is not None:
                results[name] = result
        return results

    def flush_summary_writers(self, names=None):
        if not self.is_chief:
            return
        if names is None:
            names = tuple(self._summary_writers.keys())
        names = ensure_list_or_tuple(names, str)
        for name in names:
            assert name in self._summary_writers
            writer = self._summary_writers[name]
            assert isinstance(writer, tf.train.SummaryWriter)
            writer.flush()

    def write_scalar_summary(self, tag, value, summary_writer_name, flush=False):
        """
        Directly writes a scalar Python value as a summary protobuf. Creates a new summary writer if one matching `summary_writer_name` does not already exist.
        """
        if not self.is_chief:
            return
        assert isinstance(tag, str)
        assert isinstance(summary_writer_name, str)
        assert isinstance(value, numbers.Real)

        summary = tf.Summary(value=[
            tf.Summary.Value(tag=tag, simple_value=value),
        ])
        writer = self._summary_writer(summary_writer_name)
        global_step = self.current_global_step()
        writer.add_summary(summary, global_step)

        if flush:
            writer.flush()

        return writer

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

        tensors_by_names = {t.name: t for t in summaries}
        return list(tensors_by_names.values())

    def current_queue_size(self):
        assert self.job_name in ('worker', 'local')
        return self.session.run(self._local_queue_size_tensor)

    def source_queue_size(self, name):
        assert name in self._source_queue_size_tensors
        return self.session.run(self._source_queue_size_tensors[name])

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

        directory = path.dirname(save_path)
        if not path.isdir(directory):
            log.info('mkdir %s', directory)
            os.makedirs(directory)

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
        assert set(summary_keys).issubset(self._summary_keys)

        with_default_summary_keys = list(summary_keys) + ['scalar']
        collection_keys = [self._summary_key_prefix + key for key in with_default_summary_keys]
        log.info('Adding scalar summary tag %s to collections %s.', tag, ','.join(collection_keys))
        summary_op = tf.scalar_summary(tag, value, collections=collection_keys, name=name)
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
        assert set(summary_keys).issubset(self._summary_keys)

        with tf.name_scope('histogram_summary_value') as scope:
            concat_value = tf.concat(0, [tf.reshape(v, [-1]) if v.get_shape().ndims > 0 else v for v in values], name=scope)

        with_default_summary_keys = list(summary_keys) + ['histogram']
        collection_keys = [self._summary_key_prefix + key for key in with_default_summary_keys]
        log.info('Adding histogram summary tag %s to collection %s', tag, ','.join(collection_keys))
        summary_op = tf.histogram_summary(tag=tag, name=name, values=concat_value, collections=collection_keys)
        return summary_op

    def add_image_summary(self, tag: str, value: tf.Tensor, max_images: int = 3, summary_keys=(), name: str = None) -> tf.Tensor:
        """
        A wrapper around `tf.image_summary` that prints out logs and adds to the image collection.

        :param tag: A scalar `Tensor` of type `string`. Used to build the `tag` of the summary values.
        :param value: A 4-D `uint8` or `float32` `Tensor` of shape `[batch_size, height, width, channels]` where `channels` is 1, 3, or 4.
        :param max_images: Max number of batch elements to generate images for.
        :param name: Will be the same as `tag` by default.
        :param collection_keys: Graph collection keys to which this operation is added. 'image_summaries' is included by default.
        :return: A scalar `Tensor` of type `string`. The serialized `Summary` protocol buffer.
        """
        if name is None:
            name = tag + '_summary'
        ensure_list_or_tuple(summary_keys, str)
        for summary_key in summary_keys:
            if summary_key not in self._summary_keys:
                self._summary_keys.append(summary_keys)
                log.info('Adding a new summary key: %s', summary_key)
        assert set(summary_keys).issubset(self._summary_keys)

        shape = value.get_shape().as_list()
        assert len(shape) == 4
        assert shape[-1] in [1, 3, 4]

        with_default_summary_keys = list(summary_keys) + ['image']
        collection_keys = [self._summary_key_prefix + key for key in with_default_summary_keys]
        log.info('Adding image summary tag %s of shape %s to collections %s. max_images: %d', tag, shape, ','.join(collection_keys), max_images)
        summary_op = tf.image_summary(tag, tensor=value, max_images=max_images, collections=collection_keys, name=name)
        return summary_op

    @functools.lru_cache()
    def has_dequeue_dependency(self, tensor_or_operation_name, placeholder_names):
        assert not self.needs_initialization
        assert isinstance(placeholder_names, frozenset)
        if ':' in tensor_or_operation_name:
            tensors = [self.graph.get_tensor_by_name(tensor_or_operation_name)]
        else:
            op = self.graph.get_operation_by_name(tensor_or_operation_name)
            tensors = list(op.inputs)
        target_name = '{}/dequeue'.format(self._consumer_queue_prefix)
        for v in graph_utils.find_dependencies(tensors):
            if target_name in v.name and len(placeholder_names.intersection([item.name for item in v.consumers()])) == 0:
                return True


class SampleModel(NNModel):
    def _model(self):
        out = nn_ops.batch_norm(self.placeholder('input'), is_trainable=True, is_local=True)
        loss = tf.reduce_mean(tf.square(tf.sub(out, self.placeholder('target'))), name='loss')
        return loss

    def _placeholders(self):
        return [
            QueuePlaceholder(tf.float32, shape=[None, 5, 5, 1], name='input', is_file=True),
            QueuePlaceholder(tf.float32, shape=[None, 5, 5, 1], name='target', is_file=True),
        ]
