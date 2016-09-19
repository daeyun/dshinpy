import threading
import time
import collections
import pprint
import functools
from os import path
import socket
import py
import os

import multiprocessing
from dshin import log
import numpy as np
import pytest
import tensorflow as tf
from py._path import local
import typing

from dshin import nn
from dshin.nn import model_utils
import dshin

pytestmark = pytest.mark.skipif(True, reason='Temporarily disabled')


class Net(model_utils.NNModel):
    def _model(self):
        out = self.placeholder('input')
        out = nn.ops.conv2d(out, 5, k=3, s=1, name='conv2d', use_bias=False)
        out = nn.ops.batch_norm(out, is_trainable=True, is_local=True)
        loss = tf.reduce_mean((out - self.placeholder('target')) ** 2, name='loss')
        return loss

    def _placeholders(self):
        return [
            model_utils.QueuePlaceholder(tf.float32, shape=[None, 5, 5, 1], name='input', is_file=False),
            model_utils.QueuePlaceholder(tf.float32, shape=[None, 5, 5, 1], name='target', is_file=True),
        ]


def retry_until(func, sec, args=(), sleep=0.05):
    start_time = time.time()
    i = 0
    while True:
        try:
            func(*args)
            break
        except Exception as ex:
            if time.time() - start_time > sec:
                raise ex
            time.sleep(sleep)
            i += 1


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


queue_names = ['train', 'train2']


def ps_worker(cluster_spec, task_id, values):
    config = nn.utils.default_sess_config(
        log_device_placement=False, mem=0.05,
        # device_filters=["/job:ps", "/job:worker/task:%d" % FLAGS.task_index]
    )
    server = tf.train.Server(cluster_spec,
                             job_name='ps',
                             task_index=task_id,
                             config=config)

    server.join()
    log.info('ps_worker is quitting.')


class Worker(multiprocessing.Process):
    def __init__(self, cluster_spec, job_name, task_id, shared_values, barrier=None, sync=True):
        super(Worker, self).__init__()
        self.task_id = task_id
        self.job_name = job_name
        self.shared_values = shared_values
        self._barrier = barrier
        self.cluster_spec = cluster_spec
        self.sync = sync
        self.tmpdir = py.test.ensuretemp('{}_{}'.format(job_name, self.task_id))
        self.num_workers = shared_values['num_processes'][self.job_name]
        self.shared_values[self.task_id] = collections.defaultdict(list)
        np.random.seed(task_id)

    @functools.lru_cache()
    def data(self, seed=42):
        np.random.seed(seed)
        d = {
            'input': np.ones((2, 5, 5, 1)),
            'target': 2 * np.ones((2, 5, 5, 1)).astype(np.float32),
        }
        d['input'][1] += 1
        d['target'][1] += 1
        return d

    def add_value(self, list_name, value):
        values = self.shared_values[self.task_id]
        values[list_name].append(value)
        self.shared_values[self.task_id] = values

    def values(self, list_name):
        return self.shared_values[self.task_id][list_name]

    def barrier(self, name, timeout=15):
        while True:
            start_time = time.time()
            try:
                i = self._barrier.wait(timeout=timeout)
            except threading.BrokenBarrierError  as ex:
                if time.time() - start_time < timeout:
                    time.sleep(0.03)
                    continue
                else:
                    raise ex
            break
        if i == 0:
            self._barrier.reset()
            self.log('All processes passed the "{}" barrier.'.format(name))
        assert not self._barrier.broken

    def log(self, *args):
        log.info('[{} %d] {}'.format(self.job_name, ' '.join(['%s' for _ in range(len(args))])), self.task_id, *args)

    def feed_dict(self):
        data = self.data()
        return {
            'input': data['input'],
            'target': data['target'],
        }

    def build(self, data_pipeline_only=False):
        graph = tf.Graph()
        net = Net(graph, log_dir='/tmp/tf_log_dir_{}_{}'.format(self.job_name, self.task_id))

        config = nn.utils.default_sess_config(
            log_device_placement=False, mem=0.05,
            # device_filters=["/job:ps", "/job:worker/task:%d" % FLAGS.task_index]
        )
        server = tf.train.Server(self.cluster_spec,
                                 job_name=self.job_name,
                                 task_index=self.task_id,
                                 config=config)
        assert isinstance(net, model_utils.NNModel)

        net.build(server, source_queue_names=queue_names, source_queue_sizes=[1000, 5],
                  local_queue_size=150,
                  seed=42, sync_replicas=self.sync, save_model_secs=1)
        return net

    def run(self):
        net = self.build()

        self.barrier('build')
        time.sleep(1)

        net.start_local_queue_runner('train', 40, num_threads=30)
        net.join_local_queue_runner_threads('train')
        assert net.eval('worker_queue/size') == 40

        self.barrier('populate')
        assert net.eval('queue/train/count') == 40 * self.num_workers
        assert net.eval('queue/train2/count') == 0

        self.barrier('populate')
        net.start_local_queue_runner('train', 60, num_threads=10)
        net.join_local_queue_runner_threads('train')
        assert net.eval('worker_queue/size') == 100

        self.barrier('populate')
        assert net.eval('queue/train/count') == 100 * self.num_workers
        assert net.eval('queue/train2/count') == 0

        self.barrier('populate')
        net.start_local_queue_runner('train2', 5, num_threads=10)
        net.join_local_queue_runner_threads()
        assert net.eval('worker_queue/size') == 105

        self.barrier('populate')
        assert net.eval('queue/train/count') == 100 * self.num_workers
        assert net.eval('queue/train2/count') == 5 * self.num_workers

        self.barrier('prepare_queue')

        # Not joining until the end.
        net.start_local_queue_runner('train2', 5, num_threads=10)

        data = self.data()

        self.add_value('steps', net.eval('global_step'))
        self.add_value('losses', net.eval('loss', self.feed_dict()))

        self.barrier('train')
        net.train(self.feed_dict())
        self.barrier('train')

        self.add_value('steps', net.eval('global_step'))
        self.add_value('losses', net.eval('loss', self.feed_dict()))

        self.barrier('eval')
        self.add_value('steps', net.eval('global_step'))
        self.add_value('losses', net.eval('loss', self.feed_dict()))

        assert self.values('steps')[0] == 0
        if self.sync:
            assert self.values('steps')[1] == 1
        else:
            assert self.values('steps')[1] == self.num_workers

        assert self.values('steps')[2] == self.values('steps')[1]
        assert self.values('losses')[1] < self.values('losses')[0]
        assert not np.isclose(self.values('losses')[0], self.values('losses')[1])
        assert np.isclose(self.values('losses')[1], self.values('losses')[2])

        self.barrier('eval')

        for i in range(10):
            net.train(self.feed_dict())
            self.barrier('train_{}'.format(i))
            self.add_value('steps', net.eval('global_step'))
            assert self.values('steps')[-1] == self.values('steps')[-2] + (1 if self.sync else self.num_workers)

            self.add_value('losses', net.eval('loss', self.feed_dict()))
            assert self.values('losses')[-1] < self.values('losses')[-2]
            self.barrier('eval_{}'.format(i))

        if self.task_id == 0:
            new_lr = net.eval('learning_rate') * 0.1
            net.set_learning_rate(new_lr)
            assert np.isclose(net.eval('learning_rate'), new_lr)

        self.barrier('learning_rate')

        for i in range(10):
            net.train(self.feed_dict())
            self.barrier('train_{}'.format(i))
            self.add_value('steps', net.eval('global_step'))
            assert self.values('steps')[-1] == self.values('steps')[-2] + (1 if self.sync else self.num_workers)

            self.add_value('losses', net.eval('loss', self.feed_dict()))
            assert self.values('losses')[-1] < self.values('losses')[-2]
            self.barrier('eval_{}'.format(i))

        losses1 = np.array(self.values('losses')[-20:-10])
        losses2 = np.array(self.values('losses')[-10:])
        assert (losses1[:-1] - losses1[1:]).mean() > (losses2[:-1] - losses2[1:]).mean() * 2

        self.log('Now training from the local queue data.')
        assert net.eval('worker_queue/size') == 110

        expected_worker_queue_size = 110

        assert net.eval('queue/train/count') == 100 * self.num_workers
        assert net.eval('queue/train2/count') == 10 * self.num_workers
        self.barrier('source_queue_count')

        for i in range(10):
            num = np.random.randint(1, 5)
            net.train({
                'batch_size': num
            })
            expected_worker_queue_size -= num
            self.barrier('train_{}'.format(i))
            self.add_value('steps', net.eval('global_step'))
            assert self.values('steps')[-1] == self.values('steps')[-2] + (1 if self.sync else self.num_workers)

            self.add_value('losses', net.eval('loss', self.feed_dict()))
            assert self.values('losses')[-1] < self.values('losses')[-2]
            self.barrier('eval_{}'.format(i))

        assert net.eval('worker_queue/size') == expected_worker_queue_size
        assert net.eval('queue/train/count') == 100 * self.num_workers
        assert net.eval('queue/train2/count') == 10 * self.num_workers
        self.barrier('populate')

        net.start_local_queue_runner('train', 40, num_threads=1)
        net.join_local_queue_runner_threads('train')
        expected_worker_queue_size += 40
        assert net.eval('worker_queue/size') == expected_worker_queue_size

        self.barrier('populate')
        assert net.eval('queue/train/count') == 140 * self.num_workers
        assert net.eval('queue/train2/count') == 10 * self.num_workers

        self.log(self.values('steps'))
        self.log(self.values('losses'))

        self.barrier('eval')

        self.barrier('before_shutdown')
        net.shutdown()
        self.barrier('after_shutdown')


class DataProvider(Worker):
    def __init__(self, cluster_spec, job_name, task_id, shared_values, queue_name):
        super(DataProvider, self).__init__(cluster_spec, job_name, task_id, shared_values, None, False)
        self.queue_name = queue_name

    @functools.lru_cache()
    def filenames(self):
        data = self.data()
        tmpdir = self.tmpdir
        filenames = []
        filenames.append(str(tmpdir.join('target0.npz')))
        np.savez_compressed(filenames[0], data=data['target'][[0]].astype(np.float64))
        filenames.append(str(tmpdir.join('target1.npz')))
        np.savez_compressed(filenames[1], data=data['target'][[1]].astype(np.float32))
        return filenames

    def feed_dict(self):
        data = self.data()
        return {
            'input': data['input'],
            'target': self.filenames(),
        }

    def run(self):
        net = self.build()
        try:
            while True:
                net.enqueue(self.queue_name, self.feed_dict())
        except tf.errors.CancelledError:
            pass


def test_distributed_queue():
    for sync in [True, False]:
        os.system('rm -rf /tmp/tf_log_dir*')

        manager = multiprocessing.Manager()
        shared_values = manager.dict()

        num_processes = {
            'ps': 2,
            'worker': 5,
            'data': 2,
        }

        shared_values['num_processes'] = num_processes

        ps_processes = []
        worker_processes = []
        data_processes = []

        barrier = manager.Barrier(num_processes['worker'])

        cluster = get_local_cluster_spec(num_processes)
        log.info('Cluster %s', cluster.as_dict())

        for i in range(num_processes['ps']):
            ps_processes.append(multiprocessing.Process(target=ps_worker, args=[cluster, i, shared_values]))

        for i in range(num_processes['worker']):
            worker_processes.append(Worker(cluster, 'worker', i, shared_values, barrier, sync))

        for i in range(num_processes['data']):
            data_processes.append(DataProvider(cluster, 'data', i, shared_values, queue_name=queue_names[i]))

        for p in ps_processes + worker_processes + data_processes:
            assert isinstance(p, multiprocessing.Process)
            p.daemon = True
            p.start()
            log.info('Started pid %d (%s)', p.pid, p.name)

        log.info('Started all processes.')

        for p in worker_processes:
            p.join()

        for p in worker_processes:
            if p.exitcode != 0:
                raise ChildProcessError('Return code is not 0: {}'.format([p.exitcode for p in worker_processes]))

        log.info('All worker processes finished. Terminating parameter servers.')

        for p in ps_processes:
            p.terminate()

        for p in ps_processes:
            p.join()

        for tid, values_dict in shared_values.items():
            if not isinstance(tid, int):
                continue
            log.info('%d, %s', tid, dict(values_dict))
            assert set(shared_values[0].keys()) == set(values_dict.keys())
            for k, values in shared_values[0].items():
                log.info('Max difference was %f', np.abs(np.array(values_dict[k]) - np.array(values)).max())
                assert np.allclose(values_dict[k], values)

        log.info('All ps processes finished. Waiting for data servers to finish.')

        for p in data_processes:
            p.join()

        log.info('End of main thread.')
