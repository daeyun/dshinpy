import threading
import time
from os import path

import numpy as np
import pytest
import tensorflow as tf
from py._path import local

from dshin import nn


class BN(nn.utils.NNModel):
    def _model(self):
        out = nn.ops.batch_norm(self['input'], is_trainable=True, is_local=True)
        self._loss = tf.reduce_mean((out - self['target']) ** 2, name='loss')
        tf.scalar_summary('loss', self._loss, collections=nn.utils.NNModel.summary_keys('SIMPLE'))

    def _minimize_op(self):
        return tf.train.AdamOptimizer(self['learning_rate']).minimize(self._loss)

    def _placeholders(self):
        return [
            tf.placeholder(tf.int32, shape=[None, 6, 6, 2], name='not_queued'),
            nn.utils.QueuePlaceholder(tf.float32, shape=[None, 5, 5, 1], name='input', is_file=False),
            nn.utils.QueuePlaceholder(tf.float32, shape=[None, 5, 5, 1], name='target', is_file=True),
        ]


class BNWithoutFileInput(nn.utils.NNModel):
    def _model(self):
        out = nn.ops.batch_norm(self['input'], is_trainable=True, is_local=True)
        self._loss = tf.reduce_mean((out - self['target']) ** 2, name='loss')
        tf.scalar_summary('loss', self._loss, collections=nn.utils.NNModel.summary_keys('SIMPLE'))

    def _minimize_op(self):
        return tf.train.AdamOptimizer(self['learning_rate']).minimize(self._loss)

    def _placeholders(self):
        return [
            tf.placeholder(tf.int32, shape=[None, 6, 6, 2], name='not_queued'),
            nn.utils.QueuePlaceholder(tf.float32, shape=[None, 5, 5, 1], name='input', is_file=False),
            nn.utils.QueuePlaceholder(tf.float32, shape=[None, 5, 5, 1], name='target', is_file=False),
        ]


def retry_until(func, sec, sleep=0.05):
    start_time = time.time()
    i = 0
    while True:
        try:
            func()
            break
        except Exception as ex:
            if time.time() - start_time > sec:
                raise ex
            time.sleep(sleep)
            i += 1


@pytest.fixture(scope='function')
def net(tmpdir: local.LocalPath):
    return BN(seed=42, summary_dir=str(tmpdir.join('summary')))


@pytest.fixture(scope='function')
def net_no_file(tmpdir: local.LocalPath):
    return BNWithoutFileInput(seed=42, summary_dir=str(tmpdir.join('summary')))


@pytest.fixture(scope='module')
def data():
    np.random.seed(42)
    return {
        'input': np.random.randn(2, 5, 5, 1),
        'target': np.random.randn(2, 5, 5, 1),
    }


def test_queue_placeholder_shape():
    class Net(nn.utils.NNModel):
        def _model(self):
            pass

        def _minimize_op(self):
            return tf.no_op()

        def _placeholders(self):
            return [nn.utils.QueuePlaceholder(tf.float32, shape=[1, 5, 5, 1], name='input', is_file=False)]

    with pytest.raises(Exception):
        Net()


def test_placeholder(net: nn.utils.NNModel):
    assert net['not_queued'].dtype == tf.int32
    assert net['not_queued'].get_shape().as_list() == [None, 6, 6, 2]
    assert 2 * len(net.get_all('^placeholder/.*:0')) == len(net.get_all('^placeholder/.*'))


def test_queue_array_placeholder_properties(net: nn.utils.NNModel):
    assert net['input'].name == 'placeholder/input:0'
    assert net['input'].get_shape().as_list() == [None, 5, 5, 1]
    assert net['input'].dtype == tf.float32

    assert net['queue/input'].name == 'placeholder/queue/input:0'
    assert net['queue/input'].dtype == tf.float32
    assert net['queue/input'].get_shape().as_list() == [None, 5, 5, 1]


def test_queue_operators(net: nn.utils.NNModel):
    assert isinstance(net['queue/size'], tf.Tensor)
    assert isinstance(net['worker_queue/size'], tf.Tensor)

    assert isinstance(net['queue/enqueue'], tf.Operation)

    assert isinstance(net['queue/close'], tf.Operation)
    assert isinstance(net['worker_queue/close'], tf.Operation)

    assert isinstance(net['queue/size$'], tf.Operation)
    assert isinstance(net['worker_queue/size$'], tf.Operation)


def test_queue_file_placeholder_properties(net: nn.utils.NNModel):
    assert net['target'].name == 'placeholder/target:0'
    assert net['target'].get_shape().as_list() == [None, 5, 5, 1]
    assert net['target'].dtype == tf.float32

    assert net['queue/target'].name == 'placeholder/queue/target:0'
    assert net['queue/target'].dtype == tf.string
    assert net['queue/target'].get_shape().as_list() == [None]


def test_queue_placeholder_direct_feeding(net: nn.utils.NNModel, data):
    feed_dict = {
        'input': data['input'],
        'target': data['target'],
        'learning_rate': 0.001,
    }
    loss_prev = net.eval('loss', feed_dict)
    net.train(feed_dict)
    loss = net.eval('loss', feed_dict)
    assert loss < loss_prev

    loss_prev, loss = loss, net.eval('loss', feed_dict)
    assert loss == loss_prev

    net.train(feed_dict)
    loss_prev, loss = loss, net.eval('loss', feed_dict)
    assert loss < loss_prev


def test_init_queue_runners(net: nn.utils.NNModel):
    runners = net.graph.get_collection(tf.GraphKeys.QUEUE_RUNNERS)
    assert len(runners) == 1
    assert isinstance(runners[0], tf.train.QueueRunner)
    assert runners[0].name.startswith(nn.utils.NNModel._consumer_queue_prefix)
    assert len(net.queue_runner_threads) >= net._worker_queue_num_threads
    print(len(net.queue_runner_threads))

    for thread in net.queue_runner_threads:
        assert isinstance(thread, threading.Thread)
        assert thread.is_alive()
        assert thread.daemon


def test_shutdown(net: nn.utils.NNModel):
    runner = net.graph.get_collection(tf.GraphKeys.QUEUE_RUNNERS)[0]
    assert isinstance(runner, tf.train.QueueRunner)
    assert not net.coordinator.should_stop()

    def assert_alive():
        for thread in net.queue_runner_threads:
            assert thread.is_alive()

    retry_until(assert_alive, sec=1)

    net.shutdown()

    for thread in net.queue_runner_threads:
        assert not thread.is_alive()
    assert net.coordinator.joined


def test_enqueue_array(net_no_file: nn.utils.NNModel, data):
    net = net_no_file

    feed_dict = {
        'queue/input': data['input'][:1],
        'queue/target': data['target'][:1],
    }
    assert net.eval('worker_queue/size') == 0
    net.eval('queue/enqueue', feed_dict)
    assert net.eval('worker_queue/size') == 1
    net.eval('queue/enqueue', feed_dict)
    assert net.eval('worker_queue/size') == 2
    net.shutdown()


def test_dequeue_array_eval(net_no_file: nn.utils.NNModel, data):
    net = net_no_file

    feed_dict = {
        'queue/input': data['input'][[0, 0]],
        'queue/target': data['target'][[0, 0]],
    }
    assert net.eval('worker_queue/size') == 0
    net.eval('queue/enqueue', feed_dict)
    assert net.eval('worker_queue/size') == 2
    loss = net.eval('loss', {'batch_size': 1})
    assert net.eval('worker_queue/size') == 1
    assert net.eval('loss', {'batch_size': 1}) == loss
    assert net.eval('worker_queue/size') == 0


def test_dequeue_array_train(net_no_file: nn.utils.NNModel, data):
    net = net_no_file
    queue_feed_dict = {
        'queue/input': data['input'][[0] * 4],
        'queue/target': data['target'][[0] * 4],
    }
    eval_feed_dict = {
        'input': data['input'][[0, 0]],
        'target': data['target'][[0, 0]],
    }
    train_feed_dict = {
        'learning_rate': 0.001,
        'batch_size': 2,
    }
    assert net.eval('worker_queue/size') == 0
    net.eval('queue/enqueue', queue_feed_dict)
    assert net.eval('worker_queue/size') == 4
    loss1 = net.eval('loss', eval_feed_dict)
    assert net.eval('worker_queue/size') == 4

    net.train(train_feed_dict)
    assert net.eval('worker_queue/size') == 2
    loss2 = net.eval('loss', eval_feed_dict)
    assert loss2 < loss1
    assert net.eval('worker_queue/size') == 2

    net.eval('queue/enqueue', queue_feed_dict)
    assert net.eval('worker_queue/size') == 6

    train_feed_dict['batch_size'] = 6
    net.train(train_feed_dict)
    assert net.eval('worker_queue/size') == 0
    loss3 = net.eval('loss', eval_feed_dict)
    assert net.eval('worker_queue/size') == 0
    assert loss3 < loss2


def test_enqueue_integrity(net: nn.utils.NNModel, tmpdir, data):
    net = net
    npz_filename = str(tmpdir.join('target_array.npz'))
    np.savez_compressed(npz_filename, data=data['target'][[0]].astype(np.float32))
    assert path.isfile(npz_filename)

    feed_dict = {
        'queue/input': data['input'][[0] * 2],
        'queue/target': [npz_filename, npz_filename],
    }
    assert net.eval('worker_queue/size') == 0
    net.eval('queue/enqueue', feed_dict)

    def assert_enqueue():
        assert net.eval('worker_queue/size') == 2

    retry_until(assert_enqueue, sec=1)

    # NOTE: 'input' corresponds to index 0. This might be unstable.
    dequeue = net.eval(['worker_queue:0', 'worker_queue:1'], {net['batch_size']: 2})
    assert np.allclose(dequeue['worker_queue:0'], data['input'][[0] * 2])
    assert np.allclose(dequeue['worker_queue:1'], data['target'][[0] * 2])


def test_enqueue_array_and_filename(net: nn.utils.NNModel, tmpdir, data):
    net = net
    npz_filename = str(tmpdir.join('target_array.npz'))
    np.savez_compressed(npz_filename, data=data['target'][[0]].astype(np.float32))
    assert path.isfile(npz_filename)
    npz_filename2 = str(tmpdir.join('target_array2.npz'))
    np.savez_compressed(npz_filename2, data=data['target'][[1]].astype(np.float64))
    assert path.isfile(npz_filename2)

    feed_dict = {
        'queue/input': data['input'][:2],
        'queue/target': [npz_filename, npz_filename2],
    }
    assert net.eval('worker_queue/size') == 0
    net.eval('queue/enqueue', feed_dict)

    def assert_enqueue():
        assert net.eval('worker_queue/size') == 2

    retry_until(assert_enqueue, sec=1)

# TODO(daeyun): enqueue should fail if batch dimension is inconsistent.
# TODO(daeyun): dequeue should be atomic
# TODO(daeyun): net.eval should support operations.
