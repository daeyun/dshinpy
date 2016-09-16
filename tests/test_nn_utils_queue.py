import threading
import time
from os import path

from dshin import log
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
            tf.placeholder_with_default([1], shape=[None], name='not_queued2'),
            nn.utils.QueuePlaceholder(tf.float32, shape=[None, 5, 5, 1], name='input', is_file=False),
            nn.utils.QueuePlaceholder(tf.float32, shape=[None, 5, 5, 1], name='target', is_file=False),
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


@pytest.fixture(scope='function')
def net(tmpdir: local.LocalPath):
    return BN(seed=42, summary_dir=str(tmpdir.join('summary')))


@pytest.fixture(scope='function')
def net_no_file(tmpdir: local.LocalPath):
    return BNWithoutFileInput(seed=42, summary_dir=str(tmpdir.join('summary')))


@pytest.fixture(scope='module')
def data():
    np.random.seed(42)
    d = {
        'input': np.ones((2, 5, 5, 1)),
        'target': 2 * np.ones((2, 5, 5, 1)).astype(np.float32),
    }
    d['input'][1] += 1
    d['target'][1] += 1
    return d


@pytest.fixture(scope='module')
def filenames(data, tmpdir_factory):
    tmpdir = tmpdir_factory.mktemp('npz_files')
    filenames = []
    filenames.append(str(tmpdir.join('target0.npz')))
    np.savez_compressed(filenames[0], data=data['target'][[0]].astype(np.float64))
    filenames.append(str(tmpdir.join('target1.npz')))
    np.savez_compressed(filenames[1], data=data['target'][[1]].astype(np.float32))
    return filenames


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


@pytest.mark.timeout(5)
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


@pytest.mark.timeout(5)
def test_init_queue_runners(net: nn.utils.NNModel):
    runners = net.graph.get_collection(tf.GraphKeys.QUEUE_RUNNERS)
    assert len(runners) == 1
    assert isinstance(runners[0], tf.train.QueueRunner)
    assert runners[0].name.startswith(nn.utils.NNModel._consumer_queue_prefix)
    assert len(net.queue_runner_threads) >= net._worker_queue_num_threads

    for thread in net.queue_runner_threads:
        assert isinstance(thread, threading.Thread)
        assert thread.is_alive()
        assert thread.daemon


@pytest.mark.timeout(5)
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


@pytest.mark.timeout(5)
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


@pytest.mark.timeout(5)
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


@pytest.mark.timeout(5)
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


@pytest.mark.timeout(5)
def test_enqueue_array_and_filename(net: nn.utils.NNModel, data, filenames):
    net = net
    feed_dict = {
        'queue/input': data['input'][:2],
        'queue/target': filenames[:2],
    }
    assert net.eval('worker_queue/size') == 0
    net.eval('queue/enqueue', feed_dict)

    def assert_enqueue():
        assert net.eval('worker_queue/size') == 2

    retry_until(assert_enqueue, sec=1)


@pytest.mark.timeout(5)
def test_dequeue_integrity(net: nn.utils.NNModel, data, filenames):
    net = net
    assert net.eval('worker_queue/size') == 0

    np.random.seed(9)

    expected_remaining = 3

    num_examples_pushed = 0
    for i in range(10):
        n = np.random.randint(1, 4)
        num_examples_pushed += n
        indices = [np.random.randint(2) for _ in range(n)]
        feed_dict = {
            'queue/input': data['input'][indices],
            'queue/target': [filenames[j] for j in indices],
        }
        net.eval('queue/enqueue', feed_dict)
    log.info('Enqueued %d examples.', num_examples_pushed)

    def assert_enqueue():
        assert net.eval('worker_queue/size') == num_examples_pushed

    retry_until(assert_enqueue, sec=1)

    def dequeue_and_verify(num):
        dequeue = net.eval(['worker_queue:0', 'worker_queue:1'], {net['batch_size']: num})

        # NOTE: 'input' corresponds to index 0. This might be unstable.
        dequeue_input = dequeue['worker_queue:0']
        dequeue_target = dequeue['worker_queue:1']
        assert dequeue_input.shape == dequeue_target.shape
        assert dequeue_input.shape[0] == num

        for i in range(num):
            is_first = np.allclose(dequeue_input[i], data['input'][0])
            is_second = np.allclose(dequeue_input[i], data['input'][1])
            assert is_first != is_second
            assert np.allclose(dequeue_target[i], data['target'][0 if is_first else 1])

    expected_num_popped = num_examples_pushed - expected_remaining
    dequeue_and_verify(expected_num_popped)
    assert net.eval('worker_queue/size') == expected_remaining

    dequeue_and_verify(expected_remaining)
    assert net.eval('worker_queue/size') == 0


@pytest.mark.timeout(5)
def test_dequeue_atomic(net: nn.utils.NNModel, data, filenames):
    net = net
    assert net.eval('worker_queue/size') == 0

    np.random.seed(9)

    expected_remaining = 9

    num_examples_pushed = 0
    for i in range(10):
        n = np.random.randint(1, 4)
        num_examples_pushed += n
        indices = [np.random.randint(2) for _ in range(n)]
        feed_dict = {
            'queue/input': data['input'][indices],
            'queue/target': [filenames[j] for j in indices],
        }
        net.eval('queue/enqueue', feed_dict)
    log.info('Enqueued %d examples.', num_examples_pushed)

    def assert_enqueue():
        assert net.eval('worker_queue/size') == num_examples_pushed

    retry_until(assert_enqueue, sec=1)

    def dequeue_and_verify(num, use_both=True):
        if use_both:
            dequeue = net.eval(['worker_queue:0', 'worker_queue:1'], {net['batch_size']: num})
        else:
            dequeue = net.eval(['worker_queue:0'], {net['batch_size']: num})

        # NOTE: 'input' corresponds to index 0. This might be unstable.
        dequeue_input = dequeue['worker_queue:0']
        assert dequeue_input.shape[0] == num

        if use_both:
            dequeue_target = dequeue['worker_queue:1']
            assert dequeue_input.shape == dequeue_target.shape

        for i in range(num):
            is_first = np.allclose(dequeue_input[i], data['input'][0])
            is_second = np.allclose(dequeue_input[i], data['input'][1])
            assert is_first != is_second

            if use_both:
                assert np.allclose(dequeue_target[i], data['target'][0 if is_first else 1])

    expected_num_popped = num_examples_pushed - expected_remaining
    dequeue_and_verify(expected_num_popped, use_both=False)
    assert net.eval('worker_queue/size') == expected_remaining

    dequeue_and_verify(expected_remaining, use_both=True)
    assert net.eval('worker_queue/size') == 0


@pytest.mark.timeout(5)
def test_dequeue_inconsistent_batch_dimension(net: nn.utils.NNModel, data, filenames):
    with pytest.raises(Exception):
        feed_dict = {
            'queue/input': data['input'][[0] * 2],
            'queue/target': [filenames[0]],
        }
        net.eval('queue/enqueue', feed_dict)

    with pytest.raises(Exception):
        feed_dict = {
            'queue/input': data['input'][[0]],
            'queue/target': [filenames[0], filenames[0]],
        }
        net.eval('queue/enqueue', feed_dict)

    with pytest.raises(Exception):
        feed_dict = {
            'queue/input': data['input'][[0]],
            'queue/target': [filenames[0], filenames[0]],
        }
        net.eval('queue/enqueue', feed_dict)


@pytest.mark.timeout(5)
def test_enqueue_incomplete_feed(net: nn.utils.NNModel, data):
    with pytest.raises(Exception):
        feed_dict = {
            'queue/input': data['input'][[0] * 2],
        }
        net.eval('queue/enqueue', feed_dict)


def test_optional_placeholder_queue_coverage(net: nn.utils.NNModel, data, filenames):
    def dequeue_and_trace(num, feed):
        new_feed = feed.copy()
        new_feed.update({'batch_size': num})
        dequeue = net.eval(['input', 'target'], new_feed,
                           check_optional_placeholder_coverage=False)
        inds = []
        for i in range(num):
            ind = []
            is_first = np.allclose(dequeue['input'][i], data['input'][0])
            is_second = np.allclose(dequeue['input'][i], data['input'][1])
            assert is_first != is_second
            ind.append(0 if is_first else 1)

            is_first = np.allclose(dequeue['target'][i], data['target'][0])
            is_second = np.allclose(dequeue['target'][i], data['target'][1])
            assert is_first != is_second
            ind.append(0 if is_first else 1)
            inds.append(ind)

        return np.array(inds)

    def assert_enqueue(size):
        assert net.eval('worker_queue/size') == size

    assert net.eval('queue/size') == 0
    assert net.eval('worker_queue/size') == 0
    loss1 = net.eval('loss', {
        'input': data['input'],
        'target': data['target'],
    })
    assert net.eval('queue/size') == 0
    assert net.eval('worker_queue/size') == 0

    net.enqueue({
        'input': [data['input'][1]],
        'target': [filenames[1]],
    })
    retry_until(assert_enqueue, sec=1, args=[1])

    inds = dequeue_and_trace(1, {'input': [data['input'][0]]})
    assert net.eval('worker_queue/size') == 0
    assert (inds.ravel() == [0, 1]).all()

    for i in range(9):
        net.enqueue({
            'input': [data['input'][0]],
            'target': [filenames[0]],
        })
    retry_until(assert_enqueue, sec=1, args=[9])

    inds = dequeue_and_trace(2, {'target': data['target'][[0, 1]]})
    assert net.eval('worker_queue/size') == 7
    assert (inds == [[0, 0], [0, 1]]).all()

    inds = dequeue_and_trace(2, {})
    assert net.eval('worker_queue/size') == 5
    assert (inds == [[0, 0], [0, 0]]).all()

    loss2 = net.eval('loss', {'batch_size': 1})
    assert net.eval('worker_queue/size') == 4
    assert loss1 != loss2

    loss3 = net.eval('loss', {'batch_size': 2})
    assert net.eval('worker_queue/size') == 2
    assert loss2 == loss3

    loss4 = net.eval('loss', {
        'target': data['target'][[1, 1]],
        'batch_size': 2
    }, check_optional_placeholder_coverage=False)

    assert net.eval('worker_queue/size') == 0
    assert loss1 != loss4
    assert loss2 != loss4
    assert loss3 != loss4

    with pytest.raises(ValueError):
        net.eval('loss', {
            'target': filenames,
        })

    with pytest.raises(ValueError):
        net.eval('loss', {
            'input': data['input'],
        })

    with pytest.raises(ValueError):
        net.eval('loss', {
            'input': data['input'],
        })
