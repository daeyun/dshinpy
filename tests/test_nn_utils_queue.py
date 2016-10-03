import time

import numpy as np
import pytest
import tensorflow as tf
from py._path import local

from dshin import log
from dshin import nn


class BN(nn.model_utils.NNModel):
    def _model(self):
        out = nn.ops.batch_norm(self.placeholder('input'), is_trainable=True, is_local=True)
        loss = tf.reduce_mean((out - self.placeholder('target')) ** 2, name='loss')
        tf.scalar_summary('loss', loss)
        return loss

    def _placeholders(self):
        return [
            tf.placeholder(tf.int32, shape=[None, 6, 6, 2], name='not_queued'),
            nn.model_utils.QueuePlaceholder(tf.float32, shape=[None, 5, 5, 1], name='input', file_type=False),
            nn.model_utils.QueuePlaceholder(tf.float32, shape=[None, 5, 5, 1], name='target', file_type=True),
        ]


class BNWithoutFileInput(nn.model_utils.NNModel):
    def _model(self):
        out = nn.ops.batch_norm(self.placeholder('input'), is_trainable=True, is_local=True)
        loss = tf.reduce_mean((out - self.placeholder('target')) ** 2, name='loss')
        tf.scalar_summary('loss', loss)
        return loss

    def _placeholders(self):
        return [
            tf.placeholder(tf.int32, shape=[None, 6, 6, 2], name='not_queued'),
            tf.placeholder_with_default([1], shape=[None], name='not_queued2'),
            nn.model_utils.QueuePlaceholder(tf.float32, shape=[None, 5, 5, 1], name='input', file_type=False),
            nn.model_utils.QueuePlaceholder(tf.float32, shape=[None, 5, 5, 1], name='target', file_type=False),
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
    net = BN(log_dir=str(tmpdir.join('summary')))
    config = nn.model_utils.default_sess_config(log_device_placement=False, mem=0.05)
    cluster_spec = nn.model_utils.get_local_cluster_spec({'worker': 1})
    server = tf.train.Server(cluster_spec, job_name='worker', task_index=0, config=config)
    net.build(server=server, seed=42, local_queue_size=150, source_queue_names=['train'], source_queue_sizes=[1000], save_model_secs=5)
    return net


@pytest.fixture(scope='function')
def net_no_file(tmpdir: local.LocalPath):
    net = BNWithoutFileInput(log_dir=str(tmpdir.join('summary')))
    config = nn.model_utils.default_sess_config(log_device_placement=False, mem=0.05)
    cluster_spec = nn.model_utils.get_local_cluster_spec({'worker': 1})
    server = tf.train.Server(cluster_spec, job_name='worker', task_index=0, config=config)
    net.build(server=server, seed=42, local_queue_size=150, source_queue_names=['train'], source_queue_sizes=[1000], save_model_secs=5)
    return net


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


def test_placeholder(net: nn.model_utils.NNModel):
    assert net.placeholder('not_queued').dtype == tf.int32
    assert net.placeholder('not_queued').get_shape().as_list() == [None, 6, 6, 2]
    assert net.placeholder('input').get_shape().as_list() == [None, 5, 5, 1]


def test_queue_array_placeholder_properties(net: nn.model_utils.NNModel):
    assert net.placeholder('input').name == 'placeholder/input:0'
    assert net.placeholder('input').get_shape().as_list() == [None, 5, 5, 1]
    assert net.placeholder('input').dtype == tf.float32

    assert net.placeholder('queue/train/input').name == 'placeholder/queue/train/input:0'
    assert net.placeholder('queue/train/input').dtype == tf.float32
    assert net.placeholder('queue/train/input').get_shape().as_list() == [None, 5, 5, 1]


def test_queue_operations(net: nn.model_utils.NNModel):
    assert isinstance(net.tensor('queue/train/size'), tf.Tensor)
    assert isinstance(net.tensor('worker_queue/size'), tf.Tensor)

    assert isinstance(net.operation('queue/train/enqueue'), tf.Operation)

    assert isinstance(net.operation('queue/train/close'), tf.Operation)
    assert isinstance(net.operation('worker_queue/close'), tf.Operation)

    assert isinstance(net.operation('queue/train/size'), tf.Operation)
    assert isinstance(net.operation('worker_queue/size'), tf.Operation)


def test_queue_file_placeholder_properties(net: nn.model_utils.NNModel):
    assert net.placeholder('target').name == 'placeholder/target:0'
    assert net.placeholder('target').get_shape().as_list() == [None, 5, 5, 1]
    assert net.placeholder('target').dtype == tf.float32

    assert net.placeholder('queue/train/target').name == 'placeholder/queue/train/target:0'
    assert net.placeholder('queue/train/target').dtype == tf.string
    assert net.placeholder('queue/train/target').get_shape().as_list() == [None]


def test_queue_placeholder_direct_feeding(net: nn.model_utils.NNModel, data):
    feed_dict = {
        'input': data['input'],
        'target': data['target'],
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


def test_enqueue_array(net_no_file: nn.model_utils.NNModel, data):
    net = net_no_file

    feed_dict = {
        'input': data['input'][:1],
        'target': data['target'][:1],
    }
    assert net.eval('worker_queue/size') == 0
    net.enqueue('train', feed_dict)
    assert net.eval('worker_queue/size') == 0
    net.start_local_queue_runners('train', request_size=2, num_threads=1)

    def assert_enqueue(num):
        assert net.eval('worker_queue/size') == num

    retry_until(assert_enqueue, 5, args=[1])
    net.enqueue('train', feed_dict)
    retry_until(assert_enqueue, 5, args=[2])
    net.shutdown()


def test_dequeue_array_eval(net_no_file: nn.model_utils.NNModel, data):
    net = net_no_file

    feed_dict = {
        'input': data['input'][[0, 0]],
        'target': data['target'][[0, 0]],
    }
    assert net.eval('worker_queue/size') == 0
    net.enqueue('train', feed_dict)
    assert net.eval('worker_queue/size') == 0

    net.start_local_queue_runners('train', request_size=2, num_threads=1)

    def assert_enqueue(num):
        assert net.eval('worker_queue/size') == num

    retry_until(assert_enqueue, 5, args=[2])

    loss = net.eval('loss', {'batch_size': 1})
    assert net.eval('worker_queue/size') == 1
    assert net.eval('loss', {'batch_size': 1}) == loss
    assert net.eval('worker_queue/size') == 0


def test_dequeue_array_train(net_no_file: nn.model_utils.NNModel, data):
    net = net_no_file
    queue_feed_dict = {
        'input': data['input'][[0] * 4],
        'target': data['target'][[0] * 4],
    }
    eval_feed_dict = {
        'input': data['input'][[0, 0]],
        'target': data['target'][[0, 0]],
    }
    train_feed_dict = {
        'batch_size': 2,
    }
    net.start_local_queue_runners('train', request_size=8, num_threads=1)

    def assert_enqueue(num):
        assert net.eval('worker_queue/size') == num

    assert net.eval('worker_queue/size') == 0
    net.enqueue('train', queue_feed_dict)

    retry_until(assert_enqueue, 5, args=[4])

    loss1 = net.eval('loss', eval_feed_dict)
    assert net.eval('worker_queue/size') == 4

    net.train(train_feed_dict)
    retry_until(assert_enqueue, 5, args=[2])

    loss2 = net.eval('loss', eval_feed_dict)
    assert loss2 < loss1

    assert net.eval('worker_queue/size') == 2

    net.enqueue('train', queue_feed_dict)
    retry_until(assert_enqueue, 5, args=[6])

    train_feed_dict['batch_size'] = 6
    net.train(train_feed_dict)
    assert net.eval('worker_queue/size') == 0
    loss3 = net.eval('loss', eval_feed_dict)
    assert net.eval('worker_queue/size') == 0
    assert loss3 < loss2


def test_enqueue_array_and_filename(net: nn.model_utils.NNModel, data, filenames):
    net = net
    feed_dict = {
        'input': data['input'][:2],
        'target': filenames[:2],
    }
    assert net.eval('worker_queue/size') == 0
    net.enqueue('train', feed_dict)

    def assert_enqueue():
        assert net.eval('worker_queue/size') == 2

    net.start_local_queue_runners('train', 2)
    retry_until(assert_enqueue, sec=2)
    net.join_local_queue_runner_threads()


def test_dequeue_integrity(net: nn.model_utils.NNModel, data, filenames):
    net.start_local_queue_runners('train', 1000)
    assert net.eval('worker_queue/size') == 0

    np.random.seed(9)

    expected_remaining = 3

    num_examples_pushed = 0
    for i in range(10):
        n = np.random.randint(1, 4)
        num_examples_pushed += n
        indices = [np.random.randint(2) for _ in range(n)]
        feed_dict = {
            'input': data['input'][indices],
            'target': [filenames[j] for j in indices],
        }
        net.enqueue('train', feed_dict)
    log.info('Enqueued %d examples.', num_examples_pushed)

    def assert_enqueue():
        assert net.eval('worker_queue/size') == num_examples_pushed

    retry_until(assert_enqueue, sec=1)

    def dequeue_and_verify(num):
        dequeue = net.eval(['worker_queue/dequeue:0', 'worker_queue/dequeue:1'], {'batch_size': num})

        # NOTE: 'input' corresponds to index 0. This might be unreliable.
        dequeue_input = dequeue['worker_queue/dequeue:0']
        dequeue_target = dequeue['worker_queue/dequeue:1']
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

    net.shutdown()


def test_dequeue_atomic(net: nn.model_utils.NNModel, data, filenames):
    net.start_local_queue_runners('train', 1000)
    assert net.eval('worker_queue/size') == 0

    np.random.seed(9)

    expected_remaining = 9

    num_examples_pushed = 0
    for i in range(10):
        n = np.random.randint(1, 4)
        num_examples_pushed += n
        indices = [np.random.randint(2) for _ in range(n)]
        feed_dict = {
            'input': data['input'][indices],
            'target': [filenames[j] for j in indices],
        }
        net.enqueue('train', feed_dict)
    log.info('Enqueued %d examples.', num_examples_pushed)

    def assert_enqueue():
        assert net.eval('worker_queue/size') == num_examples_pushed

    retry_until(assert_enqueue, sec=1)

    def dequeue_and_verify(num, use_both=True):
        if use_both:
            dequeue = net.eval(['worker_queue/dequeue:0', 'worker_queue/dequeue:1'], {'batch_size': num})
        else:
            dequeue = net.eval(['worker_queue/dequeue:0'], {'batch_size': num})

        # NOTE: 'input' corresponds to index 0. This might be unstable.
        dequeue_input = dequeue['worker_queue/dequeue:0']
        assert dequeue_input.shape[0] == num

        if use_both:
            dequeue_target = dequeue['worker_queue/dequeue:1']
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

    net.shutdown()


def test_dequeue_inconsistent_batch_dimension(net: nn.model_utils.NNModel, data, filenames):
    with pytest.raises(Exception):
        feed_dict = {
            'input': data['input'][[0] * 2],
            'target': [filenames[0]],
        }
        net.enqueue('train', feed_dict)

    with pytest.raises(Exception):
        feed_dict = {
            'input': data['input'][[0]],
            'target': [filenames[0], filenames[0]],
        }
        net.enqueue('train', feed_dict)

    with pytest.raises(Exception):
        feed_dict = {
            'input': data['input'][[0]],
            'target': [filenames[0], filenames[0]],
        }
        net.enqueue('train', feed_dict)


def test_enqueue_incomplete_feed(net: nn.model_utils.NNModel, data):
    with pytest.raises(Exception):
        feed_dict = {
            'input': data['input'][[0] * 2],
        }
        net.enqueue('train', feed_dict)
