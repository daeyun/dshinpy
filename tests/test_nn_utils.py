from os import path

import numpy as np
import pytest
import tensorflow as tf
from py._path import local

from dshin import nn

pytestmark = pytest.mark.skipif(True, reason='Temporarily disabled')


class BNOnly(nn.model_utils.NNModel):
    def _model(self):
        out = nn.ops.batch_norm(self.placeholder('input'), is_trainable=True, is_local=True)
        loss = tf.reduce_mean((out - self.placeholder('target')) ** 2, name='loss')
        tf.scalar_summary('loss', loss)
        return loss

    def _placeholders(self):
        return [
            tf.placeholder(tf.float32, shape=[None, 5, 5, 1], name='input'),
            tf.placeholder(tf.float32, shape=[None, 5, 5, 1], name='target'),
        ]


def bn_net_factory(tmpdir: local.LocalPath):
    dirname = str(tmpdir)
    net = BNOnly(log_dir=dirname)
    config = nn.model_utils.default_sess_config(log_device_placement=False, mem=0.05)
    cluster_spec = nn.model_utils.get_local_cluster_spec({'worker': 1})
    server = tf.train.Server(cluster_spec, job_name='worker', task_index=0, config=config)
    net.build(server=server, seed=42, local_queue_size=150, source_queue_names=['train'], source_queue_sizes=[1000], save_model_secs=5)
    return net


@pytest.fixture(scope='function')
def bn_net(tmpdir: local.LocalPath):
    return bn_net_factory(str(tmpdir))


@pytest.fixture(scope='module')
def data():
    np.random.seed(42)
    return {
        'input': np.random.randn(2, 5, 5, 1),
        'target': np.random.randn(2, 5, 5, 1),
    }


def test_get_tensor(bn_net: nn.model_utils.NNModel, data):
    net = bn_net

    assert net.placeholder('input').name == 'placeholder/input:0'
    assert net.placeholder('placeholder/input').name == 'placeholder/input:0'
    assert net.placeholder('placeholder/input:0').name == 'placeholder/input:0'
    assert net.placeholder('placeholder/input:0').name == 'placeholder/input:0'
    assert net.operation('placeholder/input').name == 'placeholder/input'
    assert isinstance(net.operation('placeholder/input'), tf.Operation)

    with pytest.raises(Exception):
        assert net.placeholder('inpu')
    with pytest.raises(Exception):
        assert net.placeholder('placeholder/input:')
    with pytest.raises(Exception):
        assert net.placeholder('placeholder/input:1')


def test_train_bn(bn_net: nn.model_utils.NNModel, data):
    net = bn_net
    net.set_learning_rate(0.001)

    feed = {
        'input': data['input'],
        'target': data['target'],
    }

    loss = net.eval(['loss'], feed)['loss']
    prev_loss, loss = loss, net.eval(['loss'], feed)['loss']
    assert prev_loss == loss

    net.train(feed)
    prev_loss, loss = loss, net.eval(['loss'], feed)['loss']
    assert prev_loss > loss

    prev_loss, loss = loss, net.eval(['loss'], feed)['loss']
    assert prev_loss == loss


def test_validate_save_path(bn_net: nn.model_utils.NNModel, tmpdir: local.LocalPath):
    # Target path should not end with a slash. It is the path to the checkpoint file.
    with pytest.raises(AssertionError):
        bn_net.save('/tmp/some_path/dirname/')

    # Target path should not exist as a directory.
    with pytest.raises(AssertionError):
        bn_net.save(str(tmpdir))


def test_save_and_restore_no_summary(bn_net: nn.model_utils.NNModel, tmpdir: local.LocalPath, data):
    net = bn_net

    def save(p=None):
        net.save(p)

    def restore(net=bn_net, p=None):
        net.restore(p)

    def train():
        net.train({'input': data['input'], 'target': data['target']})

    def loss():
        return net.eval(['loss'], {'input': data['input'], 'target': data['target']})['loss']

    net.set_learning_rate(0.0015)

    save()
    saved_loss = loss()

    train()
    after_train = loss()

    restore()
    after_restore = loss()

    train()
    after_restore_and_train = loss()

    assert saved_loss == after_restore
    assert after_train == after_restore_and_train
    assert after_restore > after_restore_and_train

    assert np.isclose(net.learning_rate(), 0.0015)
    net.set_learning_rate(0.0001)
    assert np.isclose(net.learning_rate(), 0.0001)
    restore()
    assert np.isclose(net.learning_rate(), 0.0015)

    net = bn_net_factory(net.log_dir)
    # already restored.
    assert np.isclose(net.learning_rate(), 0.0015)
    restore(net)
    assert np.isclose(net.learning_rate(), 0.0015)


def test_save_and_restore_with_summary(bn_net: nn.model_utils.NNModel, tmpdir: local.LocalPath, data):
    def save(bn_net, p=None):
        bn_net.save(p)

    def restore(logdir):
        restored = bn_net_factory(logdir)
        restored.restore()
        return restored

    def train(bn_net):
        bn_net.train({'input': data['input'], 'target': data['target']},
                     summary_keys=['scalar'])

    def loss(bn_net, summary_writer_name=None):
        return bn_net.eval(['loss'], {'input': data['input'], 'target': data['target']},
                           summary_writer_name=summary_writer_name, summary_keys=['scalar', 'image'])['loss']

    assert path.isdir(path.join(bn_net.log_dir, 'summary'))
    assert path.isdir(path.join(bn_net.log_dir, 'summary', 'train'))
    assert not path.isdir(path.join(bn_net.log_dir, 'summary', 'eval'))

    save(bn_net)
    train(bn_net)
    bn_net = restore(logdir=bn_net.log_dir)
    loss(bn_net, 'eval')

    # Assume summary files are flushed.
    assert not path.isdir(path.join(bn_net.log_dir, 'summary', 'eval'))

    train(bn_net)

    train(bn_net)
    loss(bn_net)


def test_save_and_restore_global_step(bn_net: nn.model_utils.NNModel, tmpdir: local.LocalPath, data):
    def save(bn_net):
        bn_net.save()

    def restore(logdir):
        restored = bn_net_factory(logdir)
        restored.restore()
        return restored

    def train(bn_net):
        bn_net.train({'input': data['input'], 'target': data['target']})

    def loss(bn_net):
        return bn_net.eval(['loss'], {'input': data['input'], 'target': data['target']})['loss']

    assert bn_net.eval('global_step') == 0

    train(bn_net)
    assert bn_net.eval('global_step') == 1

    save(bn_net)
    train(bn_net)
    assert bn_net.eval('global_step') == 2

    bn_net = restore(bn_net.log_dir)
    assert bn_net.eval('global_step') == 1

    train(bn_net)
    assert bn_net.eval('global_step') == 2

    loss(bn_net)
    assert bn_net.eval('global_step') == 2


def test_save_graph_summary_on_error(bn_net: nn.model_utils.NNModel, tmpdir: local.LocalPath, data):
    net = bn_net
    assert not path.isfile(path.join(net.log_dir, 'graph_summary.txt'))
    with pytest.raises(Exception):
        net.tensor('non_existant_key_392')
    assert path.isfile(path.join(net.log_dir, 'graph_summary.txt'))
