import numpy as np
import pytest
import tensorflow as tf
from py._path import local

from dshin import nn


class BNOnly(nn.utils.NNModel):
    def _model(self):
        out = nn.ops.batch_norm(self['input'], is_trainable=True, is_local=True)
        self._loss = tf.reduce_mean((out - self['target']) ** 2, name='loss')

    def _minimize_op(self):
        return tf.train.AdamOptimizer(self['learning_rate']).minimize(self._loss)

    def _placeholders(self):
        return [
            tf.placeholder(tf.float32, shape=[None, 5, 5, 1], name='input'),
            tf.placeholder(tf.float32, shape=[None, 5, 5, 1], name='target'),
        ]


@pytest.fixture(scope='function')
def bn_net(tmpdir: local.LocalPath):
    return BNOnly(seed=42, summary_dir=str(tmpdir.join('summary')))


@pytest.fixture(scope='module')
def data():
    np.random.seed(42)
    return {
        'input': np.random.randn(2, 5, 5, 1),
        'target': np.random.randn(2, 5, 5, 1),
    }


def test_train_bn(bn_net: nn.utils.NNModel, data):
    net = bn_net
    train_feed_dict = {
        'input': data['input'],
        'target': data['target'],
        'learning_rate': 0.001,
    }

    eval_feed_dict = train_feed_dict.copy()
    del eval_feed_dict['learning_rate']

    loss = net.eval(['loss'], eval_feed_dict)['loss']
    prev_loss, loss = loss, net.eval(['loss'], eval_feed_dict)['loss']
    assert prev_loss == loss

    net.train(train_feed_dict)
    prev_loss, loss = loss, net.eval(['loss'], eval_feed_dict)['loss']
    assert prev_loss > loss

    prev_loss, loss = loss, net.eval(['loss'], eval_feed_dict)['loss']
    assert prev_loss == loss


def test_toposort(bn_net: nn.utils.NNModel):
    values = bn_net.sorted_values(r'.*')
    assert len(values) > 10
    assert bn_net.last(r'.*').name == values[-1].name
    assert bn_net.last().name == values[-1].name


def test_validate_save_path(bn_net: nn.utils.NNModel, tmpdir: local.LocalPath):
    # Target path should not end with a slash.
    with pytest.raises(AssertionError):
        bn_net.save('/tmp/some_path/dirname/')

    # Target path should not exist as a directory.
    with pytest.raises(AssertionError):
        bn_net.save(str(tmpdir))


def test_save_and_restore(bn_net: nn.utils.NNModel, tmpdir: local.LocalPath, data):
    out_path = str(tmpdir.join('filename'))

    def save():
        bn_net.save(out_path)

    def restore():
        bn_net.restore(out_path)

    def train():
        bn_net.train({'input': data['input'], 'target': data['target'], 'learning_rate': 0.001})

    def loss():
        return bn_net.eval(['loss'], {'input': data['input'], 'target': data['target']})['loss']

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


def test_save_and_restore_from_scratch(bn_net: nn.utils.NNModel, tmpdir: local.LocalPath, data):
    out_path = str(tmpdir.join('filename'))

    def save(bn_net):
        bn_net.save(out_path)

    def restore():
        return BNOnly.from_file(out_path)

    def train(bn_net):
        bn_net.train({'input': data['input'], 'target': data['target'], 'learning_rate': 0.001})

    def loss(bn_net):
        return bn_net.eval(['loss'], {'input': data['input'], 'target': data['target']})['loss']

    for _ in range(2):
        save(bn_net)
        saved_loss = loss(bn_net)

        train(bn_net)
        after_train = loss(bn_net)

        bn_net = restore()
        after_restore = loss(bn_net)

        train(bn_net)
        after_restore_and_train = loss(bn_net)

        assert saved_loss == after_restore
        assert after_train == after_restore_and_train
        assert after_restore > after_restore_and_train


def test_save_and_restore_with_summary(bn_net: nn.utils.NNModel, tmpdir: local.LocalPath, data):
    out_path = str(tmpdir.join('filename'))

    def save(bn_net):
        bn_net.save(out_path)

    def restore():
        return BNOnly.from_file(out_path, summary_dir=str(tmpdir.join('summary')))

    def train(bn_net):
        bn_net.train({'input': data['input'], 'target': data['target'], 'learning_rate': 0.001})

    def loss(bn_net):
        return bn_net.eval(['loss'], {'input': data['input'], 'target': data['target']})['loss']

    save(bn_net)
    train(bn_net)
    bn_net = restore()
    train(bn_net)
    loss(bn_net)
