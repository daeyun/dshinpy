import numpy as np
import tensorflow as tf
import pytest
from dshin import nn


class BNOnly(nn.utils.NNModel):
    def _model(self):
        self._out = nn.ops.batch_norm(self['input'], is_trainable=True, is_local=True)

    def _minimize_op(self):
        loss = tf.reduce_mean((self._out - self['target']) ** 2, name='loss')
        train_op = tf.train.AdamOptimizer(self['learning_rate']).minimize(loss)
        return train_op

    def _placeholders(self):
        return [
            tf.placeholder(tf.float32, shape=[None, 5, 5, 1], name='input'),
            tf.placeholder(tf.float32, shape=[None, 5, 5, 1], name='target'),
        ]


@pytest.fixture(scope='module')
def sample_net():
    return BNOnly(seed=42)


def test_model(sample_net: nn.utils.NNModel):
    np.random.seed(42)
    data = np.random.randn(2, 5, 5, 1)
    target = np.random.randn(2, 5, 5, 1)

    net = sample_net
    train_feed_dict = {
        'input': data,
        'target': target,
        'learning_rate': 0.001,
    }

    eval_feed_dict = train_feed_dict.copy()
    del eval_feed_dict['learning_rate']

    loss = net.eval(['train/loss'], eval_feed_dict)['train/loss']
    prev_loss, loss = loss, net.eval(['train/loss'], eval_feed_dict)['train/loss']
    assert prev_loss == loss

    net.train(train_feed_dict)
    prev_loss, loss = loss, net.eval(['train/loss'], eval_feed_dict)['train/loss']
    assert prev_loss > loss

    prev_loss, loss = loss, net.eval(['train/loss'], eval_feed_dict)['train/loss']
    assert prev_loss == loss


def test_toposort(sample_net: nn.utils.NNModel):
    values = sample_net.sorted_values(r'.*')
    assert len(values) > 10
    assert sample_net.last(r'.*').name == values[-1].name
