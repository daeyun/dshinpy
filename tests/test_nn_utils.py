from os import path

import numpy as np
import pytest
import tensorflow as tf
from py._path import local

from dshin import nn


class BNOnly(nn.utils.NNModel):
    def _model(self):
        out = nn.ops.batch_norm(self['input'], is_trainable=True, is_local=True)
        self._loss = tf.reduce_mean((out - self['target']) ** 2, name='loss')
        tf.scalar_summary('loss', self._loss, collections=nn.utils.NNModel.summary_keys('SIMPLE'))

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


def test_get_tensor_by_regex(bn_net: nn.utils.NNModel, data):
    net = bn_net

    assert net['input'].name == 'placeholder/input:0'
    assert net['placeholder/input'].name == 'placeholder/input:0'
    assert net['placeholder/input:0'].name == 'placeholder/input:0'
    assert net['placeholder/input:0'].name == 'placeholder/input:0'
    assert net.get('input', prefix='.*', suffix=':0').name == 'placeholder/input:0'
    assert net.get('input', prefix='placeholder/', suffix=':0').name == 'placeholder/input:0'
    assert net.get('input').name == 'placeholder/input:0'
    assert net.get('input', prefix='placeholder/').name == 'placeholder/input:0'
    assert net.get('input', prefix='placeholder/', suffix='$').name == 'placeholder/input'
    assert net.get('input$', prefix='placeholder/').name == 'placeholder/input'
    assert net.get('input$').name == 'placeholder/input'
    assert net['input$'].name == 'placeholder/input'
    assert isinstance(net['input$'], tf.Operation)
    assert net['.*input:0'].name == 'placeholder/input:0'

    with pytest.raises(Exception):
        assert net['inpu']
    with pytest.raises(Exception):
        assert net['nput']
    with pytest.raises(Exception):
        assert net['pla']
    with pytest.raises(Exception):
        assert net['placeholder/inpu']
    with pytest.raises(Exception):
        assert net['laceholder/inpu']
    with pytest.raises(Exception):
        assert net['placeholder/input:']
    with pytest.raises(Exception):
        assert net['placeholder/input:1']
    with pytest.raises(Exception):
        assert net.get('.*input(:0)?', suffix='')


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


def test_save_and_restore_with_summary(bn_net: nn.utils.NNModel, tmpdir: local.LocalPath, data):
    out_path = str(tmpdir.join('filename'))

    def save(bn_net):
        bn_net.save(out_path)

    def restore(summary_dir=str(tmpdir.join('summary'))):
        return BNOnly.from_file(out_path, summary_dir=summary_dir)

    def train(bn_net):
        bn_net.train({'input': data['input'], 'target': data['target'], 'learning_rate': 0.001},
                     summary_modes=['TRAIN_UPDATE_RATIO'])

    def loss(bn_net, summary_writer_name=None):
        return bn_net.eval(['loss'], {'input': data['input'], 'target': data['target']},
                           summary_writer_name=summary_writer_name, summary_modes=['SIMPLE', 'IMAGE'])['loss']

    assert path.isdir(str(tmpdir.join('summary')))
    assert path.isdir(str(tmpdir.join('summary/train')))
    assert not path.isdir(str(tmpdir.join('summary2')))
    save(bn_net)
    train(bn_net)
    bn_net = restore()
    loss(bn_net)
    train(bn_net)

    bn_net = restore(str(tmpdir.join('summary2')))
    assert path.isdir(str(tmpdir.join('summary2')))

    # 'train' summary containing the graph is saved immediately.
    assert path.isdir(str(tmpdir.join('summary2/train')))

    train(bn_net)
    loss(bn_net)

    # 'eval' is the default summary name if is_training is False.
    assert path.isdir(str(tmpdir.join('summary2/eval')))
    assert not path.isdir(str(tmpdir.join('summary2/test_experiment_name')))

    loss(bn_net, summary_writer_name='test_experiment_name')
    assert path.isdir(str(tmpdir.join('summary2/test_experiment_name')))


def test_save_and_restore_global_step(bn_net: nn.utils.NNModel, tmpdir: local.LocalPath, data):
    out_path = str(tmpdir.join('filename'))

    def save(bn_net):
        bn_net.save(out_path)

    def restore():
        return BNOnly.from_file(out_path, summary_dir=str(tmpdir.join('summary')))

    def train(bn_net):
        bn_net.train({'input': data['input'], 'target': data['target'], 'learning_rate': 0.001})

    def loss(bn_net):
        return bn_net.eval(['loss'], {'input': data['input'], 'target': data['target']})['loss']

    assert bn_net.global_step() == 0

    train(bn_net)
    assert bn_net.global_step() == 1

    save(bn_net)
    train(bn_net)
    assert bn_net.global_step() == 2

    bn_net = restore()
    assert bn_net.global_step() == 1

    train(bn_net)
    assert bn_net.global_step() == 2

    loss(bn_net)
    assert bn_net.global_step() == 2


def test_save_and_restore_placeholders(bn_net: nn.utils.NNModel, tmpdir: local.LocalPath, data):
    out_path = str(tmpdir.join('filename'))

    def save(bn_net):
        bn_net.save(out_path)

    def restore():
        return BNOnly.from_file(out_path, summary_dir=str(tmpdir.join('summary')))

    assert isinstance(bn_net['placeholder/input'], tf.Tensor)
    assert isinstance(bn_net['placeholder/target'], tf.Tensor)

    save(bn_net)
    assert isinstance(bn_net['placeholder/input'], tf.Tensor)
    assert isinstance(bn_net['placeholder/target'], tf.Tensor)

    bn_net = restore()
    assert isinstance(bn_net['placeholder/input'], tf.Tensor)
    assert isinstance(bn_net['placeholder/target'], tf.Tensor)
    assert isinstance(bn_net['input'], tf.Tensor)
    assert isinstance(bn_net['target'], tf.Tensor)
