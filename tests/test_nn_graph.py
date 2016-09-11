import tensorflow as tf

from dshin import nn


def test_abs_name_scope():
    with nn.graph.abs_name_scope('placeholder'):
        with nn.graph.abs_name_scope('placeholder'):
            p = tf.placeholder(tf.float32, shape=[], name='scalar')
    # Not 'placeholder/placeholder/scalar:0'
    assert p.name == 'placeholder/scalar:0'

    with nn.graph.abs_name_scope('placeholder'):
        p = tf.placeholder(tf.float32, shape=[], name='scalar2')
    # Not 'placeholder_1/scalar:0'
    assert p.name == 'placeholder/scalar2:0'
