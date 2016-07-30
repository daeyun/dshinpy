"""
TensorFlow type annotation aliases.
"""
import typing

import tensorflow as tf

Value = (tf.Variable, tf.Tensor)
Values = typing.Sequence[Value]
Named = (tf.Variable, tf.Tensor, tf.Operation)
NamedSeq = typing.Sequence[Named]
Tensors = typing.Sequence[tf.Tensor]
Variables = typing.Sequence[tf.Variable]
Operations = typing.Sequence[tf.Operation]
