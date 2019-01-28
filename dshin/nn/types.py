"""
TensorFlow type annotation aliases.
"""
import tensorflow as tf
# import typecheck as tc
import typing

Value = typing.Union[tf.Variable, tf.Tensor]
Values = typing.Sequence[Value]
ValueOrOperation = typing.Union[tf.Variable, tf.Tensor, tf.Operation]
ValuesOrOperations = typing.Sequence[ValueOrOperation]
Tensors = typing.Sequence[tf.Tensor]
Variables = typing.Sequence[tf.Variable]
Operations = typing.Sequence[tf.Operation]


def single_or_seq_of(t):
    return typing.Union[t, typing.Sequence[t]]
