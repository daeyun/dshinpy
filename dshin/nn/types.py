"""
TensorFlow type annotation aliases.
"""
import tensorflow as tf
import typecheck as tc

Value = tc.any(tf.Variable, tf.Tensor)
Values = tc.seq_of(Value)
ValueOrOperation = tc.any(tf.Variable, tf.Tensor, tf.Operation)
ValuesOrOperations = tc.seq_of(ValueOrOperation)
Tensors = tc.seq_of(tf.Tensor)
Variables = tc.seq_of(tf.Variable)
Operations = tc.seq_of(tf.Operation)


def single_or_seq_of(t):
    return tc.any(t, tc.seq_of(t))
