"""
TensorFlow type annotation aliases.
"""
import tensorflow as tf
import typecheck as tc

Value = tc.any(tf.Variable, tf.Tensor)
Values = tc.seq_of(Value)
Named = tc.any(tf.Variable, tf.Tensor, tf.Operation)
NamedSeq = tc.seq_of(Named)
Tensors = tc.seq_of(tf.Tensor)
Variables = tc.seq_of(tf.Variable)
Operations = tc.seq_of(tf.Operation)
