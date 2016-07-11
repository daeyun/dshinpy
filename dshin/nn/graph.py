import abc
import typing

import tensorflow as tf


class Builder(metaclass=abc.ABCMeta):
    def __init__(self, name: str, in_size: int = None, out_size: int = None,
                 in_dims: int = None, out_dims: int = None,
                 is_bn_local=True, is_training: tf.Tensor = None, is_first=False, is_last=False,
                 has_activation=True, has_weight=True, has_bn=True):
        """
        Building blocks for Tensorflow graphs. in_size and out_size are
        optional, but usually one of them should be provided.

        :param name: Variable scope name.
        :param in_size: Input resolution is [in_size]*in_dims.
        :param out_size: Output resolution is [out_size]*out_dims.
        :param is_bn_local: A Tensor of type tf.bool or a constant bool. Not all builders need this.
        :param is_training: A Tensor of type tf.bool. Not all builders need this.
        """
        if is_training is not None:
            assert isinstance(is_training, tf.Tensor) and is_training.dtype == tf.bool
        assert not (is_first and is_last)
        assert has_activation or has_weight
        assert isinstance(in_dims, int) or isinstance(out_dims, int)
        assert isinstance(is_last, bool)
        assert isinstance(is_first, bool)
        assert isinstance(has_activation, bool)
        assert isinstance(has_weight, bool)
        assert isinstance(has_bn, bool)

        self.name = name
        self.in_size = in_size
        self.out_size = out_size
        self.is_bn_local = is_bn_local
        self.is_training = is_training
        self.is_first = is_first
        self.is_last = is_last
        self.has_activation = has_activation
        self.has_weight = has_weight
        self.has_bn = has_bn

    def build(self, input_value: tf.Tensor) -> tf.Tensor:
        """
        Defines operations and variables in the current graph.
        """
        with tf.variable_scope(self.name):
            return self._build(input_value)

    @abc.abstractmethod
    def _build(self, value: typing.Union[tf.Tensor, typing.List[tf.Tensor]]) -> tf.Tensor:
        pass

