import abc

import tensorflow as tf


class Builder(metaclass=abc.ABCMeta):
    def __init__(self, name: str, ch: int, in_size: int = None, out_size: int = None):
        """
        Building block for Tensorflow graphs. in_size and out_size are
        optional, but usually one of them should be defined.

        :param name: Variable scope name.
        :param ch: Number of channels.
        :param in_size: Input resolution is [in_size]*in_dims.
        :param out_size: Output resolution is [out_size]*out_dims.
        """
        self.name = name
        self.ch = ch
        self.in_size = in_size
        self.out_size = out_size

    def build(self, input_value: tf.Tensor) -> tf.Tensor:
        """
        Defines operations and variables in the current graph.
        """
        with tf.variable_scope(self.name):
            return self._build(input_value)

    @abc.abstractmethod
    def _build(self, input_value: tf.Tensor) -> tf.Tensor:
        pass

    @property
    @abc.abstractmethod
    def in_dims(self) -> int:
        """
        :return: Number of input dimensions. 1 for scalars, 2 for images, 3 for voxels.
        """
        pass

    @property
    @abc.abstractmethod
    def out_dims(self) -> int:
        """
        :return: Number of output dimensions. 1 for scalars, 2 for images, 3 for voxels.
        """
        pass
