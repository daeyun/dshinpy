"""
Utility functions and classes for TensorFlow neural nets.
"""
from distutils import version

import tensorflow as tf

assert version.LooseVersion('0.9.0') <= version.LooseVersion(tf.__version__)

from dshin.nn import types
from dshin.nn import ops
from dshin.nn import graph_utils
from dshin.nn import model_utils
from dshin.nn import utils
from dshin.nn import distributed
from dshin.nn import io

from dshin.nn.model_utils import QueuePlaceholder
from dshin.nn.model_utils import NNModel

__all__ = [
    'graph_utils',
    'ops',
    'io',
    'types',
    'model_utils',
    'utils',
    'distributed',
    'QueuePlaceholder',
    'NNModel',
]
