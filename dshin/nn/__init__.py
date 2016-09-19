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

__all__ = [
    'graph_utils.py',
    'ops',
    'types',
    'model_utils',
    'utils',
]
