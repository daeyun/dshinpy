"""
Helper functions for building TensorFlow neural net layers.
"""
import functools
import math

import numpy as np
import tflearn
import tensorflow as tf
import typecheck as tc

from dshin.nn import types as nn_types
from dshin.third_party import gflags
from tensorflow.contrib import slim
from tensorflow.contrib.slim.nets import resnet_v2
from tensorflow.contrib.slim.nets import resnet_utils


def layer(func):
    @functools.wraps(func)
    def wrapper(*args, **kwds):
        output = func(*args, **kwds)
        tf.get_default_graph().add_to_collection(tf.GraphKeys.ACTIVATIONS, output)
        return output

    return wrapper


@tc.typecheck
def _get_model_variable(name: str, shape: tc.seq_of(int), initializer: callable, trainable: bool = True) -> tf.Variable:
    # with tf.device('/cpu:0'):
    # dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    dtype = tf.float32
    var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype, trainable=trainable)
    tf.add_to_collection(tf.GraphKeys.MODEL_VARIABLES, var)
    return var


def conv_same_pad(inputs: tf.Tensor, kernel_size=3, rate=1, name='same_pad'):
    # See tensorflow.contrib.slim.nets.resnet_utils.conv2d_same
    dims = inputs.get_shape().ndims - 2
    assert dims in (2, 3)

    with tf.name_scope(name) as scope:
        kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
        pad_total = kernel_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        inputs = tf.pad(inputs, [[0, 0]] + [[pad_beg, pad_end]] * dims + [[0, 0]], name=scope)
    return inputs


def conv2d(input_tensor: tf.Tensor,
           n_out: int,
           k: int = 5,
           s: int = 1,
           name: str = "conv2d",
           padding: str = 'SAME',
           use_bias: bool = False) -> tf.Tensor:
    assert input_tensor.get_shape().ndims == 4

    with tf.variable_scope(name):
        n_in = input_tensor.get_shape().as_list()[-1]
        stddev = math.sqrt(2.0 / ((k ** 2) * n_in))
        w = _get_model_variable('w', [k, k, n_in, n_out], initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input_tensor, w, strides=[1, s, s, 1], padding=padding, use_cudnn_on_gpu=True)

        if not use_bias:
            return conv

        b = _get_model_variable('b', [n_out], initializer=tf.constant_initializer(0))
        return tf.nn.bias_add(conv, b)


def upsample_filt(size):
    """
    Make a 2D bilinear kernel suitable for upsampling of the given (h, w) size.
    Source: https://github.com/shelhamer/fcn.berkeleyvision.org/blob/master/surgery.py#L35
    """
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    return (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)


def deconv2d(input_tensor: tf.Tensor,
             n_out: int,
             k: int = 5,
             s: int = 1,
             name: str = 'deconv2d',
             padding: str = 'SAME',
             use_bias: bool = False,
             bilinear_init=False) -> tf.Tensor:
    assert input_tensor.get_shape().ndims == 4

    with tf.variable_scope(name):
        input_shape = input_tensor.get_shape().as_list()
        n_in = input_shape[-1]
        stddev = math.sqrt(2.0 / ((k ** 2) / (s ** 2) * n_in))

        if bilinear_init:
            w_value = np.tile(upsample_filt(k).reshape(k, k, 1, 1), [1, 1, n_out, n_in])
            w = _get_model_variable('w', [k, k, n_out, n_in], initializer=tf.constant_initializer(w_value, dtype=tf.float32))
        else:
            w = _get_model_variable('w', [k, k, n_out, n_in], initializer=tf.random_normal_initializer(stddev=stddev))

        if type(input_shape[0]) == int:
            batchsize = input_shape[0]
        else:
            batchsize = tf.shape(input_tensor)[0]

        output_shape = [batchsize, s * input_shape[1], s * input_shape[2], n_out]

        deconv = tf.nn.conv2d_transpose(input_tensor, w, strides=[1, s, s, 1], padding=padding, output_shape=output_shape)  # type: tf.Tensor

        if type(output_shape[0]) != int:
            output_shape[0] = None

        deconv.set_shape(output_shape)

        if not use_bias:
            return deconv

        b = _get_model_variable('b', [n_out], initializer=tf.constant_initializer(0))
        return tf.nn.bias_add(deconv, b)


def conv3d(input_tensor: tf.Tensor,
           n_out: int,
           k: int = 5,
           s: int = 1,
           name: str = "conv3d",
           padding: str = 'SAME',
           use_bias: bool = False) -> tf.Tensor:
    assert input_tensor.get_shape().ndims == 5

    with tf.variable_scope(name):
        n_in = input_tensor.get_shape().as_list()[-1]
        stddev = math.sqrt(2.0 / ((k ** 3) * n_in))
        w = _get_model_variable('w', [k, k, k, n_in, n_out], initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv3d(input_tensor, w, strides=[1, s, s, s, 1], padding=padding)

        if not use_bias:
            return conv

        b = _get_model_variable('b', [n_out], initializer=tf.constant_initializer(0))
        return tf.nn.bias_add(conv, b)


def deconv3d(input_tensor: tf.Tensor,
             n_out: int,
             k: int = 5,
             s: int = 1,
             name: str = 'deconv3d',
             padding: str = 'SAME',
             use_bias: bool = False) -> tf.Tensor:
    assert input_tensor.get_shape().ndims == 5

    with tf.variable_scope(name):
        input_shape = input_tensor.get_shape().as_list()
        n_in = input_shape[-1]
        # stddev = math.sqrt(2.0 / ((k ** 3) / (s ** 3) * n_in))
        # w = _get_model_variable('w', [k, k, k, n_out, n_in], initializer=tf.random_normal_initializer(stddev=stddev))
        w = _get_model_variable('w', [k, k, k, n_out, n_in], initializer=slim.initializers.xavier_initializer())

        if type(input_shape[0]) == int:
            batchsize = input_shape[0]
        else:
            batchsize = tf.shape(input_tensor)[0]

        output_shape = [batchsize, s * input_shape[1], s * input_shape[2], s * input_shape[3], n_out]
        deconv = tf.nn.conv3d_transpose(value=input_tensor, filter=w, output_shape=output_shape, strides=[1, s, s, s, 1], padding=padding)  # type: tf.Tensor

        if type(output_shape[0]) != int:
            output_shape[0] = None

        deconv.set_shape(output_shape)

        if not use_bias:
            return deconv

        b = _get_model_variable('b', [n_out], initializer=tf.constant_initializer(0))
        return tf.nn.bias_add(deconv, b)


def linear(input_tensor: tf.Tensor,
           n_out: int,
           name: str = 'linear',
           use_bias: bool = False) -> tf.Tensor:
    assert input_tensor.get_shape().ndims == 2
    assert isinstance(n_out, int)

    with tf.variable_scope(name):
        n_in = input_tensor.get_shape().as_list()[-1]
        stddev = math.sqrt(2.0 / n_in)
        w = _get_model_variable('w', [n_in, n_out], initializer=tf.random_normal_initializer(stddev=stddev))
        lin = tf.matmul(input_tensor, w)

        if not use_bias:
            return lin

        b = _get_model_variable('b', [n_out], initializer=tf.constant_initializer(0))
        return tf.nn.bias_add(lin, b)


@layer
@tc.typecheck
def fixed_unpool(value: nn_types.Value, name: str = 'unpool', mode: str = 'ZERO_FILLED') -> tf.Tensor:
    """
    Upsampling operation.

    :param value: A Tensor of shape [b, d0, d1, ..., dn, ch]
    :param name: A name for the operation.
    :param mode:
        ZERO_FILLED: N-dimensional version of the unpooling operation from Dosovitskiy et al.
        NEAREST: Fill with the same values.
    :return: A Tensor of shape [b, 2*d0, 2*d1, ..., 2*dn, ch]
    """
    assert mode in ['ZERO_FILLED', 'NEAREST']

    with tf.name_scope(name) as scope:
        value = tf.convert_to_tensor(value)
        sh = value.get_shape().as_list()
        dim = len(sh[1:-1])
        out = (tf.reshape(value, [-1] + sh[-dim:]))
        for i in range(dim, 0, -1):
            pad_values = {
                'ZERO_FILLED': tf.zeros_like(out),
                'NEAREST': out,
            }[mode]
            out = tf.concat(i, [out, pad_values])
        out_size = [-1] + [s * 2 for s in sh[1:-1]] + [sh[-1]]
        out = tf.reshape(out, out_size, name=scope)
    return out


@layer
@tc.typecheck
def fixed_pool(value: nn_types.Value, name: str = 'pool') -> nn_types.Value:
    """
    Downsampling operation.

    :param value: A Tensor of shape [b, d0, d1, ..., dn, ch]
    :param name: A name for the operation.
    :return: A Tensor of shape [b, d0/2, d1/2, ..., dn/2, ch]
    """
    with tf.name_scope(name) as scope:
        value = tf.convert_to_tensor(value)
        sh = value.get_shape().as_list()
        out = value
        for sh_i in sh[1:-1]:
            assert sh_i % 2 == 0
        for i in range(len(sh[1:-1])):
            out = tf.reshape(out, (-1, 2, np.prod(sh[i + 2:])))
            out = out[:, 0, :]
        out_size = [-1] + [math.ceil(s / 2) for s in sh[1:-1]] + [sh[-1]]
        out = tf.reshape(out, out_size, name=scope)
    return out


def lrelu(input_tensor: nn_types.Value, alpha: float = 0.05, name: str = 'lrelu') -> tf.Tensor:
    """
    Leaky ReLU.
    """
    return tflearn.leaky_relu(input_tensor, alpha=alpha, name=name)


@tc.typecheck
def ema_with_initial_value(value: nn_types.Value,
                           initial_value: float = 0.0,
                           ema_trainer: tc.optional(tf.train.ExponentialMovingAverage) = None,
                           decay: tc.optional(float) = None,
                           name: str = 'ema') -> nn_types.Value:
    if ema_trainer is None:
        if decay is None:
            decay = 0.99
        ema_trainer = tf.train.ExponentialMovingAverage(decay=decay)
    else:
        assert decay is None

    with tf.variable_scope(name):
        if float(initial_value) == 0.0 and not isinstance(value, tf.Variable):
            apply_op = ema_trainer.apply([value])
            moving_average = ema_trainer.average(value)
        else:
            shape = value.get_shape().as_list()
            init = tf.constant_initializer(initial_value, dtype=value.dtype)
            value_copy = _get_model_variable('value_copy', shape=shape, initializer=init, trainable=False)
            with tf.control_dependencies([tf.assign(value_copy, value)]):
                apply_op = ema_trainer.apply([value_copy])
                moving_average = ema_trainer.average(value_copy)

        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, apply_op)

        return moving_average


def ema(value, decay, name='EMA'):
    # Reset variable scope.
    with tf.variable_scope('EMA/') as scope:
        # TODO(daeyun): Reuse the trainer object so that there is only one apply op per group.
        # This currently creates a bunch of operations named `ema_#` in the root name scope. variable names are not affected.
        ema_trainer = tf.train.ExponentialMovingAverage(decay=decay, name=name)
        apply_op = ema_trainer.apply([value])
        moving_average = ema_trainer.average(value)
    tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, apply_op)
    return moving_average


@tc.typecheck
def ema_with_update_dependencies(values: nn_types.Value, initial_values: float = 0.0, decay: float = 0.99, name: str = 'ema') -> tf.Tensor:
    with tf.variable_scope(name):
        ema_trainer = tf.train.ExponentialMovingAverage(decay=decay)
        assign_ops = []
        ema_variables = []

        if isinstance(initial_values, list):
            assert len(initial_values) == len(values)
        else:
            initial_values = [float(initial_values) for _ in range(len(values))]

        for value, initial_value in zip(values, initial_values):
            shape = value.get_shape().as_list()
            init = tf.constant_initializer(initial_value)
            value_copy = _get_model_variable(value.name + '/ema', shape=shape, initializer=init, trainable=False)
            assign_ops.append(tf.assign(value_copy, value))
            ema_variables.append(value_copy)

        # Evaluating these values will update the shadow variables.
        identity_vars_with_update = []
        shadow_vars_with_update = []

        # Evaluating these values will not update anything.
        shadow_vars_without_update = []

        with tf.control_dependencies(assign_ops):
            with tf.control_dependencies(ema_trainer.apply(ema_variables)):
                for ema_variable in ema_variables:
                    identity_vars_with_update.append(tf.group(ema_variable))
                    shadow_vars_with_update.append(ema_trainer.average(ema_variable))
            for ema_variable in ema_variables:
                shadow_vars_without_update.append(ema_trainer.average(ema_variable))

        return identity_vars_with_update, shadow_vars_with_update, shadow_vars_without_update


@layer
@tc.typecheck
def batch_norm(value: nn_types.Value,
               is_local: tc.any(tf.Tensor, bool) = True,
               name: str = 'bn',
               offset: float = 0.0,
               scale: float = 1.0,
               ema_decay: float = 0.99,
               return_mean_var: bool = False,
               is_trainable: bool = True) -> tf.Tensor:
    """

    For updating the EMA variables, use the following idiom from TensorFlow documentation:

    ::
        # Create an op that will update the moving averages after each training step.
        with tf.variable_scope('train'):
            minimize = optimizer.minimize(loss, name='minimize')
            with tf.control_dependencies([minimize_op]):
                train_step = tf.group(*update_ops, name='step')

    :param is_local: A Tensor of type tf.bool or a boolean constant. Indicates whether the output tensor should use
    local stats or moving averages. This does not affect updating the shadow variables. True for training, false for testing.
    """
    assert type(is_trainable) == bool
    assert (isinstance(is_local, tf.Tensor) and is_local.dtype == tf.bool) or isinstance(is_local, bool)

    shape = value.get_shape().as_list()
    rank = len(shape)
    axes = {
        2: [0],
        4: [0, 1, 2],
        5: [0, 1, 2, 3],
    }[rank]

    n_out = shape[-1]
    with tf.variable_scope(name) as scope:
        beta = _get_model_variable('offset', shape=[n_out], initializer=tf.constant_initializer(offset), trainable=is_trainable)
        gamma = _get_model_variable('scale', shape=[n_out], initializer=tf.constant_initializer(scale), trainable=is_trainable)

        batch_mean, batch_var = tf.nn.moments(value, axes, name=scope.original_name_scope)
        batch_var = tf.nn.relu(batch_var, name='var')  # Force positive value. Renaming `variance` to `var`.

        ema_batch_mean = ema(batch_mean, ema_decay)
        ema_batch_var = ema(batch_var, ema_decay)

        if isinstance(is_local, tf.Tensor) and is_local.dtype == tf.bool:
            # All 4 variables are always evaluated regardless of what `is_local` is. Updates are not handled here.
            # Separate update ops should have been added to `tf.GraphKeys.UPDATE_OPS` in `ema()`.
            mean, var = tf.cond(is_local, lambda: (batch_mean, batch_var), lambda: (ema_batch_mean, ema_batch_var))
        else:
            assert isinstance(is_local, bool)
            mean, var = (batch_mean, batch_var) if is_local else (ema_batch_mean, ema_batch_var)

        bn = tf.nn.batch_normalization(value, mean, var, beta, gamma, 1e-5, name='batchnorm')

    if return_mean_var:
        return bn, mean, var

    return bn


@tc.typecheck
def flatten(value: nn_types.Value, name: str = 'flatten') -> tf.Tensor:
    """
    Flattens the input tensor's shape to be linear.

    :param value: A Tensor of shape [b, d0, d1, ...]
    :param name: A name for the operation.
    :return: A Tensor of shape [b, d0*d1*...]
    """
    with tf.name_scope(name) as scope:
        num_linear = np.prod(value.get_shape().as_list()[1:])
        return tf.reshape(value, [-1, num_linear], name=scope)


@tc.typecheck
def conv_reshape(value: nn_types.Value,
                 k: int,
                 num_channels: tc.optional(int) = None,
                 name: str = 'conv_reshape',
                 dims: int = 2) -> tf.Tensor:
    assert 2 <= dims <= 3

    shape = value.get_shape().as_list()
    assert len(shape) == 2

    if num_channels is None:
        num_channels = float(shape[1]) / (k ** dims)
        assert num_channels.is_integer()
        num_channels = int(num_channels)

    out_shape = [-1] + ([k] * dims) + [num_channels]

    assert np.prod(out_shape[1:]) == np.prod(shape[1:])

    return tf.reshape(value, out_shape, name=name)


@tc.typecheck
def stack_batch_dim(value_list: nn_types.Values,
                    name: str = 'batch_stack') -> tf.Tensor:
    """
    Concatenates Tensors in the batch dimension.
    """
    assert isinstance(value_list, list)
    sh = value_list[0].get_shape().as_list()
    for xi in value_list:
        sh_i = xi.get_shape().as_list()
        assert len(sh_i) == 2
        assert sh == sh_i

    with tf.name_scope(name) as scope:
        concat = tf.concat(1, [tf.expand_dims(xi, 1) for xi in value_list])
        out = tf.reshape(concat, [-1, sh[-1]], name=scope)

    return out


@tc.typecheck
def apply_concat(value_list: nn_types.Values, factory: callable, name_prefix: str = 'branch') -> tf.Tensor:
    """
    Concatenates Tensors in the channel dimension.
    """
    assert isinstance(value_list, list)
    assert callable(factory)
    branches = []
    for i, value in enumerate(value_list):
        assert isinstance(value, tf.Tensor)
        with tf.variable_scope('{}{}'.format(name_prefix, i)):
            out = factory(i, value)
        assert isinstance(out, tf.Tensor)
        branches.append(out)
    last_dim = branches[0].get_shape().ndims - 1
    out = tf.concat(last_dim, branches, name='{}_concat'.format(name_prefix))
    return out


def residual_unit(inputs: tf.Tensor,
                  is_training,
                  n_out=None,
                  name: str = 'rn',
                  mode: str = 'SAME',
                  is_first: bool = False) -> tf.Tensor:
    """
    Uses identity shortcut if `n_out` is None.
    """
    assert mode in ['DOWNSAMPLE', 'UPSAMPLE', 'SAME']

    shape = inputs.get_shape().as_list()
    dims = len(shape) - 2

    s = 1 if mode == 'SAME' else 2

    conv = {
        2: deconv2d if mode == 'UPSAMPLE' else conv2d,
        3: deconv3d if mode == 'UPSAMPLE' else conv3d,
    }[dims]

    nin = shape[-1]
    if n_out is None:
        n_out = nin

    with tf.variable_scope(name):
        def shortcut(x):
            split_out = x
            if mode == 'UPSAMPLE':
                # split_out = fixed_unpool(split_out, mode='NEAREST', name='unpool')
                h, w = split_out.get_shape().as_list()[1:3]
                split_out = tf.image.resize_nearest_neighbor(split_out, [h * 2, w * 2], name='upsample')
            elif mode == 'DOWNSAMPLE':
                split_out = fixed_pool(split_out, name='pool')
            if n_out > nin:
                padding = [[0, 0] for _ in range(len(shape))]
                padding[-1] = [int(math.floor((n_out - nin) / 2.0)), int(math.ceil((n_out - nin) / 2.0))]
                assert sum(padding[-1]) == n_out - nin
                split_out = tf.pad(split_out, padding, mode='CONSTANT', name='pad_shortcut')
            elif n_out < nin:
                split_out = conv(split_out, n_out, k=1, s=1, name='conv_shortcut')
            return split_out

        if n_out != nin:
            is_first = True

        if not is_first:
            h = shortcut(inputs)

        inputs = slim.batch_norm(inputs, activation_fn=None, decay=0.997, epsilon=1e-5, scale=True, is_training=is_training, scope='bn1')
        # inputs = lrelu(inputs, alpha=0.02, name='relu1')
        inputs = tf.nn.relu(inputs)

        if is_first:
            h = shortcut(inputs)

        n_out_bottleneck = n_out // 4

        inputs = conv(inputs, n_out_bottleneck, k=1, s=1, name='conv1')

        if mode == 'DOWNSAMPLE':
            inputs = conv_same_pad(inputs, kernel_size=3, rate=1)
            inputs = conv(inputs, n_out_bottleneck, k=3, s=s, name='conv2', padding='VALID')
        else:
            inputs = conv(inputs, n_out_bottleneck, k=3, s=s, name='conv2')

        inputs = slim.batch_norm(inputs, activation_fn=None, decay=0.997, epsilon=1e-5, scale=True, is_training=is_training, scope='bn2')
        # inputs = lrelu(inputs, alpha=0.02, name='relu2')
        inputs = tf.nn.relu(inputs)

        inputs = conv(inputs, n_out, k=1, s=1, name='conv3')

        add = tf.add(h, inputs)

    return add


def residual_unit2(inputs: tf.Tensor,
                   is_training,
                   n_out=None,
                   name: str = 'rn',
                   mode: str = 'SAME',
                   is_first: bool = False) -> tf.Tensor:
    """
    Uses identity shortcut if `n_out` is None.
    """
    assert mode in ['DOWNSAMPLE', 'UPSAMPLE', 'SAME']

    shape = inputs.get_shape().as_list()
    dims = len(shape) - 2

    s = 1 if mode == 'SAME' else 2

    assert dims == 2

    nin = shape[-1]
    if n_out is None:
        n_out = nin

    with tf.variable_scope(name):
        def shortcut(x):
            split_out = x
            if mode == 'UPSAMPLE':
                # split_out = fixed_unpool(split_out, mode='NEAREST', name='unpool')
                h, w = split_out.get_shape().as_list()[1:3]
                split_out = tf.image.resize_nearest_neighbor(split_out, [h * 2, w * 2], name='upsample')
            elif mode == 'DOWNSAMPLE':
                raise NotImplementedError()
            if n_out > nin:
                padding = [[0, 0] for _ in range(len(shape))]
                padding[-1] = [int(math.floor((n_out - nin) / 2.0)), int(math.ceil((n_out - nin) / 2.0))]
                assert sum(padding[-1]) == n_out - nin
                split_out = tf.pad(split_out, padding, mode='CONSTANT', name='pad_shortcut')
            elif n_out < nin:
                split_out = conv2d(split_out, n_out, k=1, s=1, name='conv_shortcut')
            return split_out

        if n_out != nin:
            is_first = True

        if not is_first:
            h = shortcut(inputs)

        inputs = slim.batch_norm(inputs, activation_fn=None, decay=0.997, epsilon=1e-5, scale=True, is_training=is_training, scope='bn1')
        inputs = tf.nn.relu(inputs)

        if is_first:
            h = shortcut(inputs)

        if s > 1:
            inputs = deconv2d(inputs, n_out, k=4, s=s, name='conv1')
        else:
            inputs = conv2d(inputs, n_out, k=3, s=1, name='conv1')

        inputs = slim.batch_norm(inputs, activation_fn=None, decay=0.997, epsilon=1e-5, scale=True, is_training=is_training, scope='bn2')
        inputs = tf.nn.relu(inputs)

        inputs = conv2d(inputs, n_out, k=3, s=1, name='conv2')

        add = tf.add(h, inputs)

    return add


@tc.typecheck
def trainable_variable_norms(name: str = 'weight_norms') -> dict:
    """
    Returns a dict that maps trainable variable names to l2 norm Tensors.
    :param name: Name of the operation scope.
    :return: A dict with trainable variable names as keys and Tensors as values.
    """
    all_var_norms = {}
    with tf.name_scope(name):
        for v in tf.trainable_variables():
            reduction_axes = np.arange(1, v.get_shape().ndims)
            varname = v.name
            with tf.get_default_graph().colocate_with(v):
                v = tf.identity(v)
            l2norm = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.pow(v, 2),
                                                          reduction_indices=reduction_axes)),
                                    name=varname.replace(':0', '').replace(':', '_'))
            all_var_norms[varname] = l2norm
    return all_var_norms


@tc.typecheck
def layer_summary(value: nn_types.Value,
                  tag_suffix: tc.optional(str) = None,
                  collections: tc.optional(tc.seq_of(str)) = None):
    default_collections = [tf.GraphKeys.SUMMARIES]

    if collections is not None:
        if isinstance(collections, list) or isinstance(collections, tuple):
            default_collections.extend(collections)
        elif isinstance(collections, str):
            default_collections.append(collections)
        else:
            raise ValueError()

    tf.histogram_summary(tag=value.name, values=value, collections=default_collections)

    # stddev = tf.sqrt(tf.reduce_sum((value - tf.reduce_mean(value)) ** 2) / tf.cast(tf.size(value), tf.float32))
    # tf.scalar_summary(tags=value.name + '_stddev', values=stddev, collections=default_collections)


@tc.typecheck
def weight_layer(x: nn_types.Value,
                 name: str,
                 ch: int,
                 weight_type: tc.enum(
                     'conv2d',
                     'conv3d',
                     'deconv2d',
                     'deconv3d',
                     'linear'
                 ),
                 is_training: tc.any(tf.Tensor, bool) = False,
                 k=4,
                 s=2,
                 bn=True,
                 use_bias=None,
                 relu: tc.optional(callable) = lrelu):
    out = x
    assert out.get_shape().ndims >= 2

    if use_bias is None:
        use_bias = not bn

    with tf.variable_scope(name):
        if weight_type == 'linear':
            if out.get_shape().ndims != 2:
                out = flatten(out, name='flatten')
            out = linear(out, ch, use_bias=use_bias, name='linear')
        elif '2d' in weight_type:
            if out.get_shape().ndims != 4:
                out = conv_reshape(out, k=k, name='conv_reshape', dims=2)
            if weight_type == 'conv2d':
                out = conv2d(out, ch, k=k, s=s, use_bias=use_bias, name='conv')
            elif weight_type == 'deconv2d':
                out = deconv2d(out, ch, k=k, s=s, use_bias=use_bias, name='deconv')
            else:
                raise NotImplemented
        elif '3d' in weight_type:
            if out.get_shape().ndims != 5:
                conv_reshape(out, k=k, name='conv_reshape', dims=3)
            if weight_type == 'conv3d':
                out = conv3d(out, ch, k=k, s=s, use_bias=use_bias, name='conv')
            elif weight_type == 'deconv3d':
                out = deconv3d(out, ch, k=k, s=s, use_bias=use_bias, name='deconv')
            else:
                raise NotImplemented
        else:
            raise NotImplemented
        if bn:
            out = batch_norm(out, is_local=is_training, name='bn')

        var_name = out.name
        tag = 'response/' + var_name.replace(':0', '').replace(':', '_')
        tf.histogram_summary(tag, out, name=tag)

        if relu is not None:
            out = relu(out, name='relu')
            # out = tf.nn.relu(out, name='relu')
    return out


def npz_to_tensor(filename_tensor, dtype, shape, key='data'):
    assert filename_tensor.dtype == tf.string

    def _npz_to_array(filename):
        if isinstance(filename, bytes):
            filename = filename.decode('utf-8')
        arr = np.load(filename)[key]
        arr.shape = shape
        if arr.dtype != dtype.as_numpy_dtype():
            arr = arr.astype(dtype=dtype.as_numpy_dtype())
        assert arr.dtype == dtype.as_numpy_dtype()
        return arr

    tensor = tf.py_func(_npz_to_array, [filename_tensor], dtype, stateful=False)
    tensor.set_shape(shape)

    return tensor


# def out_of_range_if(cond, name=None):
#     """
#     Returns an operation that raises `tf.errors.OutOfRangeError` if `cond` is true.
#     """
#     with tf.variable_scope(name, default_name='out_of_range_if'):
#         # TODO(daeyun): This can be re-used.
#         dummy_var = tf.get_variable('zero', dtype=tf.int32, initializer=0, trainable=False,
#                                     collections=[tf.GraphKeys.LOCAL_VARIABLES])
#
#     with tf.name_scope(name, default_name='out_of_range_if') as scope:
#         # `tf.count_up_to` here always raises an exception. It is important to define this inside the lambda.
#         raise_op = tf.cond(cond,
#                            lambda: tf.count_up_to(dummy_var, 0, name='raise'),
#                            lambda: dummy_var.value(), name=scope).op
#     return raise_op
#
#
# def run_n_times_and_raise(count_var, limit_var, name=None):
#     """
#     """
#     assert isinstance(count_var, tf.Variable)
#     assert isinstance(limit_var, tf.Variable)
#
#     with tf.name_scope(name, default_name='CountUpToVariable') as scope:
#         cond = tf.greater(tf.assign_add(count_var, 1, use_locking=True, name='increment'), limit_var, name='condition')
#
#         def raise_op():
#             return tf.group(tf.count_up_to(count_var, count_var.dtype.base_dtype.min, name='raise'))
#
#         # `tf.count_up_to` here always raises an exception and does not modify `ref`.
#         # It is important to define this inside the lambda.
#         value = tf.cond(cond, raise_op, lambda: tf.no_op())
#
#         # value = tf.Print(value, [value, cond, count_var.value(), limit_var.value()], '###############')
#
#         return tf.group(value, name=scope)
#
#
def collection_identity(value, name='identity'):
    with tf.name_scope(name) as scope:
        if isinstance(value, (list, tuple)):
            return tuple(tf.identity(item, name='{}'.format(i)) for i, item in enumerate(value))
        elif isinstance(value, dict):
            return {k: tf.identity(item, name='{}'.format(i)) for i, (k, item) in enumerate(value.items())}
        elif isinstance(value, tf.Tensor):
            return tf.identity(value, name=scope)


#
#
# def limit_evaluation_count(tensor, count_var, limit_var, name=None):
#     """
#     Returns A `Tensor` with the same value as `tensor`.
#     Raises OutOfRangeError after evaluating limit_vartimes.
#     dtype of the atomic_counter is always int64.
#
#     See also: `tf.train.limit_epochs`
#     """
#     assert isinstance(count_var, tf.Variable)
#     assert isinstance(limit_var, tf.Variable)
#     with tf.name_scope(name, "LimitEvaluationCount") as scope:
#         atomic_counter = run_n_times_and_raise(count_var, limit_var)
#         with tf.control_dependencies([atomic_counter]):
#             out_tensor = collection_identity(tensor, name=scope)
#     return out_tensor


def select_one(funcs, index_tensor, name, names):
    assert isinstance(index_tensor, tf.Tensor)
    assert len(index_tensor.get_shape().as_list()) == 0
    assert index_tensor.dtype in (tf.int32, tf.int64)
    assert len(funcs) > 1

    @functools.lru_cache()
    def last():
        return funcs[-1]()

    with tf.name_scope(name) as scope:
        selected = tf.case({tf.equal(index_tensor, i): func
                            for i, func in enumerate(funcs[:-1])},
                           default=last,
                           exclusive=True, name=scope)
    if not isinstance(selected, (list, tuple)):
        return tf.identity(selected, name=scope)
    elif isinstance(selected, (list, tuple)):
        ret = []
        for i, item in enumerate(selected):
            with tf.name_scope(names[i]) as scope:
                ret.append(tf.identity(item, name=scope))
        return ret

    raise ValueError('Unexpected return value type')


def semaphore(value, device, name='semaphore'):
    # TODO(daeyun): not stable
    with tf.device(device), tf.name_scope(name) as scope:
        lock = tf.FIFOQueue(capacity=value, dtypes=tf.bool, shapes=(),
                            shared_name='blocking_queue', name='blocking_queue')
        lock.close(cancel_pending_enqueues=True, name='close')
        return lambda: lock.enqueue(True, name='acquire'), lambda: lock.dequeue(name='release')


def barrier(size, device, name='barrier'):
    """
    A reusable barrier for synchronizing distributed workers. This function implements atomic entry and release operations using two queues.
    Takes around 2ms with 6 workers on a single machine.

    Operations:

    - {name}/barrier: First caller becomes the chief worker responsible for waiting, releasing,
                      and resetting after this is called `size` times.
                      Evaluates to a unique integer in [1, size] for each caller, in the order they entered.
    - {name}/init_op: This needs to be called once in the beginning.
    - {name}/close: An operation to close the underlying queues.
    - {name}/remaining: Number of remaining workers.

    :param size: Number of workers to synchronize.
    :param device: Device to place the queues. This should be the same one in all replicas.
    :param name: A name for the operations.
    :return: `{name}/barrier:0`.
    """
    entry_tokens = np.arange(size, dtype=np.int32)
    release_tokens = np.ones(shape=[size - 1], dtype=np.bool)

    with tf.device(device), tf.name_scope(name) as scope:
        entry_queue = tf.FIFOQueue(capacity=entry_tokens.size, dtypes=tf.int32, shapes=(),
                                   shared_name='{}/entry_queue'.format(name), name='entry_queue')
        release_queue = tf.FIFOQueue(capacity=release_tokens.size, dtypes=tf.bool, shapes=(),
                                     shared_name='{}/release_queue'.format(name), name='release_queue')
        entry_queue.size('remaining')

        # Not needed in most use-cases.
        with tf.control_dependencies([entry_queue.close(cancel_pending_enqueues=True, name='entry_queue/close')]):
            with tf.control_dependencies([]):
                tf.group(release_queue.close(cancel_pending_enqueues=True, name='release_queue/close'), name='close')

        def build_chief_wait_op():
            # Wait until all workers enter the barrier.
            with tf.control_dependencies([entry_queue.enqueue_many(entry_tokens, name='chief_wait_full')]):
                # Before unblocking `release_queue`, temporarily block `entry_queue` so that entries to the next barrier can only be accepted after `release_queue` is emptied.
                with tf.control_dependencies([entry_queue.dequeue_many(entry_tokens.size, name='block_entry')]):
                    # Unblock all num_replicas - 1 workers. At this point, they should all be waiting in the `wait_for_release` op.
                    with tf.control_dependencies([release_queue.enqueue_many(release_tokens, name='release')]):
                        # This `enqueue_many` cannot happen until `release_queue` is empty.
                        with tf.control_dependencies([release_queue.enqueue_many(release_tokens, name='wait_released')]):
                            # Clear `release_queue` for the next use.
                            with tf.control_dependencies([release_queue.dequeue_many(release_tokens.size, name='reset_release')]):
                                # Populate `entry_queue` with tokens.
                                chief_wait_op = entry_queue.enqueue_many(entry_tokens, name='accept_new_entries')
            return chief_wait_op

        init_op = entry_queue.enqueue_many(entry_tokens, name='init')

        entry_token = entry_queue.dequeue(name='enter')
        with tf.control_dependencies([entry_token.op]):
            wait = tf.cond(tf.equal(entry_token, 0), build_chief_wait_op,
                           lambda: release_queue.dequeue(name='wait_for_release'), name='wait')
            with tf.control_dependencies([wait]):
                ret = tf.identity(entry_token, name=scope)

        return ret
