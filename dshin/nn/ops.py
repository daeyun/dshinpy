"""
Helper functions for building TensorFlow neural net layers.
"""
import functools
import math

import numpy as np
import tensorflow as tf
import typecheck as tc

from dshin.nn import types as nn_types
from dshin.third_party import gflags

FLAGS = gflags.FLAGS

gflags.DEFINE_boolean('use_fp16', False, """Train the model using fp16.""")


def layer(func):
    @functools.wraps(func)
    def wrapper(*args, **kwds):
        output = func(*args, **kwds)
        tf.get_default_graph().add_to_collection(tf.GraphKeys.ACTIVATIONS, output)
        return output

    return wrapper


@tc.typecheck
def _variable_on_cpu(name: str, shape: tc.seq_of(int), initializer: callable, trainable: bool = True) -> tf.Variable:
    with tf.device('/cpu:0'):
        dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
        return tf.get_variable(name, shape, initializer=initializer, dtype=dtype, trainable=trainable)


@layer
@tc.typecheck
def conv2d(input_tensor: nn_types.Value,
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
        w = _variable_on_cpu('w', [k, k, n_in, n_out], initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input_tensor, w, strides=[1, s, s, 1], padding=padding, use_cudnn_on_gpu=True)

        if not use_bias:
            return conv

        b = _variable_on_cpu('b', [n_out], initializer=tf.constant_initializer(0))
        return tf.nn.bias_add(conv, b)


@layer
@tc.typecheck
def deconv2d(input_tensor: nn_types.Value,
             n_out: int,
             k: int = 5,
             s: int = 1,
             name: str = 'deconv2d',
             padding: str = 'SAME',
             use_bias: bool = False) -> tf.Tensor:
    assert input_tensor.get_shape().ndims == 4

    with tf.variable_scope(name):
        input_shape = input_tensor.get_shape().as_list()
        n_in = input_shape[-1]
        stddev = math.sqrt(2.0 / ((k ** 2) / (s ** 2) * n_in))
        w = _variable_on_cpu('w', [k, k, n_out, n_in], initializer=tf.random_normal_initializer(stddev=stddev))

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

        b = _variable_on_cpu('b', [n_out], initializer=tf.constant_initializer(0))
        return tf.nn.bias_add(deconv, b)


@layer
@tc.typecheck
def conv3d(input_tensor: nn_types.Value,
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
        w = _variable_on_cpu('w', [k, k, k, n_in, n_out], initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv3d(input_tensor, w, strides=[1, s, s, s, 1], padding=padding)

        if not use_bias:
            return conv

        b = _variable_on_cpu('b', [n_out], initializer=tf.constant_initializer(0))
        return tf.nn.bias_add(conv, b)


@layer
@tc.typecheck
def deconv3d(input_tensor: nn_types.Value,
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
        stddev = math.sqrt(2.0 / ((k ** 3) / (s ** 3) * n_in))
        w = _variable_on_cpu('w', [k, k, k, n_out, n_in], initializer=tf.random_normal_initializer(stddev=stddev))

        if type(input_shape[0]) == int:
            batchsize = input_shape[0]
        else:
            batchsize = tf.shape(input_tensor)[0]

        output_shape = [batchsize, s * input_shape[1], s * input_shape[2], s * input_shape[3], n_out]
        deconv = tf.nn.conv3d_transpose(value=input_tensor, filter=w, output_shape=output_shape, strides=[1, s, s, s, 1], padding=padding)  # type: tf.Tensor

        if type(output_shape[0]) != int:
            output_shape[0] = -1

        deconv.set_shape(output_shape)

        if not use_bias:
            return deconv

        b = _variable_on_cpu('b', [n_out], initializer=tf.constant_initializer(0))
        return tf.nn.bias_add(deconv, b)


@layer
@tc.typecheck
def linear(input_tensor: nn_types.Value,
           n_out: int,
           name: str = 'linear',
           use_bias: bool = False) -> tf.Tensor:
    assert input_tensor.get_shape().ndims == 2
    assert isinstance(n_out, int)

    with tf.variable_scope(name):
        n_in = input_tensor.get_shape().as_list()[-1]
        stddev = math.sqrt(2.0 / n_in)
        w = _variable_on_cpu('w', [n_in, n_out], initializer=tf.random_normal_initializer(stddev=stddev))
        lin = tf.matmul(input_tensor, w)

        if not use_bias:
            return lin

        b = _variable_on_cpu('b', [n_out], initializer=tf.constant_initializer(0))
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


@layer
@tc.typecheck
def lrelu(input_tensor: nn_types.Value, alpha: float = 0.05, name: str = 'lrelu') -> tf.Tensor:
    """
    Leaky ReLU.
    """
    with tf.variable_scope(name):
        return tf.maximum(alpha * input_tensor, input_tensor, name=name)


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
            value_copy = _variable_on_cpu('value_copy', shape=shape, initializer=init, trainable=False)
            with tf.control_dependencies([tf.assign(value_copy, value)]):
                apply_op = ema_trainer.apply([value_copy])
                moving_average = ema_trainer.average(value_copy)

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
            value_copy = _variable_on_cpu(value.name + ':ema', shape=shape, initializer=init, trainable=False)
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
    with tf.variable_scope(name):
        beta = _variable_on_cpu('beta', shape=[n_out], initializer=tf.constant_initializer(offset), trainable=is_trainable)
        gamma = _variable_on_cpu('gamma', shape=[n_out], initializer=tf.constant_initializer(scale), trainable=is_trainable)

        batch_mean, batch_var = tf.nn.moments(value, axes, name='moments')
        batch_var = tf.maximum(batch_var, 0.0)

        ema_trainer = tf.train.ExponentialMovingAverage(decay=ema_decay)

        ema_batch_mean = ema_with_initial_value(batch_mean, 0.0, ema_trainer)
        ema_batch_var = ema_with_initial_value(batch_var, 1.0, ema_trainer)

        if isinstance(is_local, tf.Tensor) and is_local.dtype == tf.bool:
            mean, var = tf.cond(is_local, lambda: (batch_mean, batch_var), lambda: (ema_batch_mean, ema_batch_var))
        else:
            assert isinstance(is_local, bool)
            mean, var = (batch_mean, batch_var) if is_local else (ema_batch_mean, ema_batch_var)

        bn = tf.nn.batch_normalization(value, mean, var, beta, gamma, 1e-5)

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


@tc.typecheck
@layer
def residual_unit(x: nn_types.Value,
                  is_training: bool,
                  n_out: tc.optional(int) = None,
                  name: str = 'rn',
                  mode: str = 'SAME',
                  is_first: bool = False) -> tf.Tensor:
    """
    Uses identity shortcut if `n_out` is None.
    """
    assert mode in ['DOWNSAMPLE', 'UPSAMPLE', 'SAME']

    shape = x.get_shape().as_list()
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
                split_out = fixed_unpool(split_out, mode='NEAREST', name='unpool')
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
            h = shortcut(x)

        x = batch_norm(x, is_training, name='bn1')
        x = lrelu(x, name='relu1')

        if is_first:
            h = shortcut(x)

        x = conv(x, n_out, k=3, s=s, name='conv1')

        x = batch_norm(x, is_training, name='bn2')
        x = lrelu(x, name='relu2')

        x = conv(x, n_out, k=3, s=1, name='conv2')

        add = tf.add(h, x)

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
