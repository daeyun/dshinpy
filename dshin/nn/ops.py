"""
Helper functions for building TensorFlow neural net layers.
"""
import math

import numpy as np
import tensorflow as tf
import tensorflow.python.framework.ops as tf_ops

from dshin.third_party import gflags

FLAGS = gflags.FLAGS

gflags.DEFINE_boolean('use_fp16', False, """Train the model using fp16.""")


def _variable_on_cpu(name, shape, initializer, trainable=True):
    with tf.device('/gpu:0'):
        dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
        return tf.get_variable(name, shape, initializer=initializer, dtype=dtype, trainable=trainable)


def conv2d(input_tensor: tf_ops.Tensor, n_out, k=5, s=1, name="conv2d", padding='SAME', use_bias=False) -> tf_ops.Tensor:
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


def deconv2d(input_tensor: tf_ops.Tensor, n_out, k=5, s=1, name='deconv2d', padding='SAME', use_bias=False) -> tf_ops.Tensor:
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


def conv3d(input_tensor: tf_ops.Tensor, n_out, k=5, s=1, name="conv3d", padding='SAME', use_bias=False) -> tf_ops.Tensor:
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


def deconv3d(input_tensor: tf_ops.Tensor, n_out, k=5, s=1, name='deconv3d', padding='SAME', use_bias=False) -> tf_ops.Tensor:
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


def linear(input_tensor: tf_ops.Tensor, n_out: int, name='linear', use_bias=False) -> tf_ops.Tensor:
    assert input_tensor.get_shape().ndims == 2
    assert isinstance(n_out, int)

    with tf.variable_scope(name):
        n_in = input_tensor.get_shape().as_list()[-1]
        stddev = math.sqrt(2.0 / n_in)
        w = _variable_on_cpu('w', [n_in, n_out], initializer=tf.random_normal_initializer(stddev=stddev))
        b = _variable_on_cpu('b', [n_out], initializer=tf.constant_initializer(0))
        lin = tf.matmul(input_tensor, w)

        if not use_bias:
            return lin

        return tf.nn.bias_add(lin, b)


def fixed_unpool(value, name='unpool', mode='ZERO_FILLED'):
    """
    Upsampling operation.

    :param value: A Tensor of shape [b, d0, d1, ..., dn, ch]
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


def fixed_pool(value, name='pool'):
    """
    Downsampling operation.

    :param value: A Tensor of shape [b, d0, d1, ..., dn, ch]
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


def lrelu(input_tensor: tf.Tensor, alpha=0.05, name='lrelu') -> tf.Tensor:
    """
    Leaky ReLU.
    """
    with tf.variable_scope(name):
        return tf.maximum(alpha * input_tensor, input_tensor, name=name)


def ema_with_initial_value(value, initial_value=0.0, ema_trainer=None, decay=None, name='ema'):
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


def ema_with_update_dependencies(values, initial_values=0.0, decay=0.99, name='ema'):
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


def batch_norm(value, is_local=True, name='bn', offset=0.0, scale=0.02, ema_decay=0.99, return_mean_var=False, is_trainable=True):
    """
    Batch normalization layer.

    For updating the EMA variables, use the following idiom from TensorFlow documentation:

    ::
        # Create an op that will update the moving averages after each training step.
        with tf.variable_scope('train'):
            minimize = optimizer.minimize(loss, name='minimize')
            with tf.control_dependencies([minimize_op]):
                train_step = tf.group(*update_ops, name='step')

    :param is_local: A Tensor of type tf.bool or a boolean constant. Indicates whether the output tensor should use
    local stats or moving averages. It is not related to updating the shadow variables.
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


def flatten(value, name='flatten'):
    """
    Flattens the input tensor's shape to be linear.

    :param value: A Tensor of shape [b, d0, d1, ...]
    :return: A Tensor of shape [b, d0*d1*...]
    """
    with tf.name_scope(name) as scope:
        num_linear = np.prod(value.get_shape().as_list()[1:])
        return tf.reshape(value, [-1, num_linear], name=scope)


def conv_reshape(value, k, num_channels=None, name='conv_reshape', dims=2):
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


def stack_batch_dim(value_list, name='batch_stack'):
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


def apply_concat(value_list, factory, name_prefix='branch') -> tf.Tensor:
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


def residual_unit(x, is_training, n_out=None, name='rn', mode='SAME', is_first=False):
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
