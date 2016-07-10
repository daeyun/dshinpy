import tensorflow as tf
from dshin import nn


class ImageEncoder(nn.build.Builder):
    @property
    def in_dims(self) -> int:
        return 2


class Simple(ImageEncoder):
    def _build(self, input_value: tf.Tensor):
        out = input_value

        out = nn.ops.conv2d(out, self.ch, k=4, s=2, name='conv1')
        out = nn.ops.lrelu(out, name='l1_relu')

        out = nn.ops.conv2d(out, self.ch * 2, k=4, s=2, name='conv2')
        out = nn.ops.lrelu(out, name='l2_relu')

        out = nn.ops.conv2d(out, self.ch * 4, k=4, s=2, name='conv3')
        out = nn.ops.lrelu(out, name='l3_relu')

        out = nn.ops.conv2d(out, self.ch * 8, k=4, s=2, name='conv4')
        out = nn.ops.lrelu(out, name='l4_relu')

        out = nn.ops.flatten(out, name='flatten')
        out = nn.ops.linear(out, self.out_size)
        out = nn.ops.lrelu(out, name='l5_relu')
        return out


if __name__ == '__main__':
    Simple('encoder')


class Simple3d(nn.build.Builder):
    def build(self, input_value: tf.Tensor):
        out = input_value
        with tf.variable_scope(self.name):
            out = nn.ops.conv3d(out, self.ch, k=4, s=2, name='conv1')
            out = nn.ops.lrelu(out, name='l1_relu')

            out = nn.ops.conv3d(out, self.ch * 2, k=4, s=2, name='conv2')
            out = nn.ops.lrelu(out, name='l2_relu')

            out = nn.ops.conv3d(out, self.ch * 4, k=4, s=2, name='conv3')
            out = nn.ops.lrelu(out, name='l3_relu')

            out = nn.ops.conv3d(out, self.ch * 8, k=4, s=2, name='conv4')
            out = nn.ops.lrelu(out, name='l4_relu')

            out = nn.ops.flatten(out, name='flatten')
            out = nn.ops.linear(out, out_linear)
            out = nn.ops.lrelu(out, name='l5_relu')
        return out
