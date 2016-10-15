import tensorflow as tf
import percache
import shelve
import plyfile
from multiview_shape import reconstruction
import functools
import glob
import copy
from dshin import transforms
from multiview_shape import mve
from dshin import io_utils
import typing
import random
from os import path
import copy
import tflearn
import tqdm
import ujson
import collections
import zlib
from dshin import log
import pprint
import array
import gzip
import sys
import time
import numpy as np
from os import path
import threading
from multiview_shape import db_model as dbm
from dshin import nn
from multiview_shape.data import cvpr15_shrec12
from multiview_shape.data import new_modelnet40
from tensorflow.contrib import slim
from tensorflow.contrib.slim import nets
import os
from multiview_shape.tf_models import loss_op
from multiview_shape import metrics
from dshin import geom2d
from dshin import geom3d
from multiview_shape.data import viewpoint
from dshin.third_party import gflags as flags
import numpy.linalg as la
import dshin.data
import dshin

from tensorflow.examples.tutorials.mnist import input_data


class MV(nn.NNModel):
    def _placeholders(self):
        return [
            nn.QueuePlaceholder(tf.float32, shape=[None, 784], name='input'),
            nn.QueuePlaceholder(tf.float32, shape=[None, 10], name='target'),
        ]

    def _model(self) -> tf.Tensor:
        x = self.placeholder('input')
        y_ = self.placeholder('target')

        W = tf.Variable(tf.zeros([784, 10]), name='weight')
        b = tf.Variable(tf.zeros([10]))
        y = tf.matmul(x, W) + b
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_), name='loss')

        return cross_entropy


def feeder(net, data_source, enqueue_batch_size=50, queue_name='train'):
    assert isinstance(net, nn.NNModel)
    log.info('Feeder %s started', queue_name)
    num_examples = data_source.num_examples
    num_examples_enqueued = 0
    while True:
        while num_examples_enqueued < num_examples:
            batch_size = min(num_examples - num_examples_enqueued, enqueue_batch_size)
            x, y = data_source.next_batch(batch_size)
            net.enqueue(queue_name, {
                'input': x,
                'target': y,
            })
            num_examples_enqueued += batch_size
        net.close_source_queue(queue_name)
        break
    log.info('Feeder %s is quitting', queue_name)


if __name__ == '__main__':
    mnist = input_data.read_data_sets('/tmp/MNIST_data', one_hot=True)

    net = MV()
    net.build()

    threading.Thread(target=feeder, args=(net, mnist.train, 20, 'train'), daemon=True).start()

    net.warn_io_bottleneck = False

    losses = net.train(source='train', batch_size=10)
