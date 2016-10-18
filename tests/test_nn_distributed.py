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

        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')

        with tf.name_scope('summary'):
            self.add_scalar_summary('loss', cross_entropy, name='loss')
            self.add_scalar_summary('accuracy', accuracy, name='accuracy')

        return cross_entropy


class DataFeeder(nn.distributed.TFProcess):
    def _enqueue_all(self, net, queue_name, data_source, enqueue_batch_size):

        num_examples = data_source.num_examples
        num_examples_enqueued = 0
        self.log('Feeding {} items to {}'.format(num_examples, queue_name))

        while num_examples_enqueued < num_examples:
            batch_size = min(num_examples - num_examples_enqueued, enqueue_batch_size)
            x, y = data_source.next_batch(batch_size)
            net.enqueue(queue_name, {
                'input': x,
                'target': y,
            })
            num_examples_enqueued += batch_size

        index = net.current_session_index()
        net.close_source_queue(queue_name)
        self.log('Closed queue {} after enqueueing {} items.'.format(queue_name, num_examples_enqueued))
        net.wait_for_session_index(index + 1)

    def _main(self, server: tf.train.Server, net: nn.NNModel):
        assert isinstance(net, nn.NNModel)

        enqueue_batch_size = 51

        net.build(
            server,
            source_queue_names=['train', 'eval', 'eval2', 'eval3'],
            source_queue_sizes=[1000],
            local_queue_size=300,
            seed=42,
            sync_replicas=True,
            save_model_secs=600,
            # parallel_read_size=20,
        )

        while True:
            mnist = input_data.read_data_sets('/tmp/MNIST_data', one_hot=True)

            self._enqueue_all(net, 'train', mnist.train, enqueue_batch_size)
            self._enqueue_all(net, 'eval', mnist.test, enqueue_batch_size)
            # self._enqueue_all(net, 'train', mnist.train, enqueue_batch_size)

        server.join()


class Trainer(nn.distributed.TFProcess):
    def _main(self, server: tf.train.Server, net: nn.NNModel):
        net.warn_io_bottleneck = False
        net.build(
            server,
            source_queue_names=['train', 'eval', 'eval2', 'eval3'],
            source_queue_sizes=[1000],
            local_queue_size=300,
            seed=42,
            sync_replicas=True,
            save_model_secs=600,
            # parallel_read_size=20,
        )
        for i in range(1000):
            losses = net.train(source='train', batch_size=50, summary_keys=['scalar'])
            net.eval_scalars('eval', ['loss', 'accuracy'], batch_size=55, summary_keys=['scalar'])

        server.join()


def main(_):
    cluster = nn.distributed.get_local_cluster_spec({
        'ps': 2,
        'worker': 6,
        'data': 1,
    })

    # session_config = nn.utils.default_sess_config(log_device_placement=False)
    session_config = None

    log_dir = '/tmp/mnist_test_19'

    while True:
        processes = []
        processes.append(nn.distributed.ParameterServer(cluster, task_id=0, gpu_ids=()))
        processes.append(nn.distributed.ParameterServer(cluster, task_id=1, gpu_ids=()))
        processes.append(DataFeeder(cluster, job_name='data', task_id=0, nnmodel_class=MV,
                                    experiment_name='mnist_train', logdir=log_dir, gpu_ids=()))

        gpu_ids = [0, 1] + [None] * 4

        for i, gid in enumerate(gpu_ids):
            gpu_ids = [] if gid is None else [gid]
            processes.append(Trainer(cluster, job_name='worker', task_id=i, nnmodel_class=MV,
                                     experiment_name='mnist_train', logdir=log_dir,
                                     session_config=session_config, gpu_ids=gpu_ids, batch_size=55))

        for p in processes:
            p.start()

        # for p in processes:
        #     if not isinstance(p, nn.distributed.ParameterServer):
        #         p.join()
        #
        # for p in processes:
        #     if isinstance(p, nn.distributed.ParameterServer):
        #         p.terminate()

        time.sleep(1000000)
        print('done!!')


if __name__ == '__main__':
    tf.app.run(main)
