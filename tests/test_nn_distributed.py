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


class DataFeeder(nn.distributed.TFProcess):
    def _main(self, server: tf.train.Server, net: nn.NNModel):
        queue_name = 'train'
        assert isinstance(net, nn.NNModel)
        log.info('Feeder %s started', queue_name)

        enqueue_batch_size = 100

        net.build(
            server,
            source_queue_names=[queue_name],
            source_queue_sizes=[1000],
            local_queue_size=300,
            seed=42,
            sync_replicas=True,
            save_model_secs=600,
            # parallel_read_size=20,
        )

        while True:
            mnist = input_data.read_data_sets('/tmp/MNIST_data', one_hot=True)
            data_source = mnist.train
            num_examples = data_source.num_examples
            num_examples_enqueued = 0
            while num_examples_enqueued < num_examples:
            # while num_examples_enqueued < 5000:
                batch_size = min(num_examples - num_examples_enqueued, enqueue_batch_size)
                x, y = data_source.next_batch(batch_size)
                net.enqueue(queue_name, {
                    'input': x,
                    'target': y,
                })
                num_examples_enqueued += batch_size

            net.close_source_queue(queue_name)

        server.join()


class Trainer(nn.distributed.TFProcess):
    def _main(self, server: tf.train.Server, net: nn.NNModel):
        queue_name = 'train'

        net.warn_io_bottleneck = False
        net.build(
            server,
            source_queue_names=[queue_name],
            source_queue_sizes=[1000],
            local_queue_size=300,
            seed=42,
            sync_replicas=True,
            save_model_secs=600,
            # parallel_read_size=20,
        )
        for i in range(10000):
            losses = net.train(source='train', batch_size=self._batch_size)
            net.eval_scalars('train', 'loss', batch_size=1)

        server.join()


def main(_):
    cluster = nn.distributed.get_local_cluster_spec({
        'ps': 2,
        'worker': 6,
        'data': 1,
    })

    # session_config = nn.utils.default_sess_config(log_device_placement=False)
    session_config = None

    log_dir = '/tmp/mnist_test_5'

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
                                     session_config=session_config, gpu_ids=gpu_ids, batch_size=50))

        for p in processes:
            p.start()

        # for p in processes:
        #     if not isinstance(p, nn.distributed.ParameterServer):
        #         p.join()
        #
        # for p in processes:
        #     if isinstance(p, nn.distributed.ParameterServer):
        #         p.terminate()

        time.sleep(10000)
        print('done!!')


if __name__ == '__main__':
    tf.app.run(main)
