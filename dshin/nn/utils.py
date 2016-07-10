import abc
import contextlib
import time
import functools
import os
import re
import math
import typing
import tqdm
from os import path

import tensorflow as tf
import toposort

from dshin import log
from dshin import timer


class NNModel(metaclass=abc.ABCMeta):
    """Neural network container.
    """

    # TF graph collection keys.
    UPDATE_OPS = tf.GraphKeys.UPDATE_OPS
    TRAIN_OPS = 'train_ops'
    PLACEHOLDERS = 'placeholders'
    OUTPUTS = 'outputs'
    LOSSES = 'losses'
    INDICATORS = 'indicators'
    METRICS = 'metrics'
    IO = 'io'

    @classmethod
    def from_file(cls, restore_path):
        net = cls()
        net.restore(restore_path=restore_path)
        return net

    def __init__(self, graph: tf.Graph):
        assert isinstance(graph, tf.Graph)

        self.graph = graph
        with self.graph.as_default():
            self._build()
            self.saver = tf.train.Saver(name='saver')

            if self.count('placeholder/is_training') != 1:
                log.warn('"placeholder/is_training" is not defined.')

        self.needs_initialization = True

    @abc.abstractmethod
    def _build(self):
        return

    def initialize(self):
        if self.needs_initialization:
            with self.graph.as_default():
                tf.initialize_all_variables().run(
                    session=tf.get_default_session())
                self.needs_initialization = False

    def restore(self, restore_path):
        with self.graph.as_default():
            self.saver.restore(tf.get_default_session(), restore_path)
            self.needs_initialization = False
            log.info("Restored model from %s", restore_path)

    def save(self, save_path):
        session = tf.get_default_session()

        with session.as_default():
            save_path = path.expanduser(save_path)

            dirpath = path.dirname(save_path)
            if not path.isdir(dirpath):
                log.info('mkdir %s', dirpath)
                os.makedirs(dirpath)

            save_path_out = self.saver.save(session, save_path)
            log.info("Model saved to file: %s" % save_path_out)

    def collection_str(self, pattern=r'.*'):
        """ Return all matching names in any collection. Used for development.
        :param pattern: Item is matched if this pattern is found in the name.
        """
        out_str = ''
        keys = self.graph.get_all_collection_keys()
        for k in keys:
            items = self.graph.get_collection(k)
            out_str += k + '\n'
            for item in items:
                if re.match(pattern, item.name):
                    out_str += '    ' + item.name + '\n'
        return out_str

    @functools.lru_cache(maxsize=2048, typed=True)
    def __getitem__(self, pattern):
        assert isinstance(pattern, str)
        return self.get(pattern)

    @functools.lru_cache(maxsize=2048, typed=True)
    def get_all(self, pattern, collection_key=None):
        matched = []
        keys = self.graph.get_all_collection_keys()
        if collection_key is None:
            for k in keys:
                for item in self.graph.get_collection(k):
                    if re.match(pattern, item.name):
                        matched.append(item)
        else:
            assert collection_key in keys
            for item in self.graph.get_collection(collection_key):
                if re.match(pattern, item.name):
                    matched.append(item)
        return matched

    @functools.lru_cache(maxsize=2048, typed=True)
    def count(self, pattern):
        return len(self.get_all(pattern))

    @functools.lru_cache(maxsize=2048, typed=True)
    def get_all_ops(self, pattern):
        matched = []
        ops = self.graph.get_operations()
        for op in ops:
            if re.match(pattern, op.name):
                matched.append(op)
        return matched

    def get(self, pattern, collection_key=None):
        matched = self.get_all(pattern, collection_key)
        if len(matched) != 1:
            raise ValueError('len(matched) != 1. matched: \n{}\n{}'.format(
                pattern, '\n'.join([item.name for item in matched])))
        return matched[0]

    @functools.lru_cache(maxsize=2048, typed=True)
    def last(self, pattern=r'.*'):
        sorted = self.toposort(pattern)
        last = list(sorted[-1])
        assert len(last) == 1
        return self.graph.get_tensor_by_name(last[0])

    @functools.lru_cache(1024)
    def toposort(self, pattern=r'.*'):
        ops = self.get_all_ops(pattern)

        # http://stackoverflow.com/a/33851308
        deps = {}
        for op in ops:
            # op node
            op_inputs = set()
            op_inputs.update([t.name for t in op.inputs])
            deps[op.name] = op_inputs
            # tensor output node
            for t in op.outputs:
                deps[t.name] = {op.name}
        return list(toposort.toposort(deps))

    def eval(self, tensor_or_op_patterns: typing.Sequence, feed_dict: dict,
             is_training: bool = False):
        assert isinstance(tensor_or_op_patterns, list)
        assert isinstance(feed_dict, dict)

        new_feed_dict = {}
        for k, v in feed_dict.items():
            if isinstance(k, str):
                new_feed_dict[self[k]] = v
            elif isinstance(k, tf.Tensor):
                new_feed_dict[k] = v
            else:
                raise ValueError('Unexpected key in feed_dict: {}'.format(k))

        if self.count('placeholder/is_training') == 1:
            new_feed_dict[self['placeholder/is_training']] = is_training
        elif is_training:
            raise ValueError(
                'is_training is True, but placeholder/is_training is not found.')

        # TODO(daeyun): warn if 'train/step' does not update moving averages.

        sess = tf.get_default_session()
        fetches = [self[pattern] for pattern in tensor_or_op_patterns]
        if is_training:
            # There might be a race condition here, but it does not matter most of the time.
            fetches += [self['train/step']]
        out_eval = sess.run(fetches, new_feed_dict)
        results = {}
        for name, result in zip(tensor_or_op_patterns, out_eval):
            if result is not None:
                results[name] = result
        return results






class EncoderDecoder(NNModel):
    def __init__(self, graph: tf.Graph, in_dims):
        super().__init__(graph)
        self.in_dims = in_dims

    def _build(self):
        with tf.variable_scope('placeholder'):
            in_shape = [None] + config.in_dims * [config.in_resolution] + [1]
            x = tf.placeholder(tf.float32, shape=in_shape, name='input')
            is_training = tf.placeholder(tf.bool, name='is_training')


if __name__ == '__main__':
    graph = tf.Graph()
    EncoderDecoder(graph)
