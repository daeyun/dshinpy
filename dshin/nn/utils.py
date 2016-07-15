"""
Helpers for managing TensorFlow neural net models.
"""
import abc
import functools
import os
import re
import typing
from os import path

import ensure
import tensorflow as tf
import toposort

from dshin import log
from dshin import nn


@ensure.ensure_annotations
def sort_tensors(ops: nn.types.Operations) -> typing.Sequence[str]:
    """
    Sorts the inputs and outputs of TensorFlow Operations in topological order.

    :param ops: List of Operations.
    :return: Sorted list of Tensor names.
    """
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

    out = []
    tensors = toposort.toposort(deps)
    for tensor in tensors:
        out.extend(list(tensor))

    return out


def match_names(values: nn.types.NamedSeq, pattern: str = None,
                prefix: str = None, suffix: str = None) -> nn.types.NamedSeq:
    """
    Filters TensorFlow graph objects by regular expression.

    :param values: A list of Variables, Tensors, or Operations.
    :param pattern: A regular expression pattern.
    :param prefix: A string prepended to `pattern`. Defaults to ``.*?``.
    :param suffix: A string appended to `pattern`. Defaults to ``.*?``.
    :return:
    """
    if pattern is None:
        pattern = r'.*?'
    if prefix is None:
        prefix = r'.*?'
    if suffix is None:
        suffix = r'.*?'

    final_pattern = ''.join([prefix, pattern, suffix])
    matched = []
    for item in values:
        if re.match(final_pattern, item.name):
            matched.append(item)
    return matched


class NNModel(metaclass=abc.ABCMeta):
    """
    TensorFlow neural net container.

    Implementation example:

    >>> class SampleNet(NNModel):
    ...     def _model(self):
    ...         input_placeholder = self['input']
    ...         target_placeholder = self['target']
    ... 
    ...         out = nn.ops.conv2d(input_placeholder, n_out=1, use_bias=False)
    ...         out = nn.ops.batch_norm(out, is_trainable=True, is_local=True)
    ...         self._loss = tf.reduce_mean((out - target_placeholder) ** 2, name='loss')
    ... 
    ...     def _minimize_op(self):
    ...         lr_placeholder = self['learning_rate']
    ...         return tf.train.AdamOptimizer(lr_placeholder).minimize(self._loss)
    ... 
    ...     def _placeholders(self):
    ...         return [
    ...             tf.placeholder(tf.float32, shape=[None, 5, 5, 1], name='input'),
    ...             tf.placeholder(tf.float32, shape=[None, 5, 5, 1], name='target'),
    ...         ]

    Training example:

    >>> import numpy as np
    >>> net = SampleNet(seed=42)
    >>> np.random.seed(42)
    >>> feed_dict = {'input': np.random.randn(2, 5, 5, 1),
    ...              'target': np.random.randn(2, 5, 5, 1),
    ...              'learning_rate': 0.001}
    >>> print('{0:.5f}'.format(net.eval(['loss'], feed_dict)['loss']))
    0.75191
    >>> net.train(feed_dict)
    >>> print('{0:.5f}'.format(net.eval(['loss'], feed_dict)['loss']))
    0.75167
    >>> print('{0:.5f}'.format(net.eval(['loss'], feed_dict)['loss']))
    0.75167

    Save and restore:

    >>> net.save('/tmp/sample_net/saved')
    >>> net_restored = SampleNet.from_file('/tmp/sample_net/saved')
    >>> print('{0:.5f}'.format(net_restored.eval(['loss'], feed_dict)['loss']))
    0.75167
    >>> net.train(feed_dict)
    >>> net_restored.train(feed_dict)
    >>> print('{0:.5f}'.format(net_restored.eval(['loss'], feed_dict)['loss']))
    0.75143
    >>> print('{0:.5f}'.format(net.eval(['loss'], feed_dict)['loss']))
    0.75143
    """

    _placeholder_prefix = 'placeholder'
    _meta_graph_suffix = '.meta'
    _cached = functools.lru_cache(maxsize=2048, typed=True)

    @classmethod
    def from_file(cls, restore_path: str):
        """
        A factory method that returns an instance of a previously saved model.

        :param restore_path: The path used to save the model.
        :return: NNModel instance.
        """
        net = RestoredNNModel(build=False)
        net.restore(restore_path=restore_path)
        return net

    def __init__(self, sess: tf.Session = None, seed: int = None, build=True):
        """
        Creates a new model instance.

        :param sess: A TensorFlow Session.
        :param seed: The graph-level random seed.
        :param build: If true, builds and initializes the model.
        """
        if sess is None:
            self.graph = tf.Graph()
            self.session = tf.Session(graph=self.graph)
        else:
            assert isinstance(sess, tf.Session)
            self.graph = sess.graph
            self.session = sess

        self.seed = seed
        self.needs_initialization = True

        if self.seed is not None:
            with self.graph.as_default():
                # Graph-level random seed.
                tf.set_random_seed(self.seed)

        if build:
            self._build_model()

    def _build_model(self):
        """
        Populates the graph using user-defined functions.
        """
        with self.graph.as_default():
            with tf.variable_scope(NNModel._placeholder_prefix):
                self.placeholders = self.default_placeholders()
                self.placeholders.extend(self._placeholders())

            # Builds the main model.
            self._model()

            with tf.variable_scope('train'):
                minimize_op = self._minimize_op()
                assert isinstance(minimize_op, tf.Operation)

                # EMA apply ops.
                update_ops = self.graph.get_collection(tf.GraphKeys.UPDATE_OPS)
                with tf.control_dependencies([minimize_op]):
                    # Accessed by self['train/step$']
                    tf.group(*update_ops, name='step')

            self.saver = tf.train.Saver(
                name='saver',
                max_to_keep=10,
                keep_checkpoint_every_n_hours=0.5,
            )

        self.initialize()

    @_cached
    def default_placeholders(self) -> typing.List[tf.Tensor]:
        """
        Default placeholders required for all models. They can be accessed
        through `get` or `__getitem__`.

        For example:
        ::
            x = self.get('placeholder/learning_rate')
            x = self['placeholder/learning_rate']

            # Equivalent if no other names contain 'learning_rate'.
            x = self['learning_rate']

        :return: List of placeholder tensors.
        """
        return [
            tf.placeholder(tf.float32, name='learning_rate'),
        ]

    @abc.abstractmethod
    def _placeholders(self) -> nn.types.Tensors:
        """
        Placeholders defined by the user.
        Side effect: Populates self.graph.

        :return: List of placeholder tensors.
        """
        return

    @abc.abstractmethod
    def _model(self):
        """
        TensorFlow model defined by the user. Return value is ignored.
        Side effect: Populates self.graph.
        """
        return

    @abc.abstractmethod
    def _minimize_op(self) -> tf.Operation:
        """
        Loss minimization operation defined by the user.
        Side effect: Populates self.graph.

        For example:
        ::
            def _minimize_op(self):
                loss = tf.reduce_mean((self._out - self['target']) ** 2, name='loss')
                train_op = tf.train.AdamOptimizer(self['learning_rate']).minimize(loss)
                return train_op

        :return: A TensorFlow Operation.
        """
        return

    def initialize(self):
        """
        Initializes all TensorFlow variables. Not needed when restoring from a file.
        """
        if self.needs_initialization:
            with self.graph.as_default():
                self.session.run(tf.initialize_all_variables())
                self.needs_initialization = False

    @ensure.ensure_annotations
    def restore(self, restore_path: str):
        """
        Restores a previously saved model.

        :param restore_path: The path used to save the model.
        """
        with self.graph.as_default():
            with self.session.as_default():
                self.saver = tf.train.import_meta_graph(restore_path + self._meta_graph_suffix)
                self.saver.restore(tf.get_default_session(), restore_path)
                log.info("Restored model from %s", restore_path)
                self.needs_initialization = False

    @ensure.ensure_annotations
    def save(self, save_path: str):
        """
        Saves variables to a file.

        :param save_path: Path to the checkpoint file.
        """
        save_path = path.expanduser(save_path)

        assert not path.isdir(save_path) and not save_path.endswith('/'), 'save_path must be a file: {}'.format(save_path)

        dirpath = path.dirname(save_path)
        if not path.isdir(dirpath):
            log.info('mkdir %s', dirpath)
            os.makedirs(dirpath)

        with self.graph.as_default():
            self.saver.export_meta_graph(save_path + self._meta_graph_suffix, as_text=True)
            save_path_out = self.saver.save(self.session, save_path, write_meta_graph=False)
            log.info("Model saved to file: %s" % save_path_out)

    @ensure.ensure_annotations
    def train(self, feed_dict: dict):
        """
        Runs a training step.

        :param feed_dict: A dictionary that maps graph elements to values. Keys can be regular expressions
        or placeholder objects.
        """
        self.eval([], feed_dict=feed_dict, is_training=True)

    @ensure.ensure_annotations
    def eval(self, tensors_or_patterns: typing.Sequence, feed_dict: dict, is_training: bool = False) -> dict:
        """
        Evaluates TensorFlow Operations.

        :param tensors_or_patterns: Similar to the `fetches` argument of `tf.Session.run`. This can
        also be regex patterns. Must be a list.
        :param feed_dict: A dictionary that maps graph elements to values. Keys can be regular expressions
        or placeholder objects.
        :param is_training: If true, executes a training step.
        :return: A dictionary that maps `tensors_or_patterns` to evaluated values.
        """
        assert not self.needs_initialization, 'Variables are not initialized.'

        names = []
        new_feed_dict = {}
        for k, v in feed_dict.items():
            if isinstance(k, str):
                new_feed_dict[self[k]] = v
                names.append(self[k].name)
            elif isinstance(k, tf.Tensor):
                new_feed_dict[k] = v
                names.append(k.name)
            else:
                raise ValueError('Unexpected key in feed_dict: {}'.format(k))

        fetches = [self[pattern] for pattern in tensors_or_patterns]
        if is_training:
            assert self['learning_rate'].name in names, \
                'learning_rate should be in feed_dict when is_training is True.'

            # There is a race condition here, but it does not matter in most use cases.
            fetches.append(self['train/step$'])

        with self.graph.as_default():
            out_eval = self.session.run(fetches, new_feed_dict)

        results = {}
        for name, result in zip(tensors_or_patterns, out_eval):
            if result is not None:
                results[name] = result
        return results

    def __getitem__(self, pattern: str) -> nn.types.Value:
        """
        Same as `get`. Returns a Variable or Tensor whose name uniquely matches the pattern.
        :param pattern: A regular expression pattern.
        :return: Matched variable or tensor.
        """
        return self.get(pattern)

    @property
    def variables(self) -> nn.types.Variables:
        """
        Returns all TensorFlow Variables defined in the graph.

        :return: All variables in the graph.
        """
        return self.graph.get_collection(tf.GraphKeys.VARIABLES)

    @property
    def tensors(self) -> nn.types.Tensors:
        """
        Returns the output values of TensorFlow Operations defined in the graph.

        :return: Output values of all operations in the graph.
        """
        out = []
        for op in self.graph.get_operations():
            out.extend(op.outputs)
        return out

    @property
    def ops(self) -> nn.types.Operations:
        """
        Returns all TensorFlow Operations in the graph that do not have an output value.

        :return: Operations that do not have an output value.
        """
        out = []
        for op in self.graph.get_operations():
            if len(op.outputs) == 0:
                out.append(op)
        return out

    @property
    def all_values(self) -> nn.types.Values:
        """
        Returns all TensorFlow Variables or Operations defined in the graph.

        :return: All variables and operations in the graph.
        """
        values = self.tensors + self.variables + self.ops
        unique = {}
        for tensor in values:
            unique[tensor.name] = tensor
        return list(unique.values())

    @_cached
    def count(self, pattern=None) -> int:
        """
        Returns the numbers of values whose name matches the pattern.

        :param pattern: A regular expression pattern.
        :return: Number of matching values.
        """
        return len(match_names(self.all_values, pattern))

    @_cached
    def collection(self, collection: str) -> nn.types.Values:
        """
        Retrieves values by a collection key.

        :param collection: A graph collection key. Must exist.
        :return: All values in the collection.
        """
        assert collection in self.graph.get_all_collection_keys()
        return self.graph.get_collection(collection)

    @_cached
    def get(self, pattern: str, prefix=None, suffix=None) -> nn.types.Value:
        """
        Returns a variable or tensor whose name uniquely matches the pattern. If `pattern` is not
        found, tries again with `pattern+':0$'`. Same as `self[pattern]`.

        :param pattern: A regular expression pattern.
        :param prefix: Prefix for `pattern`. Defaults to ``.*?``.
        :param suffix: Suffix for `pattern`. Defaults to ``.*?``.
        :return: Matched variable or tensor.
        :raises ValueError: If pattern matches more than one item.
        """
        matched = match_names(self.all_values, pattern, prefix=prefix, suffix=suffix)
        if len(matched) != 1:
            try:
                assert not pattern.endswith(r':0$')
                return self.get(pattern + r':0$', prefix=prefix, suffix='')
            except Exception as ex:
                raise ValueError('len(matched) = {}\npattern: {} {} {}\nmatched:\n{}'.format(
                    len(matched), prefix, pattern, suffix, '\n'.join([item.name for item in matched])))
        return matched[0]

    @_cached
    def sorted_values(self, pattern=None) -> nn.types.Tensors:
        """
        Topologically sorted Tensors.

        :param pattern: A regular expression pattern. Defaults to ``.*?``.
        :return: List of Tensors in topological order.
        """
        assert not self.needs_initialization

        names = sort_tensors(match_names(self.ops, pattern))
        return [self.get(name, prefix='^', suffix='$') for name in names]

    @_cached
    def last(self, pattern=None):
        """
        Last item in `self.sorted_values`. Guaranteed to not have any values that depend on it.
        There may be more than one such value. Returns only one of them.

        :param pattern: A regular expression pattern. Defaults to ``.*?``.
        :return: Last item in `self.sorted_values`.
        """
        assert not self.needs_initialization

        values = self.sorted_values(pattern=pattern)
        assert len(values) > 0
        return values[-1]


class RestoredNNModel(NNModel):
    """
    Container for models restored from file. This can only be instantiated from `NNModel.from_file`.
    """
    def _model(self):
        # Restored models should not try to build a new graph.
        raise NotImplementedError()

    def _minimize_op(self):
        raise NotImplementedError()

    def _placeholders(self):
        raise NotImplementedError()
