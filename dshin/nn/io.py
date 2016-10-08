import ujson
from os import path

import numpy as np
import tensorflow as tf


class NumpyTFRecordIO(object):
    def __init__(self, can_overwrite=False):
        self._options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.ZLIB)
        self.can_overwrite = can_overwrite

    def write_arrays_as_tf_examples(self, filename, array_dicts):
        assert self.can_overwrite or not path.exists(filename)
        assert filename.endswith('.tfrecords')
        if isinstance(array_dicts, dict):
            array_dicts = [array_dicts]
        assert isinstance(array_dicts, (list, tuple))
        writer = tf.python_io.TFRecordWriter(filename, options=self._options)
        for item in array_dicts:
            example = self.tf_example_from_array_dict(item)
            writer.write(example.SerializeToString())
        writer.close()

    def write_tensor_attribute_json(self, filename, array_or_attribute_dict):
        assert self.can_overwrite or not path.exists(filename)
        assert isinstance(array_or_attribute_dict, dict)
        assert filename.endswith('.json')
        attribute_dict = {}
        for tensor_name, v in array_or_attribute_dict.items():
            assert isinstance(tensor_name, str)
            attr = {}
            if isinstance(v, dict):
                attr.update(v)
                if 'dtype' in attr:
                    if isinstance(attr['dtype'], type):
                        attr['dtype'] = np.dtype(attr['dtype']).name
                    assert isinstance(attr['dtype'], str)
                if 'shape' in attr:
                    assert isinstance(attr['shape'], (tuple, list))
            elif isinstance(v, np.ndarray):
                attr['dtype'] = v.dtype.name
                attr['shape'] = v.shape
            attribute_dict[tensor_name] = attr
        json_text = ujson.dumps(attribute_dict)
        with open(filename, 'w') as f:
            f.write(json_text)
        return json_text

    def read_tensor_attribute_json(self, filename):
        assert filename.endswith('.json')
        with open(filename, 'r') as f:
            return ujson.load(f)

    def read_tf_examples(self, filename):
        examples = []
        for serialized in tf.python_io.tf_record_iterator(filename, options=self._options):
            examples.append(tf.train.Example.FromString(serialized))
        return examples

    def read_tf_examples_as_arrays(self, filename, attribute_dict=None, force_full_attributes=False):
        if attribute_dict is None:
            attribute_dict = {}
        examples = self.read_tf_examples(filename)
        array_dicts = []
        for example in examples:
            array_dict = {}
            for tensor_name, data in example.features.feature.items():
                if force_full_attributes:
                    assert tensor_name in attribute_dict
                    assert 'shape' in attribute_dict[tensor_name]
                    assert 'dtype' in attribute_dict[tensor_name]
                attrs = attribute_dict[tensor_name] if tensor_name in attribute_dict else {}
                kind = data.WhichOneof('kind')
                proto_dtype = {
                    'int64_list': np.int64,
                    'float_list': np.float32,
                    'bytes_list': np.byte,
                }[kind]
                dtype = attrs.get('dtype', proto_dtype)
                values = getattr(data, kind).value
                if proto_dtype == np.byte:
                    arr = [np.fromstring(v, dtype=dtype) for v in values]
                    if len(arr) == 1:
                        arr = arr[0]
                else:
                    arr = np.array(values, dtype=dtype)
                if 'shape' in attrs:
                    assert arr.size == np.prod(attrs['shape'])
                    arr.resize(attrs['shape'])
                array_dict[tensor_name] = arr
            array_dicts.append(array_dict)
        return array_dicts

    def tf_example_from_array_dict(self, array_dict):
        feature_dict = {k: self.feature_from_ndarray(v) for k, v in array_dict.items()}
        return tf.train.Example(features=tf.train.Features(feature=feature_dict))

    def feature_from_ndarray(self, arr):
        assert isinstance(arr, np.ndarray)
        if not arr.flags['C_CONTIGUOUS']:
            arr = np.ascontiguousarray(arr)
        assert arr.flags['C_CONTIGUOUS']

        dtype = arr.dtype
        value = arr.ravel()

        if dtype == np.float32:
            ret = tf.train.Feature(float_list=tf.train.FloatList(value=value.tolist()))
        elif dtype in (np.bool, np.byte):
            ret = tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.tobytes()]))
        elif dtype == np.int64:
            ret = tf.train.Feature(int64_list=tf.train.Int64List(value=value.tolist()))
        else:
            raise NotImplementedError('Unsupported dtype. Value needs to be casted.')
        return ret
