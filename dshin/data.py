"""
Helpers for querying and loading data.
"""

import concurrent.futures
import multiprocessing as mp
import typing
import signal
from os import path

import ensure
import math
import time
import abc
import numpy as np
import peewee

from dshin import log


def _load_npz(filename: str, field='data') -> np.ndarray:
    """
    Loads a zipped array saved to disk.

    :param filename: Path to a ``.npz`` file.
    :param field: Name of the field used to save the array with `np.savez` or `np.savez_compressed`.
    :return: An array.
    """
    return np.load(filename)[field]


def _init_worker():
    """
    Ignore SIGINT in multiprocessing pool.
    :return:
    """
    signal.signal(signal.SIGINT, signal.SIG_IGN)


class AsyncArrayLoader(object):
    @ensure.ensure_annotations
    def __init__(self, pool_size=16, loader_func: typing.Callable = _load_npz):
        """
        Loads arrays from files concurrently and returns a concatenated array.

        :param pool_size: Number of workers.
        :param loader_func: A function that takes a filename and returns an array.
        """
        self._pool_size = pool_size
        self._loader_func = loader_func
        self._process_pool = mp.Pool(self._pool_size, initializer=_init_worker)
        self._executor = concurrent.futures.ThreadPoolExecutor(self._pool_size)

    def __del__(self):
        self._process_pool.close()
        self._executor.shutdown(wait=False)

    def shutdown(self):
        self._process_pool.terminate()
        self._executor.shutdown(wait=False)

    @ensure.ensure_annotations
    def join_arrays(self, filenames: typing.Sequence[str]) -> np.ndarray:
        """
        Uses a process pool to load arrays and concatenates them in the first dimension.
        All arrays must have the same shape. Blocks until the return value is ready.

        :param filenames: A list of paths to array files.
        :return: An array of shape (n, ...) where n = len(filenames).
        """
        out = np.array(self._process_pool.map(self._loader_func, filenames))
        assert out.flags['C_CONTIGUOUS']
        return out

    @ensure.ensure_annotations
    def join_arrays_async(self, filenames: typing.Sequence[str]) -> concurrent.futures.Future:
        """
        Executes `join_arrays` in a thread pool and returns a Future.
        :param filenames: A list of paths to array files.
        :return: A Future object returning an array of shape (n, ...) where n = len(filenames).
        """
        assert path.isfile(filenames[0])
        return self._executor.submit(self.join_arrays, filenames)


class QueryPaginator(object):
    def __init__(self, query: peewee.SelectQuery):
        """
        Wrapper for reading select query results in batches.

        :param query: peewee.SelectQuery returning dicts or Models.
        """
        assert isinstance(query, peewee.SelectQuery)

        self.query = query.clone()
        self.count = query.count()
        self.current_row = 0
        self.current_iteration = 0
        ensure.check(True)

    def _reset_iteration(self):
        """
        Initialises the row cursor. This will shuffle the rows if the query includes random order.
        """
        self.query = self.query.clone()
        count = self.query.count()
        if count != self.count:
            log.warn('Row count changed from %d to %d', self.count, count)
        self.count = count
        self.current_iteration += 1
        self.current_row = 0

    @ensure.ensure_annotations
    def next(self, max_items: int) -> typing.List[typing.Union[dict, peewee.Model]]:
        """
        Returns a list of next n items. An empty list is returned after each iteration;
        can be thought of as an iteration terminator.

        :param max_items: Number of next items to fetch.
        :return: List of dicts or Model instances each containing a row.
        """
        items = self.query[self.current_row:self.current_row + max_items]

        if len(items) == 0:
            self._reset_iteration()

        self.current_row += len(items)

        return items


class DataSource(object):
    def __init__(self, query_dict, is_seamless=True):
        """
        Manages multiple `QueryPaginator` instances and supports continuous batching.

        :param query_dict: Nested dict (key1 -> (key2 -> (... -> query))) where `query` is a `peewee.SelectQuery`.
        `key=(key1, key2, ...)` can be used to operate on the corresponding query.
        Queries will be in `.dicts()` mode (if not already set).
        """
        self.query = query_dict
        self._paginator = self._assign_paginators_recursive(self.query)
        self._is_seamless = is_seamless

    def _assign_paginators_recursive(self, query):
        if not isinstance(query, dict):
            return QueryPaginator(query.dicts())
        out = {}
        for k, v in query.items():
            out[k] = self._assign_paginators_recursive(v)
        return out

    def _keys_recursive(self, paginator_dict):
        if not isinstance(paginator_dict, dict):
            return [[paginator_dict.count]]
        ret = []
        for k, v in paginator_dict.items():
            for keypath in self._keys_recursive(v):
                ret.append([k] + keypath)
        return ret

    def keys(self):
        ret = []
        for key in self._keys_recursive(self._paginator):
            count = key[-1]
            ret.append(('/'.join(key[:-1]), count))
        return ret

    def paginator(self, key):
        """
        `QueryPaginator` corresponding to `self.query[key1][key2][...]` given `key=(key1, key2, ...)`.

        :param key: Query key. e.g. ('test', 'novelview')
        :return: The corresponding `QueryPaginator` object.
        """
        paginator = self._paginator
        assert isinstance(key, tuple) or isinstance(key, list)
        for k in key:
            if not isinstance(paginator, dict):
                raise ValueError('Invalid data key: {}'.format(key))
            paginator = paginator[k]
        assert isinstance(paginator, QueryPaginator)
        return paginator

    def count(self, key) -> int:
        """
        Total number of items for the given key.

        :param key: Query key. e.g. ['train'], ['test', 'novelview']
        :return: Total number of items.
        """
        return self.paginator(key).count

    def current_iteration_count(self, key) -> int:
        """
        Epoch number for the next batch. Initially 0.

        :param key: Query key.
        :return: Epoch and row number for the next batch.
        """
        return self.paginator(key).current_iteration

    def current_row_count(self, key) -> int:
        """
        Row number for the next batch. Initially 0.

        :param key: Query key.
        :return: Epoch and row number for the next batch.
        """
        return self.paginator(key).current_row

    def batch_count(self, batch_size: int, key: typing.Sequence[str]) -> int:
        """
        Total number of batches per iteration.
        :param key: Query key. e.g. ['train'], ['test', 'novelview']
        :return: Total number of batches.
        """
        return int(math.ceil(self.count(key=key) / batch_size))

    def next(self, key, batch_size=1):
        start_time = time.time()
        out = self._next(key=key, batch_size=batch_size, is_seamless=self._is_seamless)

        elapsed = time.time() - start_time
        if elapsed > 0.1 and self.current_row_count(key) > 1:
            log.warn('Query is taking too long: %.3f', elapsed)

        return out

    def _next(self, key, batch_size=1, is_seamless=True):
        """
        Returns the next batch. If `is_seamless` is False, returns ``None`` after each iteration.

        :param batch_size: Number of batches.
        :param key: Query key. e.g. ['train'], ['test', 'novelview']
        :return: A list of dicts. Batch size may be smaller than `batch_size` if `is_seamless` is False and
        this is the last batch of the iteration.
        """
        paginator = self.paginator(key)
        assert paginator.count > 0

        rows = paginator.next(batch_size)

        if is_seamless:
            if self.current_row_count(key=key) == self.count(key=key):
                # len==0 indicates end of one iteration.
                # This resets current_row and current_iteration.
                assert len(paginator.next(batch_size)) == 0

            if len(rows) < batch_size:
                rows.extend(paginator.next(batch_size - len(rows)))

        elif len(rows) == 0:
            # Indicates end of one iteration.
            return None

        return [self._post_process(r) for r in rows]

    def _post_process(self, row_dict):
        """
        Can be implemented by the user.

        :param row_dict: A dict that corresponds to a single row.
        :return: Same as `row_dict`, by default.
        """
        return row_dict
