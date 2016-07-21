"""
Helpers for querying and loading data.
"""

import concurrent.futures
import multiprocessing as mp
import typing
from os import path

import ensure
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


class AsyncArrayLoader(object):
    @ensure.ensure_annotations
    def __init__(self, pool_size=16, loader_func: typing.Callable = _load_npz):
        """
        An object that asynchronously loads arrays from files and returns a concatenated array.

        :param pool_size: Number of workers.
        :param loader_func: A function that takes a filename and returns an array.
        """
        self._pool_size = pool_size
        self._loader_func = loader_func
        self._process_pool = mp.Pool(self._pool_size)
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
    @ensure.ensure_annotations
    def __init__(self, query: peewee.SelectQuery):
        """
        Wrapper for reading Peewee ORM query results in batches.

        :param query: peewee.SelectQuery returning dicts or Models.
        """
        self.query = query.clone()
        self.count = query.count()
        self.current_row = 0
        self.iterations = 0
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
        self.iterations += 1
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
