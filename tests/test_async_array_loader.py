import concurrent.futures
import time
from os import path

import numpy as np
import pytest

from dshin import data


@pytest.fixture('function')
def npz_files(tmpdir):
    files = []
    for i in range(3):
        filename = path.join(str(tmpdir), "{}.npz".format(i))
        arr = np.array([i, i])
        np.savez_compressed(filename, data=arr)
        assert path.isfile(filename)
        files.append(filename)
    return files


def load_npz_sleep(filename):
    time.sleep(0.5)
    return data._load_npz(filename)


def test_async_loader_values(npz_files):
    loader = data.AsyncArrayLoader(pool_size=4, loader_func=data._load_npz)
    results = [loader.join_arrays_async(npz_files) for _ in range(5)]

    for result in results:
        assert isinstance(result, concurrent.futures.Future)
        assert result.exception() is None
        assert isinstance(result.result(), np.ndarray)
        assert result.result().shape == (3, 2)

        for i, item in enumerate(result.result()):
            assert np.all(item == i)


def test_blocking_loader(npz_files):
    loader = data.AsyncArrayLoader(pool_size=4, loader_func=data._load_npz)
    result = loader.join_arrays(npz_files)

    assert isinstance(result, np.ndarray)
    assert result.shape == (3, 2)


def test_async_loader_concurrency(npz_files):
    # `load_npz_sleep` sleeps for 0.5 seconds before loading the npz file.
    loader = data.AsyncArrayLoader(pool_size=15, loader_func=load_npz_sleep)

    # 5 requests, 3 files each.
    start_time = time.time()
    results = [loader.join_arrays_async(npz_files) for _ in range(5)]

    for result in results:
        assert result.exception() is None
        assert not result.running()

    elapsed = time.time() - start_time
    assert elapsed < 0.6
