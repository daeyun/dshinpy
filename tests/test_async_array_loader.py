import concurrent.futures
import time
import psutil
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


def sleep_long(filename):
    for _ in range(100):
        time.sleep(1)


def wait_until_child_processes_terminate(timeout=2):
    # Wait until the number of child processes is 0.
    # This assumes that only one test runs at a time.
    interval = 0.1
    assert timeout > interval
    for i in range(int(timeout / interval + 0.5)):
        time.sleep(interval)
        if len(psutil.Process().children(recursive=True)) == 0:
            return
    pytest.fail('Timed out waiting for child processes to terminate.')


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


def test_child_process_termination(npz_files):
    wait_until_child_processes_terminate(2)

    futures = []
    # Spawns 5*20 processes total.
    # They should terminate when AsyncArrayLoader is garbage collected.
    for i in range(20):
        future = (data.AsyncArrayLoader(pool_size=5, loader_func=load_npz_sleep)
                  .join_arrays_async(npz_files))
        futures.append(future)

    # Blocks until all futures are ready.
    for future in futures:
        assert future.exception() is None

    wait_until_child_processes_terminate(2)

    assert len(psutil.Process().children(recursive=True)) == 0


def test_shutdown():
    wait_until_child_processes_terminate(2)

    loaders = [data.AsyncArrayLoader(pool_size=5, loader_func=sleep_long) for _ in range(20)]

    start_time = time.time()

    for loader in loaders:
        loader.shutdown()

    elapsed = time.time() - start_time
    assert elapsed < 2

    wait_until_child_processes_terminate(2)

    assert len(psutil.Process().children(recursive=True)) == 0
