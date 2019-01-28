import time
import collections
import contextlib
import ntplib
from dshin import log


def ntp_timestamp() -> float:
    """
    Returns seconds since the epoch independent of the system time.
    Takes around 30~200ms. Should not be called too frequently.
    :return: A timestamp from the NTP server.
    """
    call = ntplib.NTPClient()
    response = call.request('pool.ntp.org', version=3)
    return response.tx_time


class OpsPerSecond:
    def __init__(self, n):
        self.counter = 0
        self.wall_times = collections.deque(maxlen=n)
        self.cpu_times = collections.deque(maxlen=n)
        self.n = n
        self.tic()

    def tic(self):
        self.prev_wall_time = time.time()
        self.prev_cpu_time = time.clock()

    def toc(self):
        self.wall_times.append(time.time() - self.prev_wall_time)
        self.cpu_times.append(time.clock() - self.prev_cpu_time)
        self.tic()
        self.counter += 1
        return self.counter

    def times(self, as_string=False):
        out = len(self.wall_times) / sum(self.wall_times), len(
            self.cpu_times) / sum(self.cpu_times)
        if as_string:
            out = 'wall {0:.3f}, cpu {1:.3f} ops/sec'.format(*out)
        return out


@contextlib.contextmanager
def time_elapsed(msg=None):
    start_time = time.time()

    yield

    if msg is None:
        log.info("Time elapsed: %f", time.time() - start_time)
    else:
        log.info("%s - time elapsed: %f", msg, time.time() - start_time)


if __name__ == '__main__':
    timer = OpsPerSecond(3)
    time.sleep(0.2)

    for i in range(31):
        time.sleep(0.05)
        timer.toc()
