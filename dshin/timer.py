import time
import collections


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
        out = len(self.wall_times) / sum(self.wall_times), len(self.cpu_times) / sum(self.cpu_times)
        if as_string:
            out = 'wall {0:.3f}, cpu {1:.3f} ops/sec'.format(*out)
        return out


if __name__ == '__main__':
    timer = OpsPerSecond(3)
    time.sleep(0.2)

    for i in range(31):
        time.sleep(0.05)
        timer.toc()
