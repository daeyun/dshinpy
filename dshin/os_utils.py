import os
import sys
import signal
import psutil
import atexit
import time
import contextlib


def kill_child_processes():
    try:
        pid = os.getpid()
        p = psutil.Process(pid)
        for i in range(3):
            children = p.children(recursive=True)
            if len(children) == 0:
                break
            if i < 2:
                for process in children:
                    process.send_signal(signal.SIGTERM)
            else:
                time.sleep(1)
                for process in children:
                    try:
                        process.send_signal(signal.SIGKILL)
                        print('Sent SIGKILL to {}'.format(process.pid), file=sys.stderr)
                    except psutil.NoSuchProcess:
                        pass

        sys.exit()
    except Exception as ex:
        print(ex, file=sys.stderr)


@contextlib.contextmanager
def killpg_on_exit(sig=signal.SIGTERM):
    os.setpgrp()
    atexit.register(kill_child_processes)
    try:
        yield

    except KeyboardInterrupt:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        print('Received SIGINT ({})'.format(os.getpid()))
        os.killpg(os.getpgid(os.getpid()), sig)

    except SystemExit:
        os.killpg(os.getpgid(os.getpid()), sig)
