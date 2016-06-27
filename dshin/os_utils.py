import os
import sys
import signal
import psutil
import atexit
import contextlib


def kill_children_processes():
    try:
        pid = os.getpid()
        p = psutil.Process(pid)
        for i in range(3):
            children = p.children(recursive=True)
            if len(children) == 0:
                break
            for process in children:
                process.send_signal(signal.SIGTERM)
        sys.exit()
    except:
        pass


@contextlib.contextmanager
def killpg_on_exit(sig=signal.SIGTERM):
    os.setpgrp()
    atexit.register(kill_children_processes)
    try:
        yield
    except (KeyboardInterrupt, SystemExit):
        os.killpg(os.getpgid(os.getpid()), sig)
