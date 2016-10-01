import collections
import time
import abc
import importlib
import sys
import os
import threading
import socket
import multiprocessing as mp
from dshin import log
from dshin.nn import utils
import tensorflow as tf
import psutil
from tensorflow.python.client import device_lib
import dshin


def get_local_cluster_spec(num_processes):
    assert isinstance(num_processes, dict)
    jobs = collections.defaultdict(list)
    sockets = []
    for job_name, num in num_processes.items():
        for i in range(num):
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sockets.append(s)
            s.bind(('localhost', 0))
            addr, port = s.getsockname()
            host = '{}:{}'.format(addr, port)
            jobs[job_name].append(host)
            log.info('Assigned {} to /job:{}/task:{}'.format(host, job_name, i))
    for s in sockets:
        s.close()
    return tf.train.ClusterSpec(jobs)


def job_info_from_server_def(server_def):
    for job in server_def.cluster.job:
        if job.name == server_def.job_name:
            task_hostnames = list(zip(*sorted([(k, v) for k, v in job.tasks.items()])))[1]
            return job.name, task_hostnames
    raise RuntimeError('Unable to parse job info.')


class TFProcess(mp.Process, metaclass=abc.ABCMeta):
    def __init__(self, cluster_spec, job_name, task_id, nnmodel_class, log_dir, session_config=None, gpu_ids=None):
        super().__init__()
        self.daemon = True

        assert job_name in ('ps', 'worker', 'data')

        assert isinstance(cluster_spec, tf.train.ClusterSpec)

        self._job_name = job_name
        self._task_id = task_id
        self._cluster_spec = cluster_spec
        self._gpu_ids = gpu_ids
        self._nnmodel_class = nnmodel_class
        self._log_dir = log_dir

        if session_config is None:
            session_config = utils.default_sess_config(log_device_placement=False, mem=0.01)

        self._session_config = session_config
        self._thread = threading.Thread(target=self._thread_main, daemon=True)

        self._address = cluster_spec.task_address(self._job_name, self._task_id)

        self.log('Initialized process.')

    def setup_visible_devices(self):
        if self._gpu_ids is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(self._gpu_ids)
            importlib.reload(tf)
            importlib.reload(device_lib)

        if 'CUDA_VISIBLE_DEVICES' in os.environ:
            self.log('CUDA_VISIBLE_DEVICES="%s"', os.environ['CUDA_VISIBLE_DEVICES'])

        devices = collections.defaultdict(list)
        for device_attribute in device_lib.list_local_devices():
            devices[device_attribute.device_type].append('{}  {}'.format(device_attribute.name, device_attribute.physical_device_desc).strip())
        ret = []
        for device_type, devices in devices.items():
            ret.append((device_type, sorted(devices)))
        return sorted(ret)

    def new_server(self, start=True) -> tf.train.Server:
        server = tf.train.Server(self._cluster_spec,
                                 job_name=self._job_name,
                                 task_index=self._task_id,
                                 config=self._session_config,
                                 start=start)
        return server

    def log(self, msg, *args):
        msg = '[{job_name}:{task_id} {pid} {address}] {msg}'.format(
            job_name=self._job_name,
            task_id=self._task_id,
            address=self._address,
            pid=self.pid if self.pid is not None else '',
            msg=msg,
        )
        log.info(msg, *args)

    def _thread_main(self):
        return

    @abc.abstractmethod
    def _main(self, server: tf.train.Server, net):
        return

    def run(self):
        device_info = self.setup_visible_devices()

        self.log('Visible devices %s', device_info)
        self._thread.start()

        server = self.new_server()
        if self._nnmodel_class is not None:
            net = self._nnmodel_class(log_dir=self._log_dir)
        else:
            net = None
        self._main(server, net)

        if self._thread.is_alive():
            self.log('Waiting for helper thread to finish.')
        self._thread.join()


class ParameterServer(TFProcess):
    def __init__(self, cluster_spec, task_id, log_dir=None, session_config=None, gpu_ids=None):
        if gpu_ids is None and 'CUDA_VISIBLE_DEVICES' not in os.environ:
            gpu_ids = ()
        super().__init__(
            cluster_spec=cluster_spec,
            job_name='ps',
            task_id=task_id,
            nnmodel_class=None,
            log_dir=log_dir,
            session_config=session_config,
            gpu_ids=gpu_ids,
        )

    def _main(self, server: tf.train.Server, net):
        server.join()
