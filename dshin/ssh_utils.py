from concurrent import futures
import time
from dshin import log
import multiprocessing as mp
from paramiko import ssh_exception
import numpy as np
import logging
import paramiko
from os import path


def __patch_crypto():
    from cryptography.hazmat import backends

    try:
        from cryptography.hazmat.backends.commoncrypto.backend import backend as be_cc
    except ImportError:
        be_cc = None

    try:
        from cryptography.hazmat.backends.openssl.backend import backend as be_ossl
    except ImportError:
        be_ossl = None

    backends._available_backends_list = [be for be in (be_cc, be_ossl) if be is not None]


__patch_crypto()


class RemoteCommandRunner(object):
    def __init__(self, pool_size=10, key=None, user=None, host=None, remote_python_executable='python', loglevel=logging.WARNING):
        assert pool_size > 1
        self._pool_size = pool_size
        self._ssh_pool = futures.ThreadPoolExecutor(pool_size)
        self._local_pool = futures.ThreadPoolExecutor(pool_size)
        self._key = key
        self._user = user
        self._remote_python_executable = remote_python_executable
        self._connections = {}
        self._host = host
        self._future_client = None

        # Global.
        logging.getLogger("paramiko").setLevel(loglevel)

    def _get_connection(self):
        if self._future_client is None:
            self.connect()
        assert isinstance(self._future_client, futures.Future)
        client = self._future_client.result(timeout=30)
        return client

    def disconnect(self):
        if self._future_client is None:
            return None

        try:
            self._ssh_pool.shutdown(wait=True)
        except Exception as ex:
            log.warn(ex)

        try:
            self._local_pool.shutdown(wait=True)
        except Exception as ex:
            log.warn(ex)

        try:
            self._future_client.result(timeout=10).close()
        except TimeoutError as ex:
            log.warn('Timeout')
        except RuntimeError as ex:
            log.warn('Runtime error {}'.format(ex))

        self._future_client = None
        self._ssh_pool = None
        self._local_pool = None

    def _connect(self, host, user, key):
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(host, username=user, pkey=key)
        return client

    def __del__(self):
        try:
            self.disconnect()
        except:
            pass

    def connect(self, key=None, user=None, host=None):
        self.disconnect()
        if key is not None:
            self._key = key
        if user is not None:
            self._user = user
        if host is not None:
            self._host = host

        self._ssh_pool = futures.ThreadPoolExecutor(self._pool_size)
        self._local_pool = futures.ThreadPoolExecutor(self._pool_size)

        if isinstance(self._key, str):
            pkey = paramiko.RSAKey.from_private_key_file(path.expanduser(self._key))
        else:
            pkey = self._key
        assert self._host is not None

        self._future_client = self._ssh_pool.submit(self._connect, self._host, self._user, pkey)

        assert isinstance(self._future_client, futures.Future)
        return self._future_client

    def _read_and_decode(self, stdout, stderr, stdout_only=False, assert_stderr_empty=False):
        stdout_string = stdout.read().decode('utf-8')
        if not stdout_only or assert_stderr_empty:
            stderr_string = stderr.read().decode('utf-8')
            if assert_stderr_empty and len(stderr_string.strip()) > 0:
                raise RuntimeError('stderr is not empty:\n{}\nstdout:\n{}'.format(stdout_string, stderr_string))
            if not stdout_only:
                return stdout_string, stderr_string
        assert stdout_only
        return stdout_string

    def execute_command(self, command, callback_fn=None, stdout_only=False, assert_stderr_empty=False):
        assert command

        def exec_and_decode():
            client = self._get_connection()
            stdin, stdout, stderr = client.exec_command(command)
            return self._read_and_decode(stdout, stderr, stdout_only, assert_stderr_empty)

        if self._ssh_pool is None:
            self.connect()

        future_decoded_stdout_stderr = self._ssh_pool.submit(exec_and_decode)
        if callback_fn is not None:
            future_decoded_stdout_stderr.add_done_callback(callback_fn)
        return future_decoded_stdout_stderr

    def execute_python(self, command, callback_fn=None, stdout_only=True, assert_stderr_empty=True):
        escaped = command.replace(r'"', r'\"')
        python_command = '{} -c "{}"'.format(self._remote_python_executable, escaped)
        return self.execute_command(python_command, callback_fn=callback_fn, assert_stderr_empty=assert_stderr_empty, stdout_only=stdout_only)

    def ps_grep(self, query_word):
        assert query_word
        command = 'ps aux | grep -i "[{}]{}"'.format(query_word[0], query_word[1:])
        future = self.execute_command(command=command, stdout_only=True, assert_stderr_empty=True)
        future = self._local_pool.submit(lambda out: [line for line in out.result().strip().split('\n') if line.strip()], future)
        return future

    def kill(self, query_word, sigkill=False):
        assert query_word
        assert len(query_word) > 1
        if sigkill:
            command = "kill -9 $(ps aux | grep -i '[{}]{}' | awk '{{print $2}}')".format(query_word[0], query_word[1:])
        else:
            command = "kill $(ps aux | grep -i '[{}]{}' | awk '{{print $2}}')".format(query_word[0], query_word[1:])
        future = self.execute_command(command=command, stdout_only=False, assert_stderr_empty=False)
        return future

    def utilization(self, cpu_wait_seconds=2):
        command = ''.join([
            "import psutil; print(','.join(str(item) for item in psutil.cpu_percent(percpu=True, interval=",
            str(cpu_wait_seconds), ")));mem = psutil.virtual_memory(); print('{:.3f},{:.3f}'.format(mem.available/2**20, mem.total/2**20))"])

        future = self.execute_python(command, stdout_only=True, assert_stderr_empty=True)

        def _parse_usage_stats(future):
            stdout = future.result()
            items = stdout.split('\n')
            cpu_loads = [float(item) for item in items[0].split(',')]
            mem_available, mem_total = [float(item) for item in items[1].split(',')]
            return {
                'cpu': cpu_loads,
                'mem': {
                    'available': mem_available,
                    'total': mem_total,
                }
            }

        future_stats = self._local_pool.submit(_parse_usage_stats, future)
        return future_stats


def _utilization_worker(host, key, user, python_exec, query_word, logfile, logfile_word):
    try:
        runner = RemoteCommandRunner(host=host, key=key, user=user, remote_python_executable=python_exec)
        future_ps_grep = runner.ps_grep(query_word)
        future_utilization = runner.utilization(cpu_wait_seconds=3)

        ps_grep = future_ps_grep.result()
        utilization = future_utilization.result()

        runner.disconnect()

        if len(ps_grep) == 0:
            status = 'not running'
        else:
            status = '{} processes'.format(len(ps_grep))

        mean_cpu_load = np.mean(utilization['cpu'])
        max_cpu_load = np.max(utilization['cpu'])
        mem_usage = 100 * utilization['mem']['available'] / utilization['mem']['total']

        if logfile is not None:
            count = runner.execute_command('grep -roh {logfile_word} {logfile} | wc -w'.format(logfile_word=logfile_word, logfile=logfile),
                                           stdout_only=True, assert_stderr_empty=False)
            count = int(count.result().strip())
        else:
            count = 0
    except (ssh_exception.NoValidConnectionsError, ssh_exception.AuthenticationException, TimeoutError) as ex:
        return ('no connection', 0, 0, 0, 0)

    return (status, mean_cpu_load, max_cpu_load, mem_usage, count)


def _kill_worker(query_word, host, key, user, python_exec=None, force=True):
    runner = RemoteCommandRunner(host=host, key=key, user=user, remote_python_executable=python_exec)
    future = runner.kill(query_word)

    stdout, stderr = future.result()
    killed = len(stderr.strip()) == 0

    ps_grep = runner.ps_grep(query_word)

    if killed:
        time.sleep(0.1)
        res = ps_grep.result()
        if len(res) == 0:
            status = 'ok - killed'
        else:
            if force:
                log.warn('{} still running. Sending SIGKILL.'.format(len(res)))
                time.sleep(1)

                future = runner.kill(query_word, sigkill=True)
                stdout, stderr = future.result()
                killed = len(stderr.strip()) == 0
                if killed:
                    status = 'ok - SIGKILL'
                else:
                    status = 'error - KIGKILL failed'.format(len(res))
            else:
                status = 'error - {} running'.format(len(res))
    else:
        status = 'ok - not running'

    runner.disconnect()

    return status


def server_utilization(hosts, key, user, query_word, remote_python_executable=None, num_processes=20, logfile=None, logfile_word=None):
    pool = mp.Pool(num_processes)
    results = []
    start_time = time.time()
    out = pool.starmap(_utilization_worker, [(host, key, user, remote_python_executable, query_word, logfile, logfile_word) for host in hosts[:num_processes]])
    results.extend(out)
    elapsed = time.time() - start_time
    print('{:.2g} seconds/host'.format(elapsed / num_processes))
    out = pool.starmap(_utilization_worker, [(host, key, user, remote_python_executable, query_word, logfile, logfile_word) for host in hosts[num_processes:]])
    results.extend(out)
    log.info('Closing pool.')
    pool.close()
    return results


def kill_processes(query_word, hosts, key, user, remote_python_executable=None, num_processes=20):
    pool = mp.Pool(num_processes)
    results = pool.starmap(_kill_worker, [(query_word, host, key, user, remote_python_executable) for host in hosts])
    log.info('Closing pool.')
    pool.close()
    return results
