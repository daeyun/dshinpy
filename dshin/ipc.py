import numpy as np
import struct
import socket
import time
import multiprocessing as mp
from dshin import log

class Server(object):
    def __init__(self, WorkerClass, host='127.0.0.1', port=18080, backlog=30):
        assert issubclass(WorkerClass, mp.Process)
        self.Worker = WorkerClass
        self.host, self.port, self.backlog = host, port, backlog
        self.workers = []

    def serve_forever(self):
        sock = self.start_listening(self.host, self.port, self.backlog)
        try:
            while True:
                conn, request_code = self.accept_client(sock)
                self.launch_worker(conn, request_code)
                time.sleep(0.1)
        except:
            sock.close()

    def launch_worker(self, conn: socket.socket, request_code:int):
        worker = self.Worker(self, conn, request_code)
        worker.daemon = True
        worker.start()

        self.workers.append(worker)

    @staticmethod
    def start_listening(host, port, backlog):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 2 ** 20)
        bufsize = sock.getsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF)
        log.info('Buffer size: %d', bufsize)
        sock.bind((host, port))
        sock.listen(backlog)
        log.info('Started server %s:%d. Backlog: %d', host, port, backlog)
        return sock

    @staticmethod
    def accept_client(sock):
        conn, addr = sock.accept()
        log.info('New client %s', addr)
        msg = struct.unpack('=i', conn.recv(4))[0]
        assert msg == 42  # Endianness sanity check.
        conn.send(struct.pack('=i', msg + 1))
        request_code = struct.unpack('=i', conn.recv(4))[0]
        return conn, request_code

buffer_size_bytes = 2 * (2 ** 20)

def receive_array(conn):
    data = conn.recv(4)
    if len(data) == 0:
        raise ConnectionError

    packet_size = int(buffer_size_bytes / 4)

    shape_size = struct.unpack('=I', data)[0]
    shape = struct.unpack('={}I'.format(shape_size), conn.recv(4 * shape_size))
    n = int(np.prod(shape))

    arr = np.zeros(n, dtype=np.float32)

    for i in range(0, len(arr), packet_size):
        arr_i = arr.view()[i:i + packet_size]
        data = conn.recv(4 * arr_i.size)
        arr_i[:] = struct.unpack('={}f'.format(arr_i.size), data)

    return arr.reshape(shape)

def send_array(conn, arr):
    assert isinstance(arr, np.ndarray)
    assert not np.isfortran(arr)

    packet_size = int(buffer_size_bytes / 4)
    shape = arr.shape
    arr = arr.astype('=f4').flat

    conn.send(struct.pack('=I', len(shape)))
    assert conn.send(struct.pack('={}f'.format(len(shape)), *shape)) == 4 * len(shape)

    n = len(arr)
    for i in range(0, n, packet_size):
        arr_i = arr[i:i + packet_size]
        packed = arr_i.tostring()
        conn.send(packed)
