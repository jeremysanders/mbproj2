"""Module to run a forked parallel process."""

from __future__ import division, print_function
import os
import socket
import struct
import select
import signal
import cPickle as pickle

exitcode = '*[EX!T}*FORK'
sizesize = struct.calcsize('L')

class ForkParallel:
    """Execute function in remote forked process."""

    def __init__(self, func):
        """Parallel forked runner for running func."""

        self.state = None
        self.sock = None
        self.running = False
        self.func = func
        self.start()

    def send(self, args):
        """Send data to be processed."""

        if self.state != 'parent':
            raise RuntimeError('Not parent, or not started')
        if self.running:
            raise RuntimeError('Remote process is still executing')

        self.running = True
        self.sendItem(args)

    def query(self, timeout=0):
        """Return isdone,result from remote process."""

        if self.state != 'parent':
            raise RuntimeError('Not parent, or not started')
        if not self.running:
            raise RuntimeError('Remote process is already done')

        readsock, writesock, errsock = select.select([self.sock], [], [], timeout)
        if readsock:
            retn = self.recvItem()
            self.running = False
            if isinstance(retn, Exception):
                raise retn
            return True, retn
        else:
            return False, None

    def wait(self):
        """Wait until a response, and return value."""

        while True:
            done, res = self.query(timeout=6000)
            if done:
                return res

    # implementation details below

    def start(self):
        """Start the remote process."""
        parentsock, childsock = socket.socketpair()

        childpid = os.fork()
        if childpid == 0:
            self.state = 'child'
            parentsock.close()
            self.sock = childsock
            self.childWait()
        else:
            self.state = 'parent'
            childsock.close()
            self.sock = parentsock

    def __del__(self):
        if self.sock is not None:
            try:
                if self.state == 'parent':
                    self.sendItem(exitcode)
                self.sock.close()
            except socket.error:
                pass

    def recvLen(self, length):
        """Receive exactly length bytes."""
        retn = b''
        while len(retn) < length:
            retn += self.sock.recv(length-len(retn))
        return retn

    def sendItem(self, item):
        """Pickle and send item using size + pickled protocol."""
        pickled = pickle.dumps(item)
        size = struct.pack('L', len(pickled))
        self.sock.sendall(size + pickled)

    def recvItem(self):
        """Receive a pickled item."""
        packsize = struct.unpack('L', self.recvLen(sizesize))[0]
        pack = self.recvLen(packsize)
        return pickle.loads(pack)

    def childWait(self):
        """Wait for commands on the socket and execute."""

        if self.state != 'child':
            raise RuntimeError('Not child, or not started')

        # ignore ctrl+c
        signal.signal(signal.SIGINT, signal.SIG_IGN)

        # repeat until exit code or socket breaks
        try:
            while True:
                # get data to process
                args = self.recvItem()

                #print('received', args)

                # exit if parent is done
                if args == exitcode:
                    break

                # presumably no socket error in below
                try:
                    # do the work
                    retn = self.func(args)
                except Exception as e:
                    # send back an exception
                    retn = e

                # send back the result
                self.sendItem(retn)

        except socket.error:
            #print('Breaking on socket error')
            pass

        #print('Exiting child')
        os._exit(os.EX_OK)
