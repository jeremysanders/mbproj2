"""Module to run a forked parallel process."""

from __future__ import division, print_function

import os
import socket
import struct
import select
import signal

try:
    import cPickle as pickle
except ImportError:
    import pickle

# special exit code to break out of child
exitcode = b'*[EX!T}*FORK'

# type used to send object size
sizesize = struct.calcsize('L')

def recvLen(sock, length):
    """Receive exactly length bytes."""
    retn = b''
    while len(retn) < length:
        retn += sock.recv(length-len(retn))
    return retn

def sendItem(sock, item):
    """Pickle and send item using size + pickled protocol."""
    pickled = pickle.dumps(item)
    size = struct.pack('L', len(pickled))
    sock.sendall(size + pickled)

def recvItem(sock):
    """Receive a pickled item."""
    packsize = struct.unpack('L', recvLen(sock, sizesize))[0]
    pack = recvLen(sock, packsize)
    return pickle.loads(pack)

class ForkBase:
    """Base class for forking workers."""

    def __init__(self, func):
        self.func = func
        self.sock = None
        self.amparent = False

    def childLoop(self):
        """Wait for commands on the socket and execute."""

        if self.amparent:
            raise RuntimeError('Not child, or not started')

        # ignore ctrl+c
        signal.signal(signal.SIGINT, signal.SIG_IGN)

        # repeat until exit code or socket breaks
        try:
            while True:
                # get data to process
                args = recvItem(self.sock)

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
                sendItem(self.sock, retn)

        except socket.error:
            #print('Breaking on socket error')
            pass

        #print('Exiting child')
        os._exit(os.EX_OK)

class ForkParallel(ForkBase):
    """Execute function in remote forked process."""

    def __init__(self, func):
        """Parallel forked runner for running func."""

        ForkBase.__init__(self, func)
        self.running = False

        # sockets communicate between forked processes
        parentsock, childsock = socket.socketpair()

        pid = os.fork()
        self.amparent = pid != 0

        if self.amparent:
            self.sock = parentsock
            childsock.close()
        else:
            self.sock = childsock
            parentsock.close()
            self.childLoop()

    def __del__(self):
        """Tell child to close and close sockets."""

        if self.sock is not None:
            try:
                if self.amparent:
                    sendItem(self.sock, exitcode)
                self.sock.close()
            except socket.error:
                pass

    def send(self, args):
        """Send data to be processed."""

        if not self.amparent:
            raise RuntimeError('Not parent, or not started')
        if self.running:
            raise RuntimeError('Remote process is still executing')

        self.running = True
        sendItem(self.sock, args)

    def query(self, timeout=0):
        """Return isdone,result from remote process."""

        if not self.amparent:
            raise RuntimeError('Not parent, or not started')
        if not self.running:
            raise RuntimeError('Remote process is already done')

        readsock, writesock, errsock = select.select([self.sock], [], [], timeout)
        if readsock:
            retn = recvItem(self.sock)
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

class ForkQueue(ForkBase):
    """Execute function in multiple forked processes."""

    def __init__(self, func, instances, initfunc=None):
        """Initialise queue for func and with number of instances given.

        if initfunc is set, run this at first
        """

        ForkBase.__init__(self, func)

        self.socks = []

        for i in xrange(instances):
            parentsock, childsock = socket.socketpair()

            if os.fork() == 0:
                # child process
                parentsock.close()
                self.sock = childsock
                self.amparent = False

                # close other children (we don't need to talk to them)
                del self.socks

                # call the initialise function, if required
                if initfunc is not None:
                    initfunc()

                # wait for commands from parent
                self.childLoop()

                # return here, or we get a fork bomb!
                return

            else:
                # parent process - keep track of children
                self.socks.append(parentsock)
                childsock.close()

        self.amparent = True

    def __del__(self):
        """Close child forks and close sockets."""
        if self.amparent:
            for sock in self.socks:
                try:
                    sendItem(sock, exitcode)
                    sock.close()
                except socket.error:
                    pass
        else:
            try:
                self.sock.close()
            except socket.error:
                pass

    def execute(self, argslist):
        """Execute the list of items on the queue."""

        if not self.amparent:
            raise RuntimeError('Not parent, or not started')

        results = [None]*len(argslist)
        idxs = {}

        sockfree = set(self.socks)
        sockbusy = set()

        def checkbusy():
            # check results of any busy sockets
            readsock, writesock, errsock = select.select(list(sockbusy), [], [])
            for sock in readsock:
                # get result and save
                res = recvItem(sock)
                if isinstance(res, Exception):
                    raise res
                results[idxs[sock]] = res
                # move from busy to free
                sockfree.add(sock)
                sockbusy.remove(sock)

        # iterate over input values
        for i, args in enumerate(argslist):
            while not sockfree:
                checkbusy()

            sock = sockfree.pop()
            sockbusy.add(sock)
            idxs[sock] = i
            sendItem(sock, args)

        # finish remaining jobs
        while sockbusy:
            checkbusy()

        return results
