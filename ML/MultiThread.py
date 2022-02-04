# Mutli threading with retury type from 
# https://stackoverflow.com/questions/6893968/how-to-get-the-return-value-from-a-thread-in-python
from threading import Thread

class MultiThread(Thread):
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs={}, Verbose=None):
        Thread.__init__(self, group, target, name, args, kwargs)
        self._return = None

    def run(self):
        if self._target is not None:
            self._return = self._target(*self._args,
                                                **self._kwargs)
    def Join(self, *args):
        Thread.join(self, *args)
        self._return
