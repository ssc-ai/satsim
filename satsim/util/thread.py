from __future__ import division, print_function, absolute_import

import threading
from queue import Queue
import itertools
import traceback
import logging

logger = logging.getLogger(__name__)


class MultithreadedTaskQueue():
    """Class to run functions round robin."""

    def __init__(self, num_threads=5):
        self.num_threads = num_threads
        self.task_queues = [ThreadedTaskQueue() for i in range(num_threads)]
        self.iter = itertools.cycle(self.task_queues)

    def task(self, f, kwargs, tag=None):
        """Adds a task to the next thread in the round robin queue.

        Args:
            f: `function`. the function to run
            kwargs: `dict`, dictionary containing keyword arguments
        """
        next(self.iter).task(f, kwargs, tag)

    def has_tag(self, tag):
        for tq in self.task_queues:
            if tq.has_tag(tag):
                return True
        return False

    def stop(self):
        """Call stop on all threads."""
        for tq in self.task_queues:
            tq.stop()

    def waitUntilEmpty(self):
        """Blocks until all threads are done."""
        for tq in self.task_queues:
            tq.waitUntilEmpty()


class ThreadedTaskQueue():
    """Class to run functions as threads in a queue."""

    def __init__(self):
        self.tags = {}
        self.is_busy = False
        self.queue = Queue()
        self.t = threading.Thread(target=self._run)
        self.t.daemon = True
        self.t.start()

    def task(self, f, kwargs, tag=None):
        """Adds a task to this thread.

        Args:
            f: `function`. the function to run
            kwargs: `dict`, dictionary containing keyword arguments
        """
        if tag is not None:
            if tag not in self.tags:
                self.tags[tag] = 1
            else:
                self.tags[tag] = self.tags[tag] + 1

        self.queue.put({
            'f': f,
            'kwargs': kwargs,
            'tag': tag
        })

    def has_tag(self, tag):
        if tag in self.tags and self.tags[tag] > 0:
            return True
        else:
            return False

    def stop(self):
        """Stop this thread after current contents in queue are cleared."""
        self.queue.put('ThreadedTaskQueue.STOP')

    def busy(self):
        """Returns `True` if queue is not empty or if a task is running."""
        return self.queue.full() or self.is_busy

    def waitUntilEmpty(self):
        """Blocks until last task is finished."""
        self.queue.join()

    def _run(self):
        """Thread runner. This should not be called directly by user."""
        while True:
            next_task = self.queue.get()

            if next_task == 'ThreadedTaskQueue.STOP':
                self.queue.task_done()
                break

            self.is_busy = True
            try:
                next_task['f'](**next_task['kwargs'])
                next_tag = next_task['tag']
                if next_tag is not None:
                    self.tags[next_tag] = self.tags[next_tag] - 1
            except Exception as ex:
                logger.error('Exception in thread. {}'.format(ex))
                traceback.print_exc()

            self.is_busy = False
            self.queue.task_done()
