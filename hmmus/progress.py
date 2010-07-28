"""
Make a progress bar.
"""

import os
import sys
import time
from array import array
from fcntl import ioctl
import termios
import signal


class Bar:

    def __init__(self, high=100):
        """
        @param high: when the progress reaches this value then we are done
        """
        if high <= 0:
            raise ValueError()
        self.high = high
        self.outfile = sys.stderr
        self.finished = False
        # get the terminal width and reset the cached nfilled value
        self.on_resize(None, None)
        signal.signal(signal.SIGWINCH, self.on_resize)
        # set the progress to zero
        self.update(0)

    def set_high(self, high):
        """
        Call this function only rarely.
        @param high: when the progress reaches this value then we are done
        """
        if high <= 0:
            raise ValueError()
        self.high = high
        if self.progress >= high:
            self.finish()
        else:
            self.update(self.progress)

    def on_resize(self, signal_number, frame):
        """
        @param signal_number: an unused parameter passed by the signal framework
        @param frame: an unused parameter passed by the signal framework
        """
        ioctl_result = ioctl(self.outfile, termios.TIOCGWINSZ, '\0'*8)
        nrows, self.ncols = array('h', ioctl_result)[:2]
        self.cached_nfilled = None

    def get_nfilled(self):
        """
        @return: the length of the filled bar
        """
        nbar_columns = self.ncols - 2
        nfilled = (self.progress * nbar_columns) / self.high
        return nfilled

    def get_progress_line(self):
        """
        @return: the progress string to print
        """
        nbar_columns = self.ncols - 2
        nfilled = self.get_nfilled()
        progress_line = '[' + '='*nfilled + '.'*(nbar_columns-nfilled) + ']'
        return progress_line

    def increment(self, increment_amount=1):
        """
        @param increment_amount: the amount of progress to be added
        """
        self.update(self.progress + increment_amount)

    def update(self, progress):
        """
        @param progress: the total amount of progress made so far
        """
        if not (0 <= progress <= self.high):
            msg = 'progress %d is not in [%d, %d]' % (progress, 0, self.high)
            raise ValueError(msg)
        if not self.finished:
            self.progress = progress
            nfilled = self.get_nfilled()
            if self.progress == self.high:
                self.cancel()
            elif self.cached_nfilled != nfilled:
                self.cached_nfilled = nfilled
                progress_line = self.get_progress_line()
                self.outfile.write(progress_line + '\r')

    def finish(self):
        self.update(self.high)

    def cancel(self):
        nfilled = self.get_nfilled()
        self.cached_nfilled = nfilled
        progress_line = self.get_progress_line()
        self.outfile.write(progress_line + '\n')
        signal.signal(signal.SIGWINCH, signal.SIG_DFL)
        self.finished = True

