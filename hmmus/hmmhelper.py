import os

import numpy as np

g_nbytes_per_float = 8

def get_nstates(distn_name, trans_name):
    statinfo = os.stat(distn_name)
    nstates_distn, err_a = divmod(statinfo.st_size, g_nbytes_per_float)
    statinfo = os.stat(trans_name)
    nstates_squared, err_b = divmod(statinfo.st_size, g_nbytes_per_float)
    if err_a or err_b:
        raise ValueError('input file sizes should be multiples of 8 bytes')
    if nstates_distn*nstates_distn != nstates_squared:
        msg_a = 'the transition matrix should have the square '
        msg_b = 'of the number of elements in the initial distribution vector'
        raise ValueError(msg_a + msg_b)
    return nstates_distn

def load_distn_and_trans(distn_name, trans_name):
    with open(distn_name, 'rb') as fin:
        np_distn = np.fromfile(fin, dtype=float)
        nstates = len(np_distn)
    with open(trans_name, 'rb') as fin:
        np_trans = np.fromfile(fin, dtype=float).reshape((nstates, nstates))
    return np_distn, np_trans
