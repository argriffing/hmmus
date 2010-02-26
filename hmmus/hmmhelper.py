import os

g_nbytes_per_float = 8

def get_nstates(distribution_name, transitions_name):
    statinfo = os.stat(distribution_name)
    nstates_distn, err_a = divmod(statinfo.st_size, g_nbytes_per_float)
    statinfo = os.stat(transitions_name)
    nstates_squared, err_b = divmod(statinfo.st_size, g_nbytes_per_float)
    if err_a or err_b:
        raise ValueError('input file sizes should be multiples of 8 bytes')
    if nstates_distn*nstates_distn != nstates_squared:
        msg_a = 'the transition matrix should have the square '
        msg_b = 'of the number of elements in the initial distribution vector'
        raise ValueError(msg_a + msg_b)
    return nstates_distn
