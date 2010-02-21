"""
Create some example files.
This includes
a file of likelihoods,
a file of transition probabilities,
and a file of stationary distribution probabilities.
http://www.cs.cmu.edu/~nasmith/papers/smith.tut04a.pdf
"""

import struct
import sys

import argparse


def write_distribution(fout):
    p_1 = 1.0
    p_2 = 0
    p_x = 0
    for p in (p_1, p_2, p_x):
        bstr = struct.pack('d', p)
        fout.write(bstr)

def write_transitions(fout):
    transition_matrix = [
            [1/8.0, 3/4.0, 1/8.0],
            [1/2.0, 1/4.0, 1/4.0],
            [0, 0, 1.0]]
    for row in transition_matrix:
        for p in row:
            bstr = struct.pack('d', p)
            fout.write(bstr)

def write_likelihoods(fout):
    observations = 'XXXXx'
    d_1 = {'X':7/8.0, 'Y':1/8.0, 'x':0.00}
    d_2 = {'X':1/16.0, 'Y':15/16.0, 'x':0.00}
    d_x = {'X':0.00, 'Y':0.00, 'x':1.00}
    distns = (d_1, d_2, d_x)
    for distn in distns:
        assert abs(1.0 - sum(distn.values())) < 1e-10
    for obs in observations:
        for distn in distns:
            # get the python float
            value = distn[obs]
            # put the float into a binary form
            bstr = struct.pack('d', value)
            # write the binary value
            fout.write(bstr)

def main(args):
    with open(args.likelihoods_out, 'wb') as fout:
        write_likelihoods(fout)
    with open(args.transitions_out, 'wb') as fout:
        write_transitions(fout)
    with open(args.distribution_out, 'wb') as fout:
        write_distribution(fout)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--likelihoods_out', default='likelihoods.bin',
            help='likelihoods are written here')
    parser.add_argument('--transitions_out', default='transitions.bin',
            help='the transition matrix is written here')
    parser.add_argument('--distribution_out', default='distribution.bin',
            help='the initial distribution is written here')
    args = parser.parse_args()
    main(args)
