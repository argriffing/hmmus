#!/usr/bin/env python

"""
Create some example files.
This includes
a file of likelihoods,
a file of transition probabilities,
and a file of stationary distribution probabilities.
ftp://selab.janelia.org/pub/publications/Eddy-ATG4/Eddy-ATG4-reprint.pdf
NATURE BIOTECHNOLOGY VOLUME 22 NUMBER 10 OCTOBER 2004
"""

import argparse
import numpy as np


def get_distribution(fout):
    p_E = 1.0
    p_5 = 0.0
    p_I = 0.0
    p_x = 0.0
    return np.array([p_E, p_5, p_I, p_x])

def get_transitions(fout):
    return np.array([
        [0.9, 0.1, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.9, 0.1],
        [0.0, 0.0, 0.0, 1.0]])

def get_likelihoods(fout):
    observations = 'CTTCATGTGAAAGCAGACGTAAGTCAx'
    d_E = {'A':0.25, 'C':0.25, 'G':0.25, 'T':0.25, 'x':0.0}
    d_5 = {'A':0.05, 'C':0.00, 'G':0.95, 'T':0.00, 'x':0.0}
    d_I = {'A':0.40, 'C':0.10, 'G':0.10, 'T':0.40, 'x':0.0}
    d_x = {'A':0.00, 'C':0.00, 'G':0.00, 'T':0.00, 'x':1.0}
    distns = (d_E, d_5, d_I, d_x)
    for distn in distns:
        assert abs(1.0 - sum(distn.values())) < 1e-10
    return np.array([[d[obs] for d in distns] for obs in observations])

def main(args):
    with open(args.likelihoods_out, 'wb') as fout:
        get_likelihoods().tofile(fout)
    with open(args.transitions_out, 'wb') as fout:
        get_transitions().tofile(fout)
    with open(args.distribution_out, 'wb') as fout:
        get_distribution().tofile(fout)

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
