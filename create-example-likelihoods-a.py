#!/usr/bin/env python

"""
Create some example files.
This includes
a file of likelihoods,
a file of transition probabilities,
and a file of stationary distribution probabilities.
"""

import argparse
import numpy as np


def get_example_rolls():
    """
    This is a helper function for testing.
    See figure 3.5 in the eighth printing of
    Biological Sequence Analysis.
    Lines alternate between rolls and estimates.
    Each roll is a die roll in (1, 2, 3, 4, 5, 6).
    Each estimate is a 'Fair' vs. 'Loaded' estimate.
    @return: (300 observations, 300 viterbi estimates)
    """
    lines = [
            '315116246446644245311321631164',
            'FFFFFFFFFFFFFFFFFFFFFFFFFFFFFF',
            '152133625144543631656626566666',
            'FFFFFFFFFFFFFFFFFFLLLLLLLLLLLL',
            '651166453132651245636664631636',
            'LLLLLLFFFFFFFFFFFFLLLLLLLLLLLL',
            '663162326455236266666625151631',
            'LLLLLLLLLLLLLLLLLLLLLLFFFFFFFF',
            '222555441666566563564324364131',
            'FFFFFFFFFFFFFFFFFFFFFFFFFFFFFF',
            '513465146353411126414626253356',
            'FFFFFFFFFFFFFFFFFFFFFFFFFFFFFL',
            '366163666466232534413661661163',
            'LLLLLLLLLLLLFFFFFFFFFFFFFFFFFF',
            '252562462255265252266435353336',
            'FFFFFFFFFFFFFFFFFFFFFFFFFFFFFF',
            '233121625364414432335163243633',
            'FFFFFFFFFFFFFFFFFFFFFFFFFFFFFF',
            '665562466662632666612355245242',
            'LLLLLLLLLLLLLLLLLLLFFFFFFFFFFF']
    observation_lines = [line for i, line in enumerate(lines) if i%2 == 0]
    estimate_lines = [line for i, line in enumerate(lines) if i%2 == 1]
    observations = [int(x) for x in ''.join(observation_lines)]
    estimates = [x for x in ''.join(estimate_lines)]
    return observations, estimates

def get_distribution():
    p_fair = 2.0 / 3.0
    p_loaded = 1.0 / 3.0
    return np.array([p_fair, p_loaded])

def get_transitions():
    return np.array([[0.95, 0.05], [0.1, 0.9]])

def get_likelihoods():
    observations, estimates = get_example_rolls()
    fair = [None, 1/6.0, 1/6.0, 1/6.0, 1/6.0, 1/6.0, 1/6.0]
    loaded = [None, 0.1, 0.1, 0.1, 0.1, 0.1, 0.5]
    distns = (fair, loaded)
    for distn in distns:
        assert len(distn) == 7
        assert abs(1.0 - sum(distn[1:])) < 1e-10
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
