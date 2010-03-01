#!/usr/bin/env python

"""Create some input files which an HMM algorithm might find useful.
Each of the three available demos creates three files.
"""

import argparse
import numpy as np


class DurbinDemo:
    """
    From the book Biological Sequence Analysis.
    """

    def get_example_rolls(self):
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

    def get_distribution(self):
        p_fair = 2.0 / 3.0
        p_loaded = 1.0 / 3.0
        return np.array([p_fair, p_loaded])

    def get_transitions(self):
        return np.array([[0.95, 0.05], [0.1, 0.9]])

    def get_likelihoods(self):
        observations, estimates = self.get_example_rolls()
        fair = [None, 1/6.0, 1/6.0, 1/6.0, 1/6.0, 1/6.0, 1/6.0]
        loaded = [None, 0.1, 0.1, 0.1, 0.1, 0.1, 0.5]
        distns = (fair, loaded)
        for distn in distns:
            assert len(distn) == 7
            assert abs(1.0 - sum(distn[1:])) < 1e-10
        return np.array([[d[obs] for d in distns] for obs in observations])


class EddyDemo:
    """
    NATURE BIOTECHNOLOGY VOLUME 22 NUMBER 10 OCTOBER 2004
    ftp://selab.janelia.org/pub/publications/Eddy-ATG4/Eddy-ATG4-reprint.pdf
    """

    def get_distribution(self):
        p_E = 1.0
        p_5 = 0.0
        p_I = 0.0
        p_x = 0.0
        return np.array([p_E, p_5, p_I, p_x])

    def get_transitions(self):
        return np.array([
            [0.9, 0.1, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.9, 0.1],
            [0.0, 0.0, 0.0, 1.0]])

    def get_likelihoods(self):
        observations = 'CTTCATGTGAAAGCAGACGTAAGTCAx'
        d_E = {'A':0.25, 'C':0.25, 'G':0.25, 'T':0.25, 'x':0.0}
        d_5 = {'A':0.05, 'C':0.00, 'G':0.95, 'T':0.00, 'x':0.0}
        d_I = {'A':0.40, 'C':0.10, 'G':0.10, 'T':0.40, 'x':0.0}
        d_x = {'A':0.00, 'C':0.00, 'G':0.00, 'T':0.00, 'x':1.0}
        distns = (d_E, d_5, d_I, d_x)
        for distn in distns:
            assert abs(1.0 - sum(distn.values())) < 1e-10
        return np.array([[d[obs] for d in distns] for obs in observations])


class SmithDemo:
    """
    http://www.cs.cmu.edu/~nasmith/papers/smith.tut04a.pdf
    """

    def get_distribution(self):
        p_1 = 1.0
        p_2 = 0.0
        p_x = 0.0
        return np.array([p_1, p_2, p_x])

    def get_transitions(self):
        return np.array([
                [1/8.0, 3/4.0, 1/8.0],
                [1/2.0, 1/4.0, 1/4.0],
                [0, 0, 1.0]])

    def get_likelihoods(self):
        observations = 'XXXXx'
        d_1 = {'X':7/8.0, 'Y':1/8.0, 'x':0.00}
        d_2 = {'X':1/16.0, 'Y':15/16.0, 'x':0.00}
        d_x = {'X':0.00, 'Y':0.00, 'x':1.00}
        distns = (d_1, d_2, d_x)
        for distn in distns:
            assert abs(1.0 - sum(distn.values())) < 1e-10
        return np.array([[d[obs] for d in distns] for obs in observations])
