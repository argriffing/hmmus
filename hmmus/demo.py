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

    def get_expected_scaling(self):
        """
        This is for regression testing.
        """
        return np.array([
            0.2500000000000000,
            0.2250000000000000,
            0.2250000000000000,
            0.2250000000000000,
            0.2300000000000000,
            0.2288043478260870,
            0.3112589073634204,
            0.2778657280219780,
            0.2195117068016239,
            0.2947153183144838,
            0.3031082294941561,
            0.3177687204825244,
            0.1433198888830900,
            0.1406205210251966,
            0.2843698055395475,
            0.1959737111740095,
            0.3002220700816590,
            0.1435579042788666,
            0.2327771637086132,
            0.2891297743804224,
            0.2992975648883980,
            0.3146783498106365,
            0.1477836526869613,
            0.3148659166602665,
            0.1268641738877951,
            0.2970410303325410,
            0.0625004847949679])

    def get_expected_posterior(self):
        """
        This is for regression testing.
        """
        return np.array([
            [1.000000000000, 0.000000000000, 0.000000000000, 0.000000000000],
            [1.000000000000, 0.000000000000, 0.000000000000, 0.000000000000],
            [1.000000000000, 0.000000000000, 0.000000000000, 0.000000000000],
            [1.000000000000, 0.000000000000, 0.000000000000, 0.000000000000],
            [0.998930645651, 0.001069354349, 0.000000000000, 0.000000000000],
            [0.998930645651, 0.000000000000, 0.001069354349, 0.000000000000],
            [0.967184188427, 0.031746457224, 0.001069354349, 0.000000000000],
            [0.967184188427, 0.000000000000, 0.032815811573, 0.000000000000],
            [0.917580349014, 0.049603839413, 0.032815811573, 0.000000000000],
            [0.915948643771, 0.001631705244, 0.082419650986, 0.000000000000],
            [0.914928827993, 0.001019815777, 0.084051356229, 0.000000000000],
            [0.914291443132, 0.000637384861, 0.085071172007, 0.000000000000],
            [0.884015662241, 0.030275780892, 0.085708556868, 0.000000000000],
            [0.884015662241, 0.000000000000, 0.115984337759, 0.000000000000],
            [0.881525877628, 0.002489784613, 0.115984337759, 0.000000000000],
            [0.763261108520, 0.118264769108, 0.118474122372, 0.000000000000],
            [0.759370820063, 0.003890288457, 0.236738891480, 0.000000000000],
            [0.759370820063, 0.000000000000, 0.240629179937, 0.000000000000],
            [0.297399065735, 0.461971754327, 0.240629179937, 0.000000000000],
            [0.297399065735, 0.000000000000, 0.702600934265, 0.000000000000],
            [0.287901291181, 0.009497774554, 0.702600934265, 0.000000000000],
            [0.281965182084, 0.005936109097, 0.712098708819, 0.000000000000],
            [0.000000000000, 0.281965182084, 0.718034817916, 0.000000000000],
            [0.000000000000, 0.000000000000, 1.000000000000, 0.000000000000],
            [0.000000000000, 0.000000000000, 1.000000000000, 0.000000000000],
            [0.000000000000, 0.000000000000, 1.000000000000, 0.000000000000],
            [0.000000000000, 0.000000000000, 0.000000000000, 1.000000000000]])

    def get_expected_forward(self):
        """
        This is for regression testing.
        """
        return np.array([
            [1.000000000000, 0.000000000000, 0.000000000000, 0.000000000000],
            [1.000000000000, 0.000000000000, 0.000000000000, 0.000000000000],
            [1.000000000000, 0.000000000000, 0.000000000000, 0.000000000000],
            [1.000000000000, 0.000000000000, 0.000000000000, 0.000000000000],
            [0.978260869565, 0.021739130435, 0.000000000000, 0.000000000000],
            [0.961995249406, 0.000000000000, 0.038004750594, 0.000000000000],
            [0.695398351648, 0.293612637363, 0.010989010989, 0.000000000000],
            [0.563094377398, 0.000000000000, 0.436905622602, 0.000000000000],
            [0.577173020795, 0.243695275447, 0.179131703758, 0.000000000000],
            [0.440641940235, 0.009792043116, 0.549566016649, 0.000000000000],
            [0.327092526383, 0.007268722809, 0.665638750808, 0.000000000000],
            [0.231601833952, 0.005146707421, 0.763251458627, 0.000000000000],
            [0.363595123086, 0.153517940859, 0.482886936055, 0.000000000000],
            [0.581770726619, 0.000000000000, 0.418229273381, 0.000000000000],
            [0.460310521509, 0.010229122700, 0.529460355791, 0.000000000000],
            [0.528488574917, 0.223139620520, 0.248371804563, 0.000000000000],
            [0.396073244462, 0.008801627655, 0.595125127884, 0.000000000000],
            [0.620770276994, 0.000000000000, 0.379229723006, 0.000000000000],
            [0.600030132245, 0.253346055837, 0.146623811919, 0.000000000000],
            [0.466941808551, 0.000000000000, 0.533058191449, 0.000000000000],
            [0.351028271690, 0.007800628260, 0.641171100050, 0.000000000000],
            [0.250990769393, 0.005577572653, 0.743431657954, 0.000000000000],
            [0.382132408333, 0.161344794629, 0.456522797038, 0.000000000000],
            [0.273067954724, 0.000000000000, 0.726932045276, 0.000000000000],
            [0.484299766673, 0.000000000000, 0.515700233327, 0.000000000000],
            [0.366843083527, 0.008152068523, 0.625004847950, 0.000000000000],
            [0.000000000000, 0.000000000000, 0.000000000000, 1.000000000000]])


    def get_expected_backward(self):
        """
        This is for regression testing.
        """
        return np.array([
            [1.59998759e+01, 1.59998759e+01, 1.59998759e+01, 1.59998759e+01],
            [0.00000000e+00, 0.00000000e+00, 5.38641947e+00, 5.38641947e+01],
            [0.00000000e+00, 1.69832642e+01, 1.52849378e+01, 0.00000000e+00],
            [0.00000000e+00, 4.85442755e+00, 4.36898480e+00, 0.00000000e+00],
            [0.00000000e+00, 1.18253534e+01, 1.06428181e+01, 0.00000000e+00],
            [3.57002181e+00, 3.38212593e+00, 3.04391334e+00, 0.00000000e+00],
            [2.74030141e+00, 4.06807631e+00, 3.66126868e+00, 0.00000000e+00],
            [2.20284541e+00, 5.06522538e+00, 4.55870285e+00, 0.00000000e+00],
            [2.12924760e+00, 7.83359119e+00, 7.05023207e+00, 0.00000000e+00],
            [8.52110429e+00, 4.91107202e+00, 4.41996482e+00, 0.00000000e+00],
            [6.38610101e+00, 1.47223181e+00, 1.32500863e+00, 0.00000000e+00],
            [7.36952869e+00, 2.70446198e+00, 2.43401579e+00, 0.00000000e+00],
            [6.73442752e+00, 8.55933274e-01, 7.70339946e-01, 0.00000000e+00],
            [1.08058614e+01, 2.19125897e+00, 1.97213308e+00, 0.00000000e+00],
            [1.69642806e+01, 1.37603587e+00, 1.23843228e+00, 0.00000000e+00],
            [1.24231439e+01, 3.89727560e-01, 3.50754804e-01, 0.00000000e+00],
            [9.22824174e+00, 4.62877309e-01, 4.16589578e-01, 0.00000000e+00],
            [7.05314128e+00, 5.65412861e-01, 5.08871575e-01, 0.00000000e+00],
            [7.24236478e+00, 9.27279154e-01, 8.34551239e-01, 0.00000000e+00],
            [6.18148775e+00, 3.00343351e-01, 2.70309016e-01, 0.00000000e+00],
            [4.46841749e+00, 3.47375139e-01, 3.12637626e-01, 0.00000000e+00],
            [4.53835158e+00, 1.36639722e-01, 1.22975750e-01, 0.00000000e+00],
            [4.43969176e+00, 2.13870870e-01, 1.92483783e-01, 0.00000000e+00],
            [4.44444444e+00, 3.42193392e-01, 3.07974052e-01, 0.00000000e+00],
            [4.44444444e+00, 1.36877357e-01, 1.23189621e-01, 0.00000000e+00],
            [4.44444444e+00, 2.19003771e-01, 1.97103394e-01, 0.00000000e+00],
            [4.00000000e+00, 3.15365430e-01, 2.83828887e-01, 0.00000000e+00]])


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

    def get_expected_scaling(self):
        """
        This is for regression testing.
        """
        return np.array([
            8.750000000000000e-01,
            1.562500000000000e-01,
            2.453125000000000e-01,
            2.016321656050955e-01,
            1.510982230997039e-01])

    def get_expected_posterior(self):
        """
        This is for regression testing.
        """
        return np.array([
            [1.0000000000000000, 0.0000000000000000, 0.0],
            [0.5173540220498163, 0.4826459779501838, 0.0],
            [0.7060024499795835, 0.2939975500204164, 0.0],
            [0.6545528787260105, 0.3454471212739894, 0.0],
            [0.0000000000000000, 0.0000000000000000, 1.0]])

    def get_expected_forward(self):
        """
        This is for regression testing.
        """
        return np.array([
            [1.0000000000000000, 0.0000000000000000, 0.0],
            [0.7000000000000000, 0.3000000000000000, 0.0],
            [0.8471337579617835, 0.1528662420382166, 0.0],
            [0.7912142152023692, 0.2087857847976308, 0.0],
            [0.0000000000000000, 0.0000000000000000, 1.0]])

    def get_expected_backward(self):
        """
        This is for regression testing.
        """
        return np.array([
            [6.618211514904042, 6.618211514904042, 6.618211514904042],
            [4.102899142507145, 8.205798285014291, 32.82319314005716],
            [3.397305022458146, 7.839934667211106, 0.000000000000000],
            [4.730093915884034, 10.29644752960392, 0.000000000000000],
            [1.142857142857143, 2.548912092399230, 0.000000000000000]])
