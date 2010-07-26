"""
Analyze a fasta file.
"""

from StringIO import StringIO
import sys

import argparse
import numpy as np
from scipy import optimize

from hmmus import hmm

# state 0: homozygous
# state 1: heterozygous
# state 2: bad

# emission 0: ACGT
# emission 1: MRWSYK
# emission 2: N

g_fn_v = 'observation.bin'
g_fn_l = 'likelihood.bin'
g_fn_f = 'forward.bin'
g_fn_s = 'scaling.bin'
g_fn_b = 'backward.bin'
g_fn_d = 'posterior.bin'

g_letter_to_emission = {
        'A':0, 'C':0, 'G':0, 'T':0,
        'M':1, 'R':1, 'W':1, 'S':1, 'Y':1, 'K':1,
        'N':2}

def gen_nonempty_stripped(raw_lines):
    for line in raw_lines:
        line = line.strip()
        if line:
            yield line

def get_default_distn():
    return np.array([0.5, 0.25, 0.25])

def get_default_trans():
    return np.array([
        [0.4, 0.3, 0.3],
        [0.3, 0.4, 0.3],
        [0.3, 0.3, 0.4]])

def get_default_emissions():
    return np.array([
        [0.8, 0.1, 0.1],
        [0.1, 0.8, 0.1],
        [0.0, 0.0, 1.0]])

def get_stationary_distribution(T):
    """
    This is for non-baum-welch optimization.
    @param transition_matrix: a right stochastic matrix
    @return: a stochastic vector
    """
    # Do validation
    nrows, ncols = T.shape
    if nrows != ncols:
        raise ValueError('expected a square transition matrix')
    if not np.allclose(np.sum(T, axis=1), np.ones(ncols)):
        raise ValueError('expected a right stochastic transition matrix')
    # We want a left eigenvector of T.
    # Numpy's eig gives only the right eigenvectors,
    # so use the transpose of T.
    w, VT = np.linalg.eig(T.T)
    # We want the eigenvector that corresponds to the eigenvalue of 1.
    # No eigenvalue should be greater than 1.
    best_w, best_v = max((x, v) for x, v in zip(w, VT.T))
    # The eigenvector might have tiny imaginary parts, so remove them.
    best_v = np.array([abs(x) for x in best_v])
    # Force the elements of the dominant eigenvector to sum to one.
    return best_v / best_v.sum()

def fasta_to_raw_observations(raw_lines):
    """
    Assume that the first line is the header.
    @param raw_lines: lines of a fasta file with a single sequence
    @return: a single line string
    """
    lines = list(gen_nonempty_stripped(raw_lines))
    if not lines[0].startswith('>'):
        msg = 'expected the first line to start with ">"'
        raise ValueError(msg)
    data_lines = lines[1:]
    return ''.join(data_lines)

def observations_to_likelihoods(observations, emissions, fn_l):
    """
    Note that this function has been replaced by a C function.
    The C function is finite_alphabet_likelihoods.
    @param observations: a 1d vector of int8 observations
    @param emissions: a matrix of emission probabilities per state
    @param fn_l: name of the likelhoods filename to write
    """
    nstates = len(emissions)
    arr = [[emissions[i][j] for i in range(nstates)] for j in observations]
    np.array(arr, dtype=float).tofile(fn_l)

def baum_welch_update_disk(distn, emissions, trans,
        fn_v, fn_l, fn_f, fn_s, fn_b, fn_d):
    nstates, nalpha = emissions.shape
    # initialize the expecations
    trans_expectations = np.zeros((nstates, nstates))
    emission_expectations = np.zeros((nstates, nalpha))
    # do the baum welch iteration
    hmm.finite_alphabet_likelihoods(emissions, fn_v, fn_l)
    hmm.fwdbwd_alldisk(distn, trans, fn_l, fn_f, fn_s, fn_b, fn_d)
    hmm.transition_expectations(trans, trans_expectations, fn_l, fn_f, fn_b)
    hmm.emission_expectations(emission_expectations, fn_v, fn_d)
    log_likelihood = hmm.sequence_log_likelihood(fn_s)
    # return the expectations
    return log_likelihood, trans_expectations, emission_expectations

def baum_welch_update_ram(distn, emissions, trans,
        v_big, l_big, f_big, s_big, b_big, d_big):
    nstates, nalpha = emissions.shape
    # initialize the expecations
    trans_expect = np.zeros((nstates, nstates))
    emiss_expect = np.zeros((nstates, nalpha))
    # do the baum welch iteration
    hmm.finite_alphabet_likelihoods_nodisk(emissions, v_big, l_big)
    hmm.forward_nodisk(distn, trans, l_big, f_big, s_big)
    hmm.backward_nodisk(distn, trans, l_big, s_big, b_big)
    hmm.posterior_nodisk(f_big, s_big, b_big, d_big)
    hmm.transition_expectations_nodisk(trans, trans_expect,
            l_big, f_big, b_big)
    hmm.emission_expectations_nodisk(emiss_expect, v_big, d_big)
    log_likelihood = hmm.sequence_log_likelihood_nodisk(s_big)
    # return the expectations
    return log_likelihood, trans_expect, emiss_expect


class BaumWelch:

    def __init__(self, observations):
        self.distn = get_default_distn()
        self.trans = get_default_trans()
        self.emissions = get_default_emissions()
        self.nstates = 3
        self.nalpha = 3
        self.nobs = len(observations)

    def maximization_step(self, trans_expect, emiss_expect):
        self.trans = np.array([r / r.sum() for r in trans_expect])
        self.emissions = np.array([r / r.sum() for r in emiss_expect])
        #self.distn = emiss_expect.sum(axis=1) / self.nobs
        self.distn = get_stationary_distribution(self.trans)


class BaumWelchDisk(BaumWelch):

    def __init__(self, observations):
        BaumWelch.__init__(self, observations)
        observations.tofile(g_fn_v)

    def expectation_step(self):
        return baum_welch_update_disk(
                self.distn, self.emissions, self.trans,
                g_fn_v, g_fn_l, g_fn_f, g_fn_s, g_fn_b, g_fn_d)

    def get_posterior(self):
        shape = (self.nobs, self.nstates)
        return np.fromfile(g_fn_d, dtype=float).reshape(shape)


class BaumWelchRam(BaumWelch):

    def __init__(self, observations):
        BaumWelch.__init__(self, observations)
        self.v_big = observations
        self.l_big = np.zeros((self.nobs, self.nstates))
        self.f_big = np.zeros((self.nobs, self.nstates))
        self.s_big = np.zeros(self.nobs)
        self.b_big = np.zeros((self.nobs, self.nstates))
        self.d_big = np.zeros((self.nobs, self.nstates))

    def expectation_step(self):
        return baum_welch_update_ram(
                self.distn, self.emissions, self.trans,
                self.v_big, self.l_big, self.f_big,
                self.s_big, self.b_big, self.d_big)

    def get_posterior(self):
        return self.d_big

    def unpack_arg_tuple(self, arg_tuple):
        """
        @param arg_tuple: parameters defining two stochastic matrices
        """
        #TODO do this
        pass

    def __call__(self, arg_tuple):
        """
        This is called by a scipy black box function minimizer.
        @param arg_tuple: parameters defining two stochastic matrices
        """
        self.unpack_arg_tuple(arg_tuple)
        log_likelihood, trans_expect, emiss_expect = self.expectation_step()
        return -log_likelihood


def main(args):
    # read the fasta file
    with open(args.fasta) as fin:
        raw_observations = fasta_to_raw_observations(fin.readlines())
        arr = [g_letter_to_emission[c] for c in raw_observations]
        observations = np.array(arr, dtype=np.int8)
    # init a baum welch object
    if args.memory == 'disk':
        bm = BaumWelchDisk(observations)
    elif args.memory == 'ram':
        bm = BaumWelchRam(observations)
    else:
        raise ValueError('invalid memory choice')
    # begin writing iteration summaries
    out = StringIO()
    # begin storing the log likelihoods
    log_likelihoods = []
    # do a bunch of baum welch iterations
    for i in range(args.n+1):
        # run the algorithms implemented in c
        log_likelihood, trans_expect, emiss_expect = bm.expectation_step()
        if args.ticks:
            sys.stderr.write('.')
        # store the log likelihood to print later
        log_likelihoods.append(log_likelihood)
        # summarize the current state
        print >> out, 'iteration %d:' % i
        print >> out
        print >> out, 'hidden state distribution:'
        print >> out, bm.distn
        print >> out
        print >> out, 'hidden state transition matrix:'
        print >> out, bm.trans
        print >> out
        print >> out, 'emission matrix'
        print >> out, bm.emissions
        print >> out
        print >> out, 'log likelihood:'
        print >> out, log_likelihood
        if i == args.n:
            break
        print >> out
        print >> out
        # update distn, trans, and emissions for the next iteration
        bm.maximization_step(trans_expect, emiss_expect)
    if args.ticks:
        sys.stderr.write('\n')
    # report the summary
    if args.summary:
        with open(args.summary, 'w') as fout:
            print >> fout, out.getvalue()
    # report the log likelihoods
    if args.log_likelihoods:
        with open(args.log_likelihoods, 'w') as fout:
            print >> fout, '\n'.join('%f' % x for x in log_likelihoods)
    # report the posterior distribution
    if args.posterior:
        posterior = bm.get_posterior()
        hmm.pretty_print_posterior(
                raw_observations, posterior, 60, args.posterior)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fasta', required=True,
            help='read this single sequence fasta file')
    parser.add_argument('--ticks', action='store_true',
            help='draw a mark for each baum welch iteration')
    parser.add_argument('--memory', choices=('ram', 'disk'), default='disk',
            help='use this medium for storing intermediate arrays')
    parser.add_argument('--n', default=10, type=int,
            help='do this many baum welch iterations')
    parser.add_argument('--posterior',
            help='write a probabilistic posterior decoding to this file')
    parser.add_argument('--log_likelihoods',
            help='write a list of sequence log likelihoods to this file')
    parser.add_argument('--summary',
            help='write a summary of baum welch iterations to this file')
    main(parser.parse_args())
