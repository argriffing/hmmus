"""
Analyze a fasta file.
"""

from StringIO import StringIO

import argparse
import numpy as np

from hmmus import hmm

# state 0: homozygous
# state 1: heterozygous
# state 2: bad

# emission 0: ACGT
# emission 1: MRWSYK
# emission 2: N

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

def baum_welch_update(distn, emissions, trans,
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

def gen_index_groups(total, groupsize):
    """
    @param total: total number of indices
    @param groupsize: size of a group
    """
    ncompleted = 0
    while ncompleted < total:
        n = min(total-ncompleted, groupsize)
        yield range(ncompleted, ncompleted+n)
        ncompleted += n

def get_float_line(floats):
    """
    @param floats: a sequence of floats between 0 and 1
    """
    arr = [min(9, max(0, int(x*10))) for x in floats]
    return ''.join(str(x) for x in arr)

def write_posterior_text(filename, raw_observations, posterior):
    """
    @param filename: write to this text file
    @param raw_observations: a string with one character per raw observation
    @param posterior: a 2d numpy array of state probabilites per position
    """
    nobs, nstates = posterior.shape
    with open(filename, 'w') as fout:
        for indices in gen_index_groups(nobs, 60):
            low, high = indices[0], indices[0]+len(indices)
            print >> fout, raw_observations[low:high]
            chunk = posterior[low:high].T
            for i in range(nstates):
                print >> fout, get_float_line(chunk[i])
            print >> fout

def main(args):
    # read the fasta file
    with open(args.fasta) as fin:
        raw_observations = fasta_to_raw_observations(fin.readlines())
        arr = [g_letter_to_emission[c] for c in raw_observations]
        observations = np.array(arr, dtype=np.int8)
    # define some initial hmm parameters
    nstates = 3
    nalpha = 3
    distn = get_default_distn()
    trans = get_default_trans()
    emissions = get_default_emissions()
    # define some filenames
    fn_v = 'observation.bin'
    fn_l = 'likelihood.bin'
    fn_f = 'forward.bin'
    fn_s = 'scaling.bin'
    fn_b = 'backward.bin'
    fn_d = 'posterior.bin'
    # write the observation file
    nobs = len(observations)
    observations.tofile(fn_v)
    # begin writing iteration summaries
    out = StringIO()
    # begin storing the log likelihoods
    log_likelihoods = []
    # do a bunch of baum welch iterations
    for i in range(args.n+1):
        # run the algorithms implemented in c
        triple = baum_welch_update(distn, emissions, trans,
                fn_v, fn_l, fn_f, fn_s, fn_b, fn_d)
        log_likelihood, trans_expectations, emission_expectations = triple
        # store the log likelihood to print later
        log_likelihoods.append(log_likelihood)
        # summarize the current state
        print >> out, 'iteration %d:' % i
        print >> out
        print >> out, 'hidden state distribution:'
        print >> out, distn
        print >> out
        print >> out, 'hidden state transition matrix:'
        print >> out, trans
        print >> out
        print >> out, 'emission matrix'
        print >> out, emissions
        print >> out
        print >> out, 'log likelihood:'
        print >> out, log_likelihood
        if i == args.n:
            break
        print >> out
        print >> out
        # update distn, trans, and emissions for the next iteration
        distn = emission_expectations.sum(axis=1) / nobs
        trans = np.array([r / r.sum() for r in trans_expectations])
        emissions = np.array([r / r.sum() for r in emission_expectations])
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
        posterior = np.fromfile(fn_d, dtype=float).reshape((nobs, nstates))
        with open(args.posterior, 'w') as fout:
            write_posterior_text(args.posterior, raw_observations, posterior)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fasta', required=True,
            help='read this single sequence fasta file')
    parser.add_argument('--n', default=10, type=int,
            help='do this many baum welch iterations')
    parser.add_argument('--posterior',
            help='write a probabilistic posterior decoding to this file')
    parser.add_argument('--log_likelihoods',
            help='write a list of sequence log likelihoods to this file')
    parser.add_argument('--summary',
            help='write a summary of baum welch iterations to this file')
    main(parser.parse_args())
