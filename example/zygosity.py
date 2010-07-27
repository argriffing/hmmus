"""
Analyze a fasta file.
"""

from StringIO import StringIO
import sys

import argparse
import numpy as np
from scipy import optimize

from hmmus import hmm
from hmmus import estimation

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

def main(args):
    # read the fasta file
    with open(args.fasta) as fin:
        raw_observations = fasta_to_raw_observations(fin.readlines())
        arr = [g_letter_to_emission[c] for c in raw_observations]
        observations = np.array(arr, dtype=np.int8)
    # init the model
    model = estimation.FiniteModel(
            get_default_trans(), get_default_emissions(), observations)
    # begin writing iteration summaries
    out = StringIO()
    # begin storing the log likelihoods
    log_likelihoods = []
    # do a bunch of baum welch iterations
    for i in range(args.n+1):
        # run the algorithms implemented in c
        log_likelihood = model.get_log_likelihood()
        trans_expect, emiss_expect = model.get_expectations()
        if args.ticks:
            sys.stderr.write('.')
        # store the log likelihood to print later
        log_likelihoods.append(log_likelihood)
        # summarize the current state
        print >> out, 'iteration %d:' % i
        print >> out
        print >> out, 'hidden state distribution:'
        print >> out, model.distn
        print >> out
        print >> out, 'hidden state transition matrix:'
        print >> out, model.trans
        print >> out
        print >> out, 'emission matrix'
        print >> out, model.emiss
        print >> out
        print >> out, 'log likelihood:'
        print >> out, log_likelihood
        if i == args.n:
            break
        print >> out
        print >> out
        # update transitions and emissions for the next iteration
        trans = np.array([r / r.sum() for r in trans_expect])
        emiss = np.array([r / r.sum() for r in emiss_expect])
        model.update(trans, emiss)
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
        hmm.pretty_print_posterior(
                raw_observations, model.get_posterior(), 60, args.posterior)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fasta', required=True,
            help='read this single sequence fasta file')
    parser.add_argument('--ticks', action='store_true',
            help='draw a mark for each baum welch iteration')
    parser.add_argument('--n', default=10, type=int,
            help='do this many baum welch iterations')
    parser.add_argument('--posterior',
            help='write a probabilistic posterior decoding to this file')
    parser.add_argument('--log_likelihoods',
            help='write a list of sequence log likelihoods to this file')
    parser.add_argument('--summary',
            help='write a summary of baum welch iterations to this file')
    main(parser.parse_args())
