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

def gen_nonempty_stripped(raw_lines):
    for line in raw_lines:
        line = line.strip()
        if line:
            yield line

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


class Summary:

    def __init__(self, model, show_ticks=False):
        """
        @param model: something like an estimation.FiniteModel object
        """
        self.model = model
        self.out = StringIO()
        self.iteration = 0
        self.log_likelihoods = []
        self.show_ticks = show_ticks

    def finish(self):
        if self.show_ticks:
            sys.stderr.write('\n')
        return self.out.getvalue().rstrip()

    def after_fmin(self, params):
        """
        This is called after each black box optimization iteration.
        @param params: the parameters used for the most recent evaluation.
        """
        self.after()

    def after_baum_welch(self, trans, emiss):
        """
        This is called after each Baum-Welch EM iteration.
        @param trans: a matrix used for the most recent evaluation.
        @param emiss: a matrix used for the most recent evaluation.
        """
        self.after()

    def after(self):
        """
        This is called after each optimization iteration.
        """
        print >> self.out, 'iteration %d:' % self.iteration
        print >> self.out
        print >> self.out, 'hidden state distribution:'
        print >> self.out, self.model.get_distn()
        print >> self.out
        print >> self.out, 'hidden state transition matrix:'
        print >> self.out, self.model.get_trans()
        print >> self.out
        print >> self.out, 'emission matrix'
        print >> self.out, self.model.get_emiss()
        print >> self.out
        print >> self.out, 'log likelihood:'
        print >> self.out, self.model.get_log_likelihood()
        print >> self.out
        print >> self.out
        self.log_likelihoods.append(self.model.get_log_likelihood())
        if self.show_ticks:
            sys.stderr.write('.')
        self.iteration += 1


def main(args, letter_to_emission, trans, emiss):
    with open(args.fasta) as fin:
        raw_observations = fasta_to_raw_observations(fin.readlines())
        arr = [letter_to_emission[c] for c in raw_observations]
        observations = np.array(arr, dtype=np.int8)
    model = estimation.FiniteModel(trans, emiss, observations)
    summary = Summary(model)
    estimation.baum_welch(
        model.update, model.get_expectations, trans, emiss, args.n,
        callback=summary.after_baum_welch)
    summary_text = summary.finish()
    if args.summary:
        with open(args.summary, 'w') as fout:
            print >> fout, summary_text
    if args.log_likelihoods:
        with open(args.log_likelihoods, 'w') as fout:
            print >> fout, '\n'.join('%f' % x for x in summary.log_likelihoods)
    if args.posterior:
        hmm.pretty_print_posterior(raw_observations,
                model.get_posterior(), 60, args.posterior)
    if args.posterior_decoding:
        hmm.pretty_print_posterior_decoding(raw_observations,
                model.get_posterior(), 60, args.posterior_decoding)

def run(letter_to_emission, trans, emiss):
    parser = argparse.ArgumentParser()
    parser.add_argument('--fasta', required=True,
            help='read this single sequence fasta file')
    parser.add_argument('--ticks', action='store_true',
            help='draw a mark for each baum welch iteration')
    parser.add_argument('--n', default=20, type=int,
            help='do this many total search iterations')
    #parser.add_argument('--method', choices=['bw', 'nm'], default='bw',
    #help='pure Baum-Welch vs. Nelder-Mead-supplemented search')
    parser.add_argument('--posterior',
            help='write a probabilistic posterior decoding to this file')
    parser.add_argument('--posterior_decoding',
            help='write a hard posterior decoding to this file')
    parser.add_argument('--log_likelihoods',
            help='write a list of sequence log likelihoods to this file')
    parser.add_argument('--summary',
            help='write a summary of baum welch iterations to this file')
    main(parser.parse_args(), letter_to_emission, trans, emiss)