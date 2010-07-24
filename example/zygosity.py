"""
Analyze a fasta file.
"""

from StringIO import StringIO

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

def gen_nonempty_rstripped(raw_lines):
    for line in raw_lines:
        line = line.rstrip()
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

def fasta_to_int8(raw_lines):
    """
    Assume that the first line is the header.
    @param raw_lines: lines of a fasta file with a single sequence
    @return: a numpy array of int8 emissions
    """
    lines = list(gen_nonempty_rstripped(raw_lines))
    if not lines[0].startswith('>'):
        msg = 'expected the first line to start with ">"'
        raise ValueError(msg)
    data_lines = lines[1:]
    arr = [g_letter_to_emission[c] for c in ''.join(data_lines)]
    return np.array(arr, dtype=np.int8)

def main():
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
    # read the fasta file
    with open('sample.fasta') as fin:
        data = fasta_to_int8(fin.readlines())
    nobs = len(data)
    # write the observation file
    data.tofile(fn_v)
    # begin writing iteration summaries
    out = StringIO()
    # do a bunch of baum welch iterations
    niterations = 10
    for i in range(niterations):
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
        print >> out
        if i == niterations-1:
            break
        # construct the likelihoods
        arr = [[emissions[i][j] for i in range(nstates)] for j in data]
        likelihoods = np.array(arr, dtype=float)
        # write the likelihoods file
        likelihoods.tofile(fn_l)
        # do the forward and backward algorithm creating a ton of files
        hmm.fwdbwd_alldisk(distn, trans, fn_l, fn_f, fn_s, fn_b, fn_d)
        # get the posterior transition expectations
        trans_expectations = np.zeros((nstates, nstates))
        hmm.transition_expectations(trans, trans_expectations, fn_l, fn_f, fn_b)
        # get the posterior emission expectations
        emission_expectations = np.zeros((nstates, nalpha))
        hmm.emission_expectations(emission_expectations, fn_v, fn_d)
        # update distn, trans, and emissions
        distn = emission_expectations.sum(axis=1) / nobs
        trans = np.array([r / r.sum() for r in trans_expectations])
        emissions = np.array([r / r.sum() for r in emission_expectations])
    # print the summary
    with open('summary.txt', 'w') as fout:
        print >> fout, out.getvalue()
    # print the posterior distribution
    posterior = np.fromfile(fn_d, dtype=float).reshape((nobs, nstates))
    with open('post.txt', 'w') as fout:
        for c, row in zip(data, posterior):
            print >> fout, c, row.tolist()

if __name__ == '__main__':
    main()
