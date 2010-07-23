"""
Analyze a fasta file.
"""

import numpy as np
from hmmus import hmm

# state 0: homozygous
# state 1: heterozygous
# state 2: bad

def get_default_distribution():
    return np.array([0.5, 0.25, 0.25])

def get_default_trans():
    return np.array([
        [0.5, 0.25, 0.25],
        [0.25, 0.5, 0.25],
        [0.25, 0.25, 0.5]])

def main():
    lines = open('sample.fasta').readlines()
    print set(''.join(lines[1:]))

if __name__ == '__main__':
    main()
