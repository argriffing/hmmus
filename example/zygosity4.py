"""
Analyze a fasta file using a four-state HMM.
"""

import numpy as np

import zygohelper

# state 0: homozygous-rich
# state 1: heterozygous-rich
# state 2: null-rich
# state 3: wildcard

# emission 0: ACGT
# emission 1: MRWSYK
# emission 2: N

g_letter_to_emission = {
        'A':0, 'C':0, 'G':0, 'T':0,
        'M':1, 'R':1, 'W':1, 'S':1, 'Y':1, 'K':1,
        'N':2}

g_default_trans = np.array([
    [0.7, 0.1, 0.1, 0.1],
    [0.1, 0.7, 0.1, 0.1],
    [0.1, 0.1, 0.7, 0.1],
    [0.1, 0.1, 0.1, 0.7]])

g_default_emiss = np.array([
    [0.8, 0.1, 0.1],
    [0.1, 0.8, 0.1],
    [0.1, 0.1, 0.8],
    [0.4, 0.3, 0.3]])

if __name__ == '__main__':
    zygohelper.run(__doc__,
            g_letter_to_emission, g_default_trans, g_default_emiss)
