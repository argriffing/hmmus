"""
Analyze a fasta file using two-state HMM.
"""

import numpy as np

import zygohelper

# state 0: homozygous and missing
# state 1: heterozygous

# emission 0: ACGTN
# emission 1: MRWSYK

g_letter_to_emission = {
        'A':0, 'C':0, 'G':0, 'T':0, 'N':0,
        'M':1, 'R':1, 'W':1, 'S':1, 'Y':1, 'K':1}

g_default_trans = np.array([
    [0.9, 0.1],
    [0.1, 0.9]])

g_default_emiss = np.array([
    [0.9, 0.1],
    [0.5, 0.5]])

if __name__ == '__main__':
    zygohelper.run(__doc__,
            g_letter_to_emission, g_default_trans, g_default_emiss)
