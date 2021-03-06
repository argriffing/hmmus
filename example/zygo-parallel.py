"""
Analyze a fasta file with missing data using two-state HMM.
"""

import numpy as np

from hmmus import hmm
from hmmus import zygohelper

# state 0: homozygous and missing
# state 1: heterozygous

# emission 0: ACGT
# emission 1: MRWSYK
# missing data: N

g_letter_to_emission = {
        'A':0, 'C':0, 'G':0, 'T':0,
        'M':1, 'R':1, 'W':1, 'S':1, 'Y':1, 'K':1,
        'N':hmm.MISSING}

g_default_trans = np.array([
    [0.9, 0.1],
    [0.1, 0.9]])

g_default_emiss = np.array([
    [0.9, 0.1],
    [0.5, 0.5]])

if __name__ == '__main__':
    zygohelper.run_parallel(
            g_letter_to_emission, g_default_trans, g_default_emiss, __doc__)
