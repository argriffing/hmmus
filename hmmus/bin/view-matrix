#!/usr/bin/env python

import argparse
import numpy as np

def main(args):
    """
    Print a view of the matrix.
    """
    M = np.fromfile(args.infile, dtype=float)
    nrows = len(M) / args.ncols
    shape = (nrows, args.ncols)
    M = M.reshape(shape)
    print M

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('ncols', type=int, help='number of columns'),
    parser.add_argument('infile', type=argparse.FileType('rb'),
            help='binary file to view')
    main(parser.parse_args())

