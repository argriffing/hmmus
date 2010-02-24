#!/usr/bin/env python

import sys

import argparse
import numpy as np

def main(args):
    shape = (args.nrows, args.ncols)
    M = np.fromfile(args.infile, dtype=float).reshape(shape)
    print M

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('nrows', type=int, help='number of rows'),
    parser.add_argument('ncols', type=int, help='number of columns'),
    parser.add_argument('infile',
            type=argparse.FileType('rb'), nargs='?', default=sys.stdin,
            help='binary file to view')
    main(parser.parse_args())

