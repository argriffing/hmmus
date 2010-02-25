Description
===========

This project has some HMM algorithms implemented in C,
and it is meant to be useful under the following conditions:

* The sequence of observations to be analyzed is so long
  that it does not fit conveniently in RAM.
* Likelihoods per hidden state per position have been precalculated.
* Numerical stability is important, but is not so important
  that error bounds on the output are required.
* The number of hidden states is small.
* The matrix of probabilities of transitions between hidden states is dense.
* Binary data files are acceptable as input and output.
* It should be fast

This project would be especially useless in the following cases:

* User friendly or pedagogically informative software is desired.
* All of the data can fit in RAM and numerical stability is not an issue.
* The hidden state transitions are defined by a large sparse graph.
* The emission distributions are uncomplicated (e.g. finite or normal).


Requirements
============

Operating system requirements:

* This project was developed using Ubuntu,
  so it will probably work on Debian-based Linux distributions.
* It might work with non-Debian-based Unix variants.
* It probably will not work on Windows.

Major dependencies:

* A C compiler which is not too different from gcc.

Minor dependencies
(for optional tools):

* A recent version of .. _Python-2.x: http://www.python.org/ (2.6+).
* The .. _argparse: http://code.google.com/p/argparse/ module.


Usage
=====

Right now everything is hardcoded.

To create some files which define the hidden Markov model
together with likelihoods of some observations conditional
on the hidden state, try:

    create-example-likelihoods-a.py

To attempt to analyze the files
produced from the python script above, try:

    gcc hmmus.c
    ./a.out
