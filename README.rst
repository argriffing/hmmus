About hmmus
===========

Hmmus has some
C implementations of HMM algorithms
with Python bindings,
and it is meant to be useful under the following conditions:

* The sequence of observations to be analyzed is so long
  that it does not fit conveniently in RAM.
* Likelihoods per hidden state per position have been precalculated.
* Numerical stability is important, but is not so important
  that error bounds on the output are required.
* Speed is important.
* The number of hidden states is small.
* The matrix of probabilities of transitions between hidden states is dense.
* Binary data files are acceptable as input and output.

This project would be especially useless in the following cases:

* User friendly or pedagogically informative software is desired.
* All of the data can fit in RAM and numerical stability is not an issue.
* The hidden state transitions are defined by a large sparse graph.
* The emission distributions are uncomplicated (e.g. finite or normal).
* A variable number of observations are emitted per hidden state.
* Silent states other than start and stop states are used.


Requirements
============

Operating system requirements:

* This project was developed using Ubuntu,
  so it will probably work on Debian-based Linux distributions.
* It might work with non-Debian-based Unix variants.
* It probably will not work on Windows.

Major dependencies:

* A recent version of Python-2.x_ (2.6+).
* A C compiler which is not too different from gcc.

Python package and module dependencies:

* numpy_ (version 2.0+ to support the new-style buffer interface;
  if this has not been released yet,
  then use a development version from the subversion repository)
* argparse_ (included in Python-2.7+ and in Python-3.2+)


Installation
============

Setting up virtualenv and pip
-----------------------------

A good way to install hmmus is with virtualenv_ and pip_.
If you are already using these programs and you've activated
a virtual environment, then you can ignore this section.

These programs have been packaged for Ubuntu and probably Debian,
and can be installed from the Linux distribution package repository
as follows::

    $ sudo apt-get install python-virtualenv
    $ sudo apt-get install python-pip

Next use virtualenv to create a virtual python environment::

    $ virtualenv ~/myenv

Now activate the virtual environment::

    $ . ~/myenv/bin/activate

Installing required Python modules and packages
-----------------------------------------------

The following packages and modules should be installed:

* The ``numpy`` package should be installed
  by ``sudo apt-get install python-numpy`` on Debian and Ubuntu.
  Or to get a newer version, install from subversion.
* The ``argparse`` module can be installed
  by ``pip install argparse`` in the activated virtual environment.

Installing hmmus
----------------

The easiest way to install hmmus is from the
python package index pypi_ as follows::

    $ pip install hmmus

If pypi is inaccessible for some reason,
then hmmus can alternatively be installed directly from its github_
repository as follows::

    $ pip install git+git://github.com/argriffing/hmmus

If you are developing hmmus or have cloned the git repo
as ``~/repos/hmmus`` for some other reason,
hmmus can be installed from this local repository as follows::

    $ pip install -e ~/repos/hmmus


Uninstalling hmmus
------------------

It is easy to uninstall hmmus using pip::

    $ pip uninstall hmmus

If this fails for some reason and you really want to get rid of hmmus,
then you can delete the virtual environment into which hmmus
was installed.


Demo
====

In its current incarnation
hmmus provides some scripts for doing posterior decoding,
using unfriendly binary files for input and output.
The following commands create an empty directory
and then fill it with some sample input files::

    $ mkdir mydemo
    $ cd mydemo
    $ hmm-demo smith

This creates the files 
``distribution.bin``,
``transitions.bin``, and
``likelihoods.bin``
from a numerical example in the paper
http://www.cs.cmu.edu/~nasmith/papers/smith.tut04a.pdf
which explains posterior decoding.
The first two binary files define the initial distribution
and the transition matrix of the HMM.
The third binary file defines the sequence of
likelihoods at each position conditional on each hidden state.

To get the position specific posterior distributions of hidden states,
run these three commands::

    $ hmm-forward
    $ hmm-backward
    $ hmm-posterior

This should create four more binary files in the ``mydemo`` directory,
including one named ``posterior.bin`` which has the distributions of interest.
To look at this binary file, use the octal display utility with a format
of 8-byte floating point numbers and a width of 24 bytes per row::

    $ od --format=f8 --width=24 posterior.bin

Until better documentation is written,
information about the usage of the hmmus-associated scripts can be found
using commands like this::

    $ hmm-backward --help


Usage
=====

For now, the only interface to the
posterior decoding is through the binary files.


.. _Python-2.x: http://www.python.org
.. _argparse: http://code.google.com/p/argparse
.. _virtualenv: http://virtualenv.openplans.org
.. _pip: http://pip.openplans.org
.. _pypi: http://pypi.python.org
.. _github: http://github.com
.. _numpy: http://numpy.scipy.org
