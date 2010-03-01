"""A thin wrapper around the C extension.
In this module the term 'matrix' is abused to mean numpy array.
"""

import itertools

import numpy as np

import hmmusc

def is_stochastic_vector(v):
    if len(v.shape) != 1:
        return False
    if any(x<0 for x in v):
        return False
    if abs(1.0 - v.sum()) > 1e-7:
        return False
    return True

def is_square_matrix(M):
    if len(M.shape) != 2:
        return False
    if len(set(M.shape)) != 1:
        return False
    return True

def is_right_stochastic_matrix(M):
    if not is_square_matrix(M):
        return False
    if not all(is_stochastic_vector(v) for v in M):
        return False
    return True

def _simplify(distribution, transitions):
    """
    @param distribution: initial state distribution
    @param transitions: transition probabilities
    """
    # get the initial state distribution as a numpy array
    np_distn = np.array(distribution, dtype=float)
    if not is_stochastic_vector(np_distn):
        msg = 'the initial distribution should be a stochastic vector'
        raise ValueError(msg)
    # get the transition matrix as a numpy array
    np_trans = np.array(transitions, dtype=float)
    if not is_right_stochastic_matrix(np_trans):
        msg = 'the transition matrix should be a right stochastic matrix'
        raise ValueError(msg)
    # the vector and matrix should be conformant
    nstates = np_distn.shape[0]
    if np_trans.shape != (nstates, nstates):
        msg_a = 'the number of states in the initial distribution does not '
        msg_b = 'match the number of states in the transition matrix'
        raise ValueError(msg_a + msg_b)
    # dumb down the numpy arrays into tuples of floats
    tuple_distn = tuple(np_distn.tolist())
    tuple_trans = tuple(itertools.chain.from_iterable(np_trans))
    return tuple_distn, tuple_trans

def forward(distribution, transitions,
        likelihoods_name, forward_name, scaling_name):
    """
    @param distribution: initial state distribution
    @param transitions: transition probabilities
    """
    tuple_distn, tuple_trans = _simplify(distribution, transitions)
    return hmmusc.forward(tuple_distn, tuple_trans,
        likelihoods_name, forward_name, scaling_name)

def backward(distribution, transitions,
        likelihoods_name, scaling_name, backward_name):
    """
    @param distribution: initial state distribution
    @param transitions: transition probabilities
    """
    tuple_distn, tuple_trans = _simplify(distribution, transitions)
    return hmmusc.backward(tuple_distn, tuple_trans,
        likelihoods_name, scaling_name, backward_name)

def posterior(distribution, transitions,
        forward_name, scaling_name, backward_name, posterior_name):
    """
    @param distribution: initial state distribution
    @param transitions: transition probabilities
    """
    tuple_distn, tuple_trans = _simplify(distribution, transitions)
    return hmmusc.posterior(tuple_distn, tuple_trans,
        forward_name, scaling_name, backward_name, posterior_name)

def fwdbwd_alldisk(distribution, transitions,
        likelihoods_name,
        forward_name, scaling_name, backward_name,
        posterior_name):
    """
    @param distribution: initial state distribution
    @param transitions: transition probabilities
    """
    forward(distribution, transitions,
            likelihoods_name, forward_name, scaling_name)
    backward(distribution, transitions,
            likelihoods_name, scaling_name, backward_name)
    posterior(distribution, transitions,
            forward_name, scaling_name, backward_name, posterior_name)

def fwdbwd_somedisk(distribution, transitions,
        likelihoods_name, posterior_name):
    """
    @param distribution: initial state distribution
    @param transitions: transition probabilities
    """
    tuple_distn, tuple_trans = _simplify(distribution, transitions)
    return hmmusc.fwdbwd_somedisk(tuple_distn, tuple_trans,
        likelihoods_name, posterior_name)

def fwdbwd_nodisk(distribution, transitions, likelihoods):
    """
    @param distribution: initial state distribution
    @param transitions: transition probabilities
    @param likelihoods: likelihoods at each state
    @return: posterior distribution at each state
    """
    tuple_distn, tuple_trans = _simplify(distribution, transitions)
    np_likelihoods = np.array(likelihoods, float)
    if len(np_likelihoods.shape) != 2:
        msg = 'the matrix of likelihoods should be rectangular'
        raise ValueError(msg)
    nlikelihoods_rows, nlikelihoods_cols = np_likelihoods.shape
    if nlikelihoods_cols != len(distribution):
        msg = 'likelihood columns should conform to the distribution'
        raise ValueError(msg)
    tuple_likelihoods = tuple(itertools.chain.from_iterable(np_likelihoods))
    tuple_posterior = hmmusc.fwdbwd_nodisk(tuple_distn, tuple_trans,
            tuple_likelihoods)
    np_posterior_unshaped = np.array(tuple_posterior, dtype=float)
    np_posterior = np_posterior_unshaped.reshape(np_likelihoods.shape)
    return np_posterior
