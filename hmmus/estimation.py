"""
This module is about the estimation of the HMM parameters.
From this viewpoint the hidden state labels are nuisances
which we do not care about directly.
An example of the estimation of HMM parameters is Baum-Welch.
Baum-Welch is an EM algorithm for HMMs,
but black box optimization is also possible.
Matrices and vectors are numpy arrays.
One exception is the tuple of serialized parameters
for black box optimization.
Another restriction for this estimation framework is that we
assume that we see only a single sequence,
and that the first element of the sequence is chosen
according to the stationary distribution of the model,
and that that sequence is ended at an arbitrary point.
"""

import numpy as np

EVAL_NONE = 0
EVAL_STATIONARY = 1
EVAL_LOG_LIKELIHOOD = 2
EVAL_POSTERIOR = 3
EVAL_ALL = 4

def get_stationary_distribution(T):
    """
    @param T: a right stochastic matrix
    @return: a stochastic vector defining the stationary distribution
    """
    # Do validation
    nrows, ncols = T.shape
    if nrows != ncols:
        raise ValueError('expected a square transition matrix')
    if not np.allclose(np.sum(T, axis=1), np.ones(ncols)):
        raise ValueError('expected a right stochastic transition matrix')
    # We want a left eigenvector of T.
    # Numpy's eig gives only the right eigenvectors,
    # so use the transpose of T.
    w, VT = np.linalg.eig(T.T)
    # We want the eigenvector that corresponds to the eigenvalue of 1.
    # No eigenvalue should be greater than 1.
    best_w, best_v = max((x, v) for x, v in zip(w, VT.T))
    # The eigenvector might have tiny imaginary parts, so remove them.
    best_v = np.array([abs(x) for x in best_v])
    # Force the elements of the dominant eigenvector to sum to one.
    return best_v / best_v.sum()

class FiniteModel:
    """
    This is a restricted hidden Markov model.
    Each hidden state emits a single observed symbol according to a
    finite distribution over an alphabet of less than 128 letters.
    Therefore silent states are not allowed.
    This model is also restricted to relatively short
    observation arrays such that 32*n*k bytes fits easily in RAM where n is
    the number of observations and k is the number of hidden states.
    """

    def __init__(self, trans, emiss, obs):
        """
        Initialize the model.
        Over the lifetime of the instance, a few invariants should hold.
        The zeros in the transition and emission matrices
        will not change under baum-welch estimation,
        and the dimensionality of the parameter space is intentionally
        reduced so that they will not change under black box
        optimization procedures either.
        The array of observations does not change.
        The shapes of the transition and emission matrices are invariant.
        The dimensionality of the model parameters does not change,
        because it depends only on the shapes and zeros of the
        transition and emission matrices.
        @param trans: a right stochastic transition matrix
        @param emiss: a right stochastic emission matrix
        @param obs: a dtype np.int8 observation array
        """
        # get some shapes
        self.nobs = obs.shape[0]
        self.nstates = emiss.shape[0]
        self.nalpha = emiss.shape[1]
        # initialize using the inputs
        self.trans = trans
        self.emiss = emiss
        self.v_big = obs
        # allocate the large arrays
        self.l_big = np.zeros((self.nobs, self.nstates))
        self.f_big = np.zeros((self.nobs, self.nstates))
        self.s_big = np.zeros(self.nobs)
        self.b_big = np.zeros((self.nobs, self.nstates))
        self.d_big = np.zeros((self.nobs, self.nstates))
        # initialize some quantities which have not yet been defined
        self.distn = None
        self.trans_expect = None
        self.emiss_expect = None
        # define the evaluation level
        self.eval_level = EVAL_NONE

    def _eval_stationary(self):
        self.distn = get_stationary_distribution(self.trans)
        self.eval_level = max(self.eval_level, EVAL_STATIONARY)

    def _eval_log_likelihood(self):
        if self.eval_level < EVAL_STATIONARY:
            self._eval_stationary()
        hmm.finite_alphabet_likelihoods_nodisk(
                self.emiss, self.v_big, self.l_big)
        hmm.forward_nodisk(
                self.distn, self.trans, self.l_big, self.f_big, self.s_big)
        self.log_likelihood = hmm.sequence_log_likelihood_nodisk(
                self.s_big)
        self.eval_level = max(self.eval_level, EVAL_LOG_LIKELIHOOD)

    def _eval_posterior(self):
        if self.eval_level < EVAL_LOG_LIKELIHOOD:
            self._eval_stationary()
        hmm.backward_nodisk(
                self.distn, self.trans, self.l_big, self.s_big, self.b_big)
        hmm.posterior_nodisk(
                self.f_big, self.s_big, self.b_big, self.d_big)
        self.eval_level = max(self.eval_level, EVAL_POSTERIOR)

    def _eval_all(self):
        if self.eval_level < EVAL_POSTERIOR:
            self._eval_stationary()
        self.trans_expect = np.zeros((self.nstates, self.nstates))
        self.emiss_expect = np.zeros((self.nstates, self.nalpha))
        hmm.transition_expectations_nodisk(
                self.trans, self.trans_expect,
                self.l_big, self.f_big, self.b_big)
        hmm.emission_expectations_nodisk(
                self.emiss_expect, self.v_big, self.d_big)
        self.eval_level = max(self.eval_level, EVAL_ALL)

    def update(self, trans, emiss):
        self.trans = trans
        self.emiss = emiss
        self.eval_level = EVAL_NONE

