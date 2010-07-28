"""
This module is about the estimation of the HMM parameters.
From this viewpoint the hidden state labels are nuisances
which we do not care about directly.
An example of the estimation of HMM parameters is Baum-Welch.
Baum-Welch is an EM algorithm for HMMs,
but black box optimization is also possible.
Another restriction for this estimation framework is that we
assume that we see only a single sequence,
and that the first element of the sequence is chosen
according to the stationary distribution of the model,
and that that sequence is ended at an arbitrary point.
In this modules, matrices and vectors are numpy arrays.
"""

import math
import unittest

import numpy as np

from hmmus import hmm

EVAL_NONE = 0
EVAL_STATIONARY = 1
EVAL_LOG_LIKELIHOOD = 2
EVAL_POSTERIOR = 3
EVAL_ALL = 4

def baum_welch(update, expect, trans, emiss, iterations, callback=None):
    """
    Do some Baum-Welch iterations.
    The call style of this function mimics scipy.optimize.fmin.
    There are differences so it cannot be used interchangeably with fmin.
    @param update: call this to send new transition and emission matrices
    @param expect: call this to get new transition and emission expectations
    @param trans: initial transition matrix
    @param emiss: initial emission matrix
    @param iterations: do this many iterations
    @param callback: send this the transitions and emissions after each update
    """
    for i in range(iterations+1):
        update(trans, emiss)
        callback(trans, emiss)
        if i == iterations:
            return
        trans_expect, emiss_expect = expect()
        trans = np.array([r / r.sum() for r in trans_expect])
        emiss = np.array([r / r.sum() for r in emiss_expect])

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
        msg = 'expected a right stochastic transition matrix: ' + str(T)
        raise ValueError(msg)
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

def irregular_grouper(seq, sizes):
    """
    Yield irregularly sized chunks of a sequence.
    @param seq: a sequence
    @param sizes: an iterable of sizes
    """
    i = 0
    for k in sizes:
        yield seq[i:i+k]
        i += k

def _row_mask_to_nparams(mask):
    """
    @param mask: a boolean array
    """
    nparams = mask.sum() - 1
    if nparams < 0:
        msg = 'a stochastic vector cannot be all zeros'
        raise ValueError(msg)
    return nparams

def _serialize_row(row, mask):
    """
    This is for stochastic vectors where some elements are forced to zero.
    Such a vector is defined by a number of parameters equal to
    the length of the vector minus one and minus the number of elements
    forced to zero.
    @param row: an array of floats
    @param mask: row-conformant bools such that False forces zero probability
    @return: a shorter array of transformed floats
    """
    nparams = _row_mask_to_nparams(mask)
    if not nparams:
        return []
    ref_value = row[mask][0]
    params = np.log(row[mask][1:] / ref_value)
    return params

def _deserialize_row(params, mask):
    """
    This is for stochastic vectors where some elements are forced to zero.
    Such a vector is defined by a number of parameters equal to
    the length of the vector minus one and minus the number of elements
    forced to zero.
    @param params: an array of statistical parameters
    @param mask: bools such that False forces zero probability
    @return: a mask-conformant list of nonnegative floats
    """
    row = np.zeros(mask.shape)
    row[mask] = [1.0] + np.exp(params).tolist()
    row /= row.sum()
    return row

def _matrix_mask_to_nparams(mask):
    """
    Input is a mask for a constrained stochastic vector.
    @param mask: a boolean array
    """
    return sum(_row_mask_to_nparams(row) for row in mask)

def _serialize_matrix(M, mask):
    return np.hstack([_serialize_row(r, m) for r, m in zip(M, mask)])

def _deserialize_matrix(params, mask):
    sizes = [_row_mask_to_nparams(row) for row in mask]
    ps = list(irregular_grouper(params, sizes))
    return np.vstack([_deserialize_row(p, r) for p, r in zip(ps, mask)])


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
        if len(trans.shape) != 2:
            raise ValueError('expected a two dimensional transition matrix')
        if len(emiss.shape) != 2:
            raise ValueError('expected a two dimensional emission matrix')
        if len(obs.shape) != 1:
            raise ValueError('expected a one dimensional observation matrix')
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
        # mark the locations of nonzeros in original matrices
        self.trans_mask = (self.trans != 0)
        self.emiss_mask = (self.emiss != 0)
        # define the evaluation level
        self.eval_level = EVAL_NONE

    def update(self, trans, emiss):
        self.trans = trans
        self.emiss = emiss
        self.eval_level = EVAL_NONE

    def serialize_params(self):
        """
        @return: a float array
        """
        return np.hstack([
            _serialize_matrix(self.trans, self.trans_mask),
            _serialize_matrix(self.emiss, self.emiss_mask)])

    def deserialize_params(self, params):
        """
        @return: a transition matrix and an emission matrix
        """
        ntrans_params = _matrix_mask_to_nparams(self.trans_mask)
        nemiss_params = _matrix_mask_to_nparams(self.emiss_mask)
        if len(params) != ntrans_params + nemiss_params:
            msg_a = 'got %d parameters ' + len(params)
            msg_b = 'but expected %d' % ntrans_params + nemiss_params
            raise ValueError(msg_a + msg_b)
        trans_params = params[:ntrans_params]
        emiss_params = params[-nemiss_params:]
        trans = _deserialize_matrix(trans_params, self.trans_mask)
        emiss = _deserialize_matrix(emiss_params, self.emiss_mask)
        return trans, emiss

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
            self._eval_log_likelihood()
        hmm.backward_nodisk(
                self.distn, self.trans, self.l_big, self.s_big, self.b_big)
        hmm.posterior_nodisk(
                self.f_big, self.s_big, self.b_big, self.d_big)
        self.eval_level = max(self.eval_level, EVAL_POSTERIOR)

    def _eval_all(self):
        if self.eval_level < EVAL_POSTERIOR:
            self._eval_posterior()
        self.trans_expect = np.zeros((self.nstates, self.nstates))
        self.emiss_expect = np.zeros((self.nstates, self.nalpha))
        hmm.transition_expectations_nodisk(
                self.trans, self.trans_expect,
                self.l_big, self.f_big, self.b_big)
        hmm.emission_expectations_nodisk(
                self.emiss_expect, self.v_big, self.d_big)
        self.eval_level = max(self.eval_level, EVAL_ALL)

    def get_distn(self):
        """
        @return: stationary distribution of hidden states
        """
        if self.eval_level < EVAL_STATIONARY:
            self._eval_stationary()
        return self.distn

    def get_forward(self):
        if self.eval_level < EVAL_LOG_LIKELIHOOD:
            self._eval_log_likelihood()
        return self.forward

    def get_log_likelihood(self):
        """
        @return: log likelihood
        """
        if self.eval_level < EVAL_LOG_LIKELIHOOD:
            self._eval_log_likelihood()
        return self.log_likelihood

    def get_backward(self):
        if self.eval_level < EVAL_POSTERIOR:
            self._eval_posterior()
        return self.backward

    def get_posterior(self):
        """
        @return: posterior distribution over hidden states at each position
        """
        if self.eval_level < EVAL_POSTERIOR:
            self._eval_log_likelihood()
        return self.d_big

    def get_expectations(self):
        """
        @return: transition expectations and emission expectations
        """
        if self.eval_level < EVAL_ALL:
            self._eval_all()
        return self.trans_expect, self.emiss_expect

    def get_trans(self):
        return self.trans

    def get_emiss(self):
        return self.emiss

    def get_trans_expect(self):
        trans_expect, emiss_expect = self.get_expectations()
        return trans_expect

    def get_emiss_expect(self):
        trans_expect, emiss_expect = self.get_expectations()
        return emiss_expect

    def fmin_objective(self, params):
        """
        This function should be called by a black box optimization framework.
        @param params: serialized parameters describing some matrices
        @return: the value of the objective function to be minimized
        """
        trans, emiss = self.deserialize_params(params)
        self.update(trans, emiss)
        return -self.get_log_likelihood()
    

class TestEstimation(unittest.TestCase):

    def setUp(self):
        obs = np.array([0,1,2], dtype=np.int8)
        trans = np.array([
            [0.4, 0.3, 0.3],
            [0.3, 0.4, 0.3],
            [0.3, 0.3, 0.4]])
        emiss = np.array([
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.0, 0.0, 1.0]])
        self.m = FiniteModel(trans, emiss, obs)

    def test_parameter_serialization_consistency(self):
        params = self.m.serialize_params()
        trans, emiss = self.m.deserialize_params(params)
        self.assertTrue(np.allclose(trans, self.m.trans))
        self.assertTrue(np.allclose(emiss, self.m.emiss))

    def test_parameter_serialization(self):
        e_params = np.array([
                -0.28768207, -0.28768207,
                0.28768207, 0.,
                0., 0.28768207,
                -2.07944154, -2.07944154,
                2.07944154, 0.])
        o_params = self.m.serialize_params()
        self.assertTrue(np.allclose(e_params, o_params))


if __name__ == '__main__':
    unittest.main()
