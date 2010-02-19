#include <stdio.h>
#include <stdlib.h>

struct TM
{
  /*
   * a transition matrix
   * @member order: number of states
   * @member value: row major matrix elements
   */
  int order;
  double *value;
};

int TM_init(struct TM *p, int order)
{
  /* constructor */
  p->order = order;
  p->value = malloc(order*order*sizeof(double));
  return 0;
}

int TM_del(struct TM *p)
{
  /* destructor */
  free(p->value);
  p->value = NULL;
  return 0;
}

int forward(struct TM *ptm, FILE *fin_l, FILE *fout_f, FILE *fout_s)
{
  /*
   * Run the forward algorithm.
   * @param ptm: address of a transition matrix
   * @param fin_l: file of likelihood vectors
   * @param fout_f: file of forward vectors
   * @param fout_s: file of scaling values
   */
  double *likelihoods = malloc(ptm->order*sizeof(double));
  if (!likelihoods)
  {
    fprintf(stderr, "failed to allocate likelihood array\n");
    return 1;
  }
  double *forward = malloc(ptm->order*sizeof(double));
  if (!forward)
  {
    fprintf(stderr, "failed to allocate forward array\n");
    return 1;
  }
  while (fread(likelihoods, sizeof(double), ptm->order, fin_l))
  {
    ;
  }
  free(likelihoods);
  free(forward);
  return 0;
  /*
    def forward(self, observations):
        """
        For each position yield a scaled f vector and its scaling factor.
        This function may fail if the observed sequence has very low likelihood.
        @param observations: an observation source
        """
        nhidden = len(self.hidden_state_objects)
        # yield f vectors and scaling factors
        for i, obs in enumerate(observations):
            likelihoods = self.get_likelihoods(obs)
            if i:
                f_curr_unscaled = list(likelihoods)
                for sink_index in range(nhidden):
                    p = 0
                    for source_index in range(nhidden):
                        tprob = self.T.get_transition_probability(
                                source_index, sink_index)
                        p += f_prev[source_index] * tprob
                    f_curr_unscaled[sink_index] *= p
            else:
                f_curr_unscaled = [x * p
                        for x, p in zip(likelihoods, self.initial_distribution)]
            scaling_factor = sum(f_curr_unscaled)
            if not scaling_factor:
                raise ValueError('scaling factor is zero at position %d' % i)
            f_curr = [x / scaling_factor for x in f_curr_unscaled]
            yield tuple(f_curr), scaling_factor
            f_prev = f_curr
    */
}

int main(int argc, char* argv[])
{
  struct TM tm;
  FILE *fin_l = fopen("test.likelihoods", "rb");
  if (!fin_l)
  {
    fprintf(stderr, "failed to open the likelihoods file for reading\n");
    return 1;
  }
  FILE *fout_f = fopen("test.forward", "wb");
  if (!fout_f)
  {
    fprintf(stderr, "failed to open the forward vector file for writing\n");
    return 1;
  }
  FILE *fout_s = fopen("test.scaling", "wb");
  if (!fout_s)
  {
    fprintf(stderr, "failed to open the scaling factor file for writing\n");
    return 1;
  }
  printf("init tm\n");
  TM_init(&tm, 2);
  printf("forward\n");
  forward(&tm, fin_l, fout_f, fout_s);
  printf("del tm\n");
  TM_del(&tm);
  printf("close files\n");
  fclose(fin_l);
  fclose(fout_f);
  fclose(fout_s);
  return 0;
}

/*
"""
Use hidden Markov model algorithms on long observed sequences.
The observation list and the forward and backward tables
are not required to fit in memory, so external files are used.
Unobserved emitted states at some positions are allowed.
"""

import itertools
import unittest

import numpy as np

import Util
import HMM
import FastHMM
import DiscreteEndpoint
import TransitionMatrix
import lineario


class Model:
    """
    Methods of this class implement forward and backward HMM algorithms.
    Some of the calling conventions may seem convoluted,
    with seemingly arbitrary required orderings of the input sequences.
    The purpose of the ordering is to allow everything 
    to be done with generators.
    This facilitates dynamic programming algorithms that run forward and 
    backward on external files whose rows are accessed sequentially,
    thus allowing analysis of observed sequences too large too fit in memory.
    """

    def __init__(self, T, hidden_state_objects, cache_size=0):
        """
        @param T: a transition object
        @param hidden_state_objects: a conformant list of hidden state objects
        @param cache_size: the number of observations that are cached
        """
        self.T = T
        self.hidden_state_objects = hidden_state_objects
        self.initial_distribution = T.get_stationary_distribution()
        self.cache_size = cache_size
        self.cache = {}

    def get_likelihoods(self, obs):
        """
        Note that the state may be unknown.
        This function uses memoization.
        @param obs: an emitted state or None
        @return: a tuple of likelihoods
        """
        if obs is None:
            return [1.0] * len(self.hidden_state_objects)
        likelihoods = self.cache.get(obs, None)
        if likelihoods:
            return likelihoods
        likelihoods = tuple(m.get_likelihood(obs)
                for m in self.hidden_state_objects)
        if len(self.cache) < self.cache_size:
            self.cache[obs] = likelihoods
        return likelihoods

    def forward(self, observations):
        """
        For each position yield a scaled f vector and its scaling factor.
        This function may fail if the observed sequence has very low likelihood.
        @param observations: an observation source
        """
        nhidden = len(self.hidden_state_objects)
        # yield f vectors and scaling factors
        for i, obs in enumerate(observations):
            likelihoods = self.get_likelihoods(obs)
            if i:
                f_curr_unscaled = list(likelihoods)
                for sink_index in range(nhidden):
                    p = 0
                    for source_index in range(nhidden):
                        tprob = self.T.get_transition_probability(
                                source_index, sink_index)
                        p += f_prev[source_index] * tprob
                    f_curr_unscaled[sink_index] *= p
            else:
                f_curr_unscaled = [x * p
                        for x, p in zip(likelihoods, self.initial_distribution)]
            scaling_factor = sum(f_curr_unscaled)
            if not scaling_factor:
                raise ValueError('scaling factor is zero at position %d' % i)
            f_curr = [x / scaling_factor for x in f_curr_unscaled]
            yield tuple(f_curr), scaling_factor
            f_prev = f_curr

    def backward(self, reverse_observations, reverse_scaling_factors):
        """
        Yield scaled b vectors in reverse order.
        Observations are expected in reverse order.
        Note that the scaling factors are expected in reverse order;
        that is, reverse with respect to the order in which
        they are generated by the forward algorithm and used
        in posterior decoding.
        @param reverse_observations: an observation source
        @param reverse_scaling_factors: a scaling factor source
        """
        nhidden = len(self.hidden_state_objects)
        # yield b vectors
        for i, (obs, sf) in enumerate(
                itertools.izip(reverse_observations, reverse_scaling_factors)):
            if i:
                likelihoods = self.get_likelihoods(obs_prev)
                b_curr_unscaled = [0.0] * nhidden
                for source_index in range(nhidden):
                    for sink_index in range(nhidden):
                        p = self.T.get_transition_probability(
                                source_index, sink_index)
                        p *= likelihoods[sink_index] * b_prev[sink_index]
                        b_curr_unscaled[source_index] += p
            else:
                b_curr_unscaled = [1.0] * nhidden
            b_curr = [x / sf for x in b_curr_unscaled]
            yield tuple(b_curr)
            b_prev = b_curr
            obs_prev = obs

    def posterior(self, forward, scaling_factors, backward):
        """
        Yield position-specific posterior hidden state distributions.
        @param forward: a source of forward vectors
        @param scaling_factors: a source of scaling factors
        @param backward: a source of backward vectors
        """
        for fs, si, bs in itertools.izip(forward, scaling_factors, backward):
            yield tuple(x*y*si for x, y in zip(fs, bs))

    def transition_expectations(self, observations, forward, backward):
        """
        @param observations: an observation source
        @param forward: a source of forward vectors
        @param backward: a source of backward vectors
        @return: a matrix of expected hidden state transition counts
        """
        nhidden = len(self.hidden_state_objects)
        # initialize the matrix of expected counts
        A = np.zeros((nhidden, nhidden))
        # get the expected counts for each transition
        dp_source = itertools.izip(observations, forward, backward)
        for old, new in Util.pairwise(dp_source):
            o_old, f_old, b_old = old
            o_new, f_new, b_new = new
            likelihoods = self.get_likelihoods(o_new)
            for i, j in itertools.product(range(nhidden), repeat=2):
                tprob = self.T.get_transition_probability(i, j)
                A[i, j] += f_old[i] * tprob * likelihoods[j] * b_new[j]
        return A



class InternalModel:
    """
    This is a wrapper that requires that everything fits into memory.
    It is only for testing.
    """

    def __init__(self, T, hidden_state_objects):
        """
        @param T: a transition object
        @param hidden_state_objects: a conformant list of hidden state objects
        """
        self.model = Model(T, hidden_state_objects)

    def get_dp_info(self, observations):
        """
        Do the dynamic programming and return the results.
        @param observations: the sequence of observations
        @return: (observations, f, s, b)
        """
        observations = list(observations)
        f_s_pairs = list(self.model.forward(observations))
        f, s = zip(*f_s_pairs)
        b_reversed = list(
                self.model.backward(reversed(observations), reversed(s)))
        b = list(reversed(b_reversed))
        return (observations, f, s, b)

    def posterior(self, dp_info):
        """
        @param dp_info: dynamic programming info returned by get_dp_info
        @return: a list of position specific posterior distributions
        """
        observations, f, s, b = dp_info
        return list(self.model.posterior(f, s, b))

    def get_transition_expectations(self, dp_info):
        """
        @param dp_info: dynamic programming info returned by get_dp_info
        @return: a matrix of expected hidden state transition counts
        """
        observations, f, s, b = dp_info
        return self.model.transition_expectations(observations, f, b)

    def get_expectations(self, dp_info):
        """
        @param dp_info: dynamic programming info returned by get_dp_info
        @return: a vector of hidden state occupancy expectations
        """
        expectations = sum(np.array(distribution)
                for distribution in self.posterior(dp_info))
        return expectations


class ExternalModel:

    def __init__(self, T, hidden_state_objects, dp_filenames):
        """
        If a filename is None then that stream will be done in memory.
        The dynamic programming streams are the forward stream,
        the backward stream, and the scaling factor stream.
        @param T: a transition object
        @param hidden_state_objects: a conformant list of hidden state objects
        @param dp_filenames: a tuple of (f, s, b); each is a filename or None
        """
        self.model = Model(T, hidden_state_objects)
        # Define the data type of each stream.
        f_type = lineario.FloatTupleConverter()
        s_type = lineario.FloatConverter()
        b_type = lineario.FloatTupleConverter()
        # initialize the streams for dynamic programming
        f_name, s_name, b_name = dp_filenames
        if f_name is None:
            self.f_stream = lineario.SequentialStringIO(f_type)
        else:
            self.f_stream = lineario.SequentialDiskIO(f_type, f_name)
        if s_name is None:
            self.s_stream = lineario.SequentialStringIO(s_type)
        else:
            self.s_stream = lineario.SequentialDiskIO(s_type, s_name)
        if b_name is None:
            self.b_stream = lineario.SequentialStringIO(b_type)
        else:
            self.b_stream = lineario.SequentialDiskIO(b_type, b_name)

    def init_dp(self, o_stream):
        """
        Initialize the streams using a sequential observation stream.
        @param o_stream: a sequential observation stream
        """
        # Create the forward stream and the scaling factor stream.
        o_stream.open_read()
        self.f_stream.open_write()
        self.s_stream.open_write()
        for f, s in self.model.forward(o_stream.read_forward()):
            self.f_stream.write(f)
            self.s_stream.write(s)
        o_stream.close()
        self.f_stream.close()
        self.s_stream.close()
        # Create the backward stream.
        o_stream.open_read()
        self.s_stream.open_read()
        self.b_stream.open_write()
        o_reversed = o_stream.read_backward()
        s_reversed = self.s_stream.read_backward()
        for b in self.model.backward(o_reversed, s_reversed):
            self.b_stream.write(b)
        o_stream.close()
        self.s_stream.close()
        self.b_stream.close()

    def posterior(self):
        """
        Yield posterior distributions using initialized dp streams.
        """
        self.f_stream.open_read()
        self.s_stream.open_read()
        self.b_stream.open_read()
        f_forward = self.f_stream.read_forward()
        s_forward = self.s_stream.read_forward()
        b_backward = self.b_stream.read_backward()
        for d in self.model.posterior(f_forward, s_forward, b_backward):
            yield d
        self.f_stream.close()
        self.s_stream.close()
        self.b_stream.close()


class TestExternalHMM(unittest.TestCase):

    def test_model_compatibility(self):
        # define the dishonest casino model
        fair_state = HMM.HiddenDieState(1/6.0)
        loaded_state = HMM.HiddenDieState(0.5)
        M = np.array([[0.95, 0.05], [0.1, 0.9]])
        T = TransitionMatrix.MatrixTransitionObject(M)
        hidden_states = [fair_state, loaded_state]
        # define a sequence of observations
        observations = [1, 2, 6, 6, 1, 2, 3, 4, 5, 6]
        # create the reference hidden markov model object
        hmm_old = HMM.TrainedModel(M, hidden_states)
        # create the testing hidden markov model object
        hmm_new = InternalModel(T, hidden_states)
        # get posterior distributions
        distributions_old = hmm_old.scaled_posterior_durbin(observations)
        distributions_new = hmm_new.posterior(hmm_new.get_dp_info(observations))
        # assert that the distributions are the same
        self.assertTrue(np.allclose(distributions_old, distributions_new))

    def test_chain_compatibility(self):
        # define the sequence distribution
        M = np.array([
            [.1, .6, .3],
            [.1, .1, .8],
            [.8, .1, .1]])
        T = TransitionMatrix.MatrixTransitionObject(M)
        nstates = T.get_nstates()
        nsteps = 5
        # create the chain object
        chain = DiscreteEndpoint.Chain(T)
        # create the internal memory model object to be tested
        hidden_states = [FastHMM.FixedState(i) for i in range(nstates)]
        model = InternalModel(T, hidden_states)
        # compare the matrices defining the transition and state expectations
        for i_state, e_state in itertools.product(range(nstates), repeat=2):
            sequence = [i_state] + [None]*(nsteps-1) + [e_state]
            chain_dp_info = chain.get_dp_info(i_state, e_state, nsteps)
            model_dp_info = model.get_dp_info(sequence)
            # compute the chain and model transition expectations
            A_chain = chain.get_transition_expectations(chain_dp_info)
            A_model = model.get_transition_expectations(model_dp_info)
            # Assert that for each method the sum of the expectations
            # is equal to the number of steps.
            self.assertAlmostEqual(np.sum(A_chain), nsteps)
            self.assertAlmostEqual(np.sum(A_model), nsteps)
            # assert that the methods give the same result
            self.assertTrue(np.allclose(A_chain, A_model))
            # Compute the chain
            # and model state expectations for the missing states.
            v_chain = chain.get_expectations(chain_dp_info)
            v_model = sum(np.array(v)
                    for v in model.posterior(model_dp_info)[1:-1])
            # Assert that for each method the sum of the expectations
            # is equal to the number of steps minus one.
            self.assertAlmostEqual(np.sum(v_chain), nsteps-1)
            self.assertAlmostEqual(np.sum(v_model), nsteps-1)
            # assert that the methods give the same result
            self.assertTrue(np.allclose(v_chain, v_model))

    def test_inequality(self):
        """
        Compare two posterior distributions.
        The hidden state for the unobserved coin is less likely to be fair
        in the first case.
        """
        # define the dishonest casino model
        fair_state = HMM.HiddenDieState(1/6.0)
        loaded_state = HMM.HiddenDieState(0.5)
        M = np.array([[0.95, 0.05], [0.1, 0.9]])
        T = TransitionMatrix.MatrixTransitionObject(M)
        hidden_states = [fair_state, loaded_state]
        # create the hidden markov model object
        hmm_new = InternalModel(T, hidden_states)
        # define a sequence of observations
        observations_a = [1, 6, 6, None, 6, 2, 3, 4, 5, 1]
        observations_b = [1, 6, 6, 6, 6, 2, 3, 4, None, 1]
        # get posterior distributions
        distributions_a = hmm_new.posterior(hmm_new.get_dp_info(observations_a))
        distributions_b = hmm_new.posterior(hmm_new.get_dp_info(observations_b))
        # Compare the posterior probability that the die was fair
        # at each interesting position.
        p_fair_a = distributions_a[3][0]
        p_fair_b = distributions_b[-2][0]
        self.assertTrue(p_fair_a < p_fair_b)
        self.assertNotAlmostEqual(p_fair_a, p_fair_b)

    def test_external_string_model_compatibility(self):
        """
        Test StringIO streams for dynamic programming.
        """
        # define the dishonest casino model
        fair_state = HMM.HiddenDieState(1/6.0)
        loaded_state = HMM.HiddenDieState(0.5)
        M = np.array([[0.95, 0.05], [0.1, 0.9]])
        T = TransitionMatrix.MatrixTransitionObject(M)
        hidden_states = [fair_state, loaded_state]
        # define a sequence of observations
        observations = [1, 2, 6, 6, 1, 2, 3, 4, 5, 6]
        # define the observation stream
        o_converter = lineario.IntConverter()
        o_stream = lineario.SequentialStringIO(o_converter)
        o_stream.open_write()
        for x in observations:
            o_stream.write(x)
        o_stream.close()
        # create the reference hidden markov model object
        hmm_old = HMM.TrainedModel(M, hidden_states)
        # create the testing hidden markov model object
        hmm_new = ExternalModel(T, hidden_states, (None, None, None))
        # get posterior distributions
        distributions_old = hmm_old.scaled_posterior_durbin(observations)
        hmm_new.init_dp(o_stream)
        distributions_new = list(hmm_new.posterior())
        # assert that the distributions are the same
        self.assertTrue(np.allclose(distributions_old, distributions_new))

    def test_external_file_model_compatibility(self):
        """
        Test StringIO streams for dynamic programming.
        """
        # define the dishonest casino model
        fair_state = HMM.HiddenDieState(1/6.0)
        loaded_state = HMM.HiddenDieState(0.5)
        M = np.array([[0.95, 0.05], [0.1, 0.9]])
        T = TransitionMatrix.MatrixTransitionObject(M)
        hidden_states = [fair_state, loaded_state]
        # define a sequence of observations
        observations = [1, 2, 6, 6, 1, 2, 3, 4, 5, 6]
        # define the observation stream
        o_converter = lineario.IntConverter()
        o_stream = lineario.SequentialStringIO(o_converter)
        o_stream.open_write()
        for x in observations:
            o_stream.write(x)
        o_stream.close()
        # create the reference hidden markov model object
        hmm_old = HMM.TrainedModel(M, hidden_states)
        # create the testing hidden markov model object
        names = ('tmp_f.tmp', 'tmp_s.tmp', 'tmp_b.tmp')
        hmm_new = ExternalModel(T, hidden_states, names)
        # get posterior distributions
        distributions_old = hmm_old.scaled_posterior_durbin(observations)
        hmm_new.init_dp(o_stream)
        distributions_new = list(hmm_new.posterior())
        # assert that the distributions are the same
        self.assertTrue(np.allclose(distributions_old, distributions_new))


if __name__ == '__main__':
    unittest.main()
*/
