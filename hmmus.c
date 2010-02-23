#include <stdio.h>
#include <stdlib.h>

#include "hmmus/hmmguts/hmmguts.h"

int main(int argc, char* argv[])
{
  int nstates=3;
  /* read the stationary distribution */
  double *distribution = get_doubles(nstates, "distribution.bin");
  if (!distribution) return 1;
  /* read the transition matrix */
  double *transitions = get_doubles(nstates*nstates, "transitions.bin");
  if (!transitions) return 1;
  /* initialize the transition matrix object */
  struct TM tm;
  tm.order = nstates;
  tm.value = transitions;
  tm.initial_distn = distribution;
  do_forward(&tm, "likelihoods.bin", "test.forward", "test.scaling");
  do_backward(&tm, "likelihoods.bin", "test.scaling", "test.backward");
  do_posterior(&tm, "test.forward", "test.scaling", "test.backward",
      "test.posterior");
  /* clean up */
  TM_del(&tm);
  return 0;
}
