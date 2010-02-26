#include <stdio.h>
#include <stdlib.h>

#include "hmmguts/hmmguts.h"

int main(int argc, char* argv[])
{
  struct TM tm;
  int nstates=3;
  TM_init_from_names(&tm, nstates, "distribution.bin", "transitions.bin");
  do_forward(&tm, "likelihoods.bin", "test.forward", "test.scaling");
  do_backward(&tm, "likelihoods.bin", "test.scaling", "test.backward");
  do_posterior(&tm, "test.forward", "test.scaling", "test.backward",
      "test.posterior");
  TM_del(&tm);
  return 0;
}
