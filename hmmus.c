#include <stdio.h>
#include <stdlib.h>

#include "hmmguts/hmmguts.h"

int main(int argc, char* argv[])
{
  struct TM tm;
  const char *distn_name = "distribution.bin";
  const char *trans_name = "transitions.bin";
  const char *l_name = "likelihoods.bin";
  const char *f_name = "forward.bin";
  const char *s_name = "scaling.bin";
  const char *b_name = "backward.bin";
  const char *p_name = "posterior.bin";
  int nstates=3;
  if (TM_init_from_names(&tm, nstates, distn_name, trans_name) < 0) goto fail;
  if (do_forward(&tm, l_name, f_name, s_name) < 0) goto fail;
  if (do_backward(&tm, l_name, s_name, b_name) < 0) goto fail;
  if (do_posterior(&tm, f_name, s_name, b_name, p_name) < 0) goto fail;
  return EXIT_SUCCESS;
fail:
  TM_del(&tm);
  return EXIT_FAILURE;
}
