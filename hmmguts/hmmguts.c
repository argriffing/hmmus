#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>
#include <sys/stat.h>

#include "hmmguts.h"

int fsafeclose(FILE *f)
{
  if (f)
  {
    return fclose(f);
  } else {
    return 0;
  }
}

double* get_doubles(int ndoubles, const char *filename)
{
  /*
   * Read some double precision floating point numbers from a binary file.
   */
  struct stat buf;
  if (stat(filename, &buf) < 0)
  {
    fprintf(stderr, "the file %s was not found\n", filename);
    return NULL;
  }
  if (buf.st_size != ndoubles*sizeof(double))
  {
    fprintf(stderr, "unexpected number of bytes ");
    fprintf(stderr, "in the file %s\n", filename);
    return NULL;
  }
  FILE *fin = fopen(filename, "rb");
  if (!fin)
  {
    fprintf(stderr, "failed to open the file ");
    fprintf(stderr, "%s for reading\n", filename);
    return NULL;
  }
  double *arr = malloc(ndoubles*sizeof(double));
  size_t nbytes = fread(arr, sizeof(double), ndoubles, fin);
  fclose(fin);
  if (!nbytes)
  {
    fprintf(stderr, "no bytes read from %s\n", filename);
    return NULL;
  }
  return arr;
}

int TM_init(struct TM *p, int nstates)
{
  p->nstates = nstates;
  p->distn = NULL;
  p->trans = NULL;
  if (nstates <= 0)
  {
    fprintf(stderr, "invalid transition matrix size\n");
    return -1;
  }
  p->distn = malloc(nstates*sizeof(double));
  p->trans = malloc(nstates*nstates*sizeof(double));
  return 0;
}

int TM_init_from_names(struct TM *p,
    const char *distn_name, const char *trans_name)
{
  p->nstates = 0;
  p->trans = NULL;
  p->distn = NULL;
  struct stat buf;
  if (stat(distn_name, &buf) < 0)
  {
    fprintf(stderr, "the file %s was not found\n", distn_name);
    goto fail;
  }
  if (buf.st_size % sizeof(double) != 0)
  {
    fprintf(stderr, "%s should have double precision floats\n", distn_name);
    goto fail;
  }
  p->nstates = buf.st_size / sizeof(double);
  int nstates_squared = p->nstates * p->nstates;
  if (stat(trans_name, &buf) < 0)
  {
    fprintf(stderr, "the file %s was not found\n", trans_name);
    goto fail;
  }
  if (buf.st_size != nstates_squared * sizeof(double))
  {
    fprintf(stderr, "the transition matrix is incompatible\n");
    goto fail;
  }
  if (!(p->distn = get_doubles(p->nstates, distn_name))) goto fail;
  if (!(p->trans = get_doubles(nstates_squared, trans_name))) goto fail;
  return 0;
fail:
  TM_del(p);
  return -1;
}

int TM_del(struct TM *p)
{
  p->nstates = 0;
  free(p->distn); p->distn = NULL;
  free(p->trans); p->trans = NULL;
  return 0;
}

double sum(double *v, int n)
{
  int i;
  double total=0;
  for (i=0; i<n; i++) total += v[i];
  return total;
}

int forward_innerloop(size_t pos, const struct TM *ptm,
    const double *l_curr, const double *f_prev, double *f_curr, double *s_curr)
{
  double scaling_factor;
  double p;
  double tprob;
  int isource, isink;
  int nstates = ptm->nstates;
  memcpy(f_curr, l_curr, nstates*sizeof(double));
  if (pos>0)
  {
    for (isink=0; isink<nstates; isink++)
    {
      p = 0.0;
      for (isource=0; isource<nstates; isource++)
      {
        tprob = ptm->trans[isource*nstates + isink];
        p += f_prev[isource] * tprob;
      }
      f_curr[isink] *= p;
    }
  } else {
    for (isink=0; isink<nstates; isink++)
    {
      f_curr[isink] *= ptm->distn[isink];
    }
  }
  scaling_factor = sum(f_curr, nstates);
  if (scaling_factor == 0.0)
  {
    fprintf(stderr, "scaling factor 0.0 at pos %zd\n", pos);
    return -1;
  }
  for (isink=0; isink<nstates; isink++)
  {
    f_curr[isink] /= scaling_factor;
  }
  *s_curr = scaling_factor;
  return 0;
}

int backward_innerloop(size_t pos, const struct TM *ptm,
    const double *l_prev, const double *l_curr,
    double scaling_factor, const double *b_prev, double *b_curr)
{
  int nstates = ptm->nstates;
  int isource, isink;
  int i;
  double p;
  if (pos)
  {
    for (i=0; i<nstates; i++) b_curr[i] = 0.0;
    for (isource=0; isource<nstates; isource++)
    {
      for (isink=0; isink<nstates; isink++)
      {
        p = ptm->trans[isource*nstates + isink];
        p *= l_prev[isink] * b_prev[isink];
        b_curr[isource] += p;
      }
    }
  } else {
    for (i=0; i<nstates; i++) b_curr[i] = 1.0;
  }
  for (isource=0; isource<nstates; isource++)
  {
    b_curr[isource] /= scaling_factor;
  }
  return 0;
}

int forward_alldisk(const struct TM *ptm,
    FILE *fin_l, FILE *fout_f, FILE *fout_s)
{
  /*
   * Run the forward algorithm.
   * @param ptm: address of a transition matrix
   * @param fin_l: file of likelihood vectors
   * @param fout_f: file of forward vectors
   * @param fout_s: file of scaling factors
   */
  int errcode = 0;
  int nstates = ptm->nstates;
  double *f_tmp;
  double *l_curr = malloc(nstates*sizeof(double));
  double *f_curr = malloc(nstates*sizeof(double));
  double *f_prev = malloc(nstates*sizeof(double));
  if (!f_curr || !f_prev || !l_curr)
  {
    fprintf(stderr, "failed to allocate an array\n");
    errcode = -1; goto end;
  }
  double sf;
  size_t pos = 0;
  while (fread(l_curr, sizeof(double), nstates, fin_l))
  {
    if (forward_innerloop(pos, ptm, l_curr, f_prev, f_curr, &sf) < 0)
    {
      errcode = -1; goto end;
    }
    fwrite(f_curr, sizeof(double), nstates, fout_f);
    fwrite(&sf, sizeof(double), 1, fout_s);
    pos++;
    f_tmp = f_curr; f_curr = f_prev; f_prev = f_tmp;
  }
end:
  free(l_curr);
  free(f_curr);
  free(f_prev);
  return errcode;
}

int backward_alldisk(const struct TM *ptm,
    FILE *fin_l, FILE *fin_s, FILE *fout_b)
{
  /*
   * @param ptm: pointer to the transition matrix struct
   * @param fin_l: file of the likelihood vectors open for reading
   * @param fin_s: file of scaling factors open for reading
   * @param fout_b: file of backward vectors open for writing
   */
  int nstates = ptm->nstates;
  /* seek to near the end of the likelihood and scaling files */
  int result = 0;
  result |= fseek(fin_l, -nstates*sizeof(double), SEEK_END);
  result |= fseek(fin_s, -sizeof(double), SEEK_END);
  if (result)
  {
    fprintf(stderr, "seek error\n");
    return -1;
  }
  /* read the likelihood and scaling files in reverse */
  size_t nbytes;
  double *ptmp;
  double *b_curr = malloc(nstates*sizeof(double));
  double *b_prev = malloc(nstates*sizeof(double));
  double *l_curr = malloc(nstates*sizeof(double));
  double *l_prev = malloc(nstates*sizeof(double));
  double sf;
  size_t pos = 0;
  do
  {
    /* read the likelihood vector and the scaling factor */
    nbytes = fread(l_curr, sizeof(double), nstates, fin_l);
    nbytes = fread(&sf, sizeof(double), 1, fin_s);
    /* do the inner loop */
    backward_innerloop(pos, ptm, l_prev, l_curr, sf, b_prev, b_curr);
    /* write the vector */
    fwrite(b_curr, sizeof(double), nstates, fout_b);
    /* swap buffers and increment the position */
    ptmp = b_curr; b_curr = b_prev; b_prev = ptmp;
    ptmp = l_curr; l_curr = l_prev; l_prev = ptmp;
    pos++;
    /* seek back and break if we go too far */
    result = 0;
    result |= fseek(fin_l, -2*nstates*sizeof(double), SEEK_CUR);
    result |= fseek(fin_s, -2*sizeof(double), SEEK_CUR);
  } while (!result);
  /* clean up */
  free(b_curr);
  free(b_prev);
  free(l_curr);
  free(l_prev);
  return 0;
}

int posterior_alldisk(int nstates,
    FILE *fi_f, FILE *fi_s, FILE *fi_b, FILE *fo_d)
{
  /*
   * @param nstates: the number of hidden states
   * @param fi_f: file of forward vectors open for reading
   * @param fi_s: file of scaling factors open for reading
   * @param fi_b: file of backward vectors open for reading
   * @param fo_d: file of posterior probability vectors open for writing
   */
  size_t nbytes;
  int result;
  /* seek to near the end of the backward file */
  result = fseek(fi_b, -nstates*sizeof(double), SEEK_END);
  if (result)
  {
    fprintf(stderr, "seek error\n");
    return -1;
  }
  /* multiply stuff together and write to the output file */
  int i;
  double scaling_factor;
  double *arr_f = malloc(nstates*sizeof(double));
  double *arr_b = malloc(nstates*sizeof(double));
  do
  {
    nbytes = fread(arr_f, sizeof(double), nstates, fi_f);
    nbytes = fread(arr_b, sizeof(double), nstates, fi_b);
    nbytes = fread(&scaling_factor, sizeof(double), 1, fi_s);
    /* construct and write the output vector */
    for (i=0; i<nstates; i++)
    {
      arr_f[i] *= scaling_factor * arr_b[i];
    }
    fwrite(arr_f, sizeof(double), nstates, fo_d);
    /* go back a bit */
    result = fseek(fi_b, -2*nstates*sizeof(double), SEEK_CUR);
  } while (!result);
  /* clean up */
  free(arr_f);
  free(arr_b);
  return 0;
}

int forward_somedisk(const struct TM *ptm, FILE *fin_l,
    double *f_big, double *s_big)
{
  /*
   * Run the forward algorithm.
   * @param ptm: address of a transition matrix
   * @param fin_l: file of likelihood vectors
   * @param f_big: the output array of forward vectors
   * @param s_big: the output array of scaling factors
   */
  int errcode = 0;
  int nstates = ptm->nstates;
  double *l_curr = malloc(nstates*sizeof(double));
  double *s_curr = s_big;
  double *f_curr = f_big;
  double *f_prev = NULL;
  size_t pos=0;
  while (fread(l_curr, sizeof(double), nstates, fin_l))
  {
    if (forward_innerloop(pos, ptm, l_curr, f_prev, f_curr, s_curr) < 0)
    {
      errcode = -1; goto end;
    }
    pos++;
    f_prev = f_curr;
    f_curr += nstates;
    s_curr++;
  }
end:
  free(l_curr);
  return errcode;
}

int backward_somedisk(const struct TM *ptm, size_t nobs, FILE *fin_l,
    const double *s_big, double *b_big)
{
  /*
   * @param ptm: pointer to the transition matrix struct
   * @param nobs: the number of observations
   * @param fin_l: file of the likelihood vectors open for reading
   * @param s_big: scaling factors
   * @param b_big: file of backward vectors open for writing
   * @return: negative on error
   */
  int errcode = 0;
  int nstates = ptm->nstates;
  const double *s_curr = NULL;
  double *b_curr = NULL;
  double *b_prev = NULL;
  double *l_curr = malloc(nstates*sizeof(double));
  double *l_prev = malloc(nstates*sizeof(double));
  /* seek to near the end of the likelihood file */
  if (fseek(fin_l, -nstates*sizeof(double), SEEK_END))
  {
    fprintf(stderr, "seek error\n");
    errcode = -1; goto end;
  }
  /* start at the end of the scaling factor array */
  s_curr = s_big + (nobs - 1);
  /* start at the beginning of the backward array */
  b_curr = b_big;
  size_t nbytes;
  double *ptmp;
  int result;
  size_t pos = 0;
  do
  {
    nbytes = fread(l_curr, sizeof(double), nstates, fin_l);
    backward_innerloop(pos, ptm, l_prev, l_curr, *s_curr, b_prev, b_curr);
    ptmp = l_curr; l_curr = l_prev; l_prev = ptmp;
    b_prev = b_curr;
    b_curr += nstates;
    s_curr--;
    pos++;
    /* seek back and break if we go too far */
    result = fseek(fin_l, -2*nstates*sizeof(double), SEEK_CUR);
  } while (!result);
end:
  free(l_curr);
  free(l_prev);
  return errcode;
}

int posterior_somedisk(int nstates, size_t nobs,
    const double *f_big, const double *s_big, const double *b_big,
    FILE *fout_d)
{
  /*
   * @param nstates: the number of hidden states
   * @param nobs: the number of observations
   * @param f_big: array of forward vectors
   * @param s_big: array of scaling factors
   * @param b_big: file of backward vectors
   * @param fout_d: file of posterior probability vectors open for writing
   */
  size_t pos;
  double posterior;
  int i;
  const double *f_curr = f_big;
  const double *s_curr = s_big;
  const double *b_curr = b_big + nstates * (nobs - 1);
  /* multiply stuff together and write to the output file */
  for (pos=0; pos<nobs; pos++)
  {
    for (i=0; i<nstates; i++)
    {
      posterior = f_curr[i] * s_curr[0] * b_curr[i];
      fwrite(&posterior, sizeof(double), 1, fout_d);
    }
    f_curr += nstates;
    s_curr++;
    b_curr -= nstates;
  }
  return 0;
}

int fwdbwd_somedisk(const struct TM *ptm, size_t nobs,
    FILE *fin_l, FILE *fout_d)
{
  int errcode = 0;
  int nstates = ptm->nstates;
  /* allocate the big forward, scaling, and backward arrays */
  double *f_big = malloc(nobs*nstates*sizeof(double));
  double *s_big = malloc(nobs*sizeof(double));
  double *b_big = malloc(nobs*nstates*sizeof(double));
  /* make sure that the arrays were allocated */
  if (!f_big || !s_big || !b_big)
  {
    fprintf(stderr, "failed to allocate an array\n");
    errcode = -1; goto end;
  }
  /* run the forward and backward algorithms */
  if (forward_somedisk(ptm, fin_l, f_big, s_big) < 0) {
    errcode = -1; goto end;
  }
  if (backward_somedisk(ptm, nobs, fin_l, s_big, b_big) < 0) {
    errcode = -1; goto end;
  }
  if (posterior_somedisk(nstates, nobs, f_big, s_big, b_big, fout_d) < 0) {
    errcode = -1; goto end;
  }
end:
  free(f_big);
  free(s_big);
  free(b_big);
  return errcode;
}

int forward_nodisk(const struct TM *ptm, size_t nobs,
    const double *l_big,
    double *f_big, double *s_big)
{
  /*
   * Run the forward algorithm.
   * @param ptm: address of a transition matrix
   * @param nobs: the number of observations
   * @param l_big: the input array of likelihood vectors
   * @param f_big: the output array of forward vectors
   * @param s_big: the output array of scaling factors
   */
  int errcode = 0;
  int nstates = ptm->nstates;
  double *s_curr = s_big;
  double *f_curr = f_big;
  double *f_prev = NULL;
  const double *l_curr = l_big;
  size_t pos;
  for (pos=0; pos<nobs; pos++)
  {
    if (forward_innerloop(pos, ptm, l_curr, f_prev, f_curr, s_curr) < 0)
    {
      errcode = -1; goto end;
    }
    f_prev = f_curr;
    f_curr += nstates;
    s_curr++;
    l_curr += nstates;
  }
end:
  return errcode;
}

int backward_nodisk(const struct TM *ptm, size_t nobs,
    const double *l_big,
    const double *s_big, double *b_big)
{
  /*
   * @param ptm: pointer to the transition matrix struct
   * @param nobs: the number of observations
   * @param l_big: likelihood vectors
   * @param s_big: scaling factors
   * @param b_big: backward vectors
   * @return: negative on error
   */
  int nstates = ptm->nstates;
  const double *s_curr = s_big + (nobs - 1);
  const double *l_curr = l_big + nstates * (nobs - 1);
  const double *l_prev = NULL;
  double *b_curr = b_big;
  double *b_prev = NULL;
  size_t pos;
  for (pos=0; pos<nobs; pos++)
  {
    backward_innerloop(pos, ptm, l_prev, l_curr, *s_curr, b_prev, b_curr);
    l_prev = l_curr;
    l_curr -= nstates;
    b_prev = b_curr;
    b_curr += nstates;
    s_curr--;
  }
  return 0;
}

int posterior_nodisk(int nstates, size_t nobs,
    const double *f_big, const double *s_big, const double *b_big,
    double *d_big)
{
  /*
   * @param nstates: the number of hidden states
   * @param nobs: the number of observations
   * @param f_big: array of forward vectors
   * @param s_big: array of scaling factors
   * @param b_big: array of backward vectors
   * @param d_big: array of posterior vectors
   */
  size_t pos;
  int i;
  const double *f_curr = f_big;
  const double *s_curr = s_big;
  const double *b_curr = b_big + nstates * (nobs - 1);
  double *d_curr = d_big;
  /* multiply stuff together and write to the output file */
  for (pos=0; pos<nobs; pos++)
  {
    for (i=0; i<nstates; i++)
    {
      d_curr[i] = f_curr[i] * s_curr[0] * b_curr[i];
    }
    d_curr += nstates;
    f_curr += nstates;
    s_curr++;
    b_curr -= nstates;
  }
  return 0;
}

int fwdbwd_nodisk(const struct TM *ptm, size_t nobs,
    const double *l_big, double *d_big)
{
  int errcode = 0;
  int nstates = ptm->nstates;
  /* allocate the big forward, scaling, and backward arrays */
  double *f_big = malloc(nobs*nstates*sizeof(double));
  double *s_big = malloc(nobs*sizeof(double));
  double *b_big = malloc(nobs*nstates*sizeof(double));
  /* make sure that the arrays were allocated */
  if (!f_big || !s_big || !b_big)
  {
    fprintf(stderr, "failed to allocate an array\n");
    errcode = -1; goto end;
  }
  /* run the forward and backward algorithms */
  if (forward_nodisk(ptm, nobs, l_big, f_big, s_big) < 0) {
    errcode = -1; goto end;
  }
  if (backward_nodisk(ptm, nobs, l_big, s_big, b_big) < 0) {
    errcode = -1; goto end;
  }
  if (posterior_nodisk(nstates, nobs, f_big, s_big, b_big, d_big) < 0) {
    errcode = -1; goto end;
  }
end:
  free(f_big);
  free(s_big);
  free(b_big);
  return errcode;
}


int do_fwdbwd_somedisk(const struct TM *ptm,
    const char *likelihoods_name, const char *posterior_name)
{
  int nbytes_per_pos = ptm->nstates * sizeof(double);
  int errcode = 0;
  size_t nobs = 0;
  FILE *fin_l = NULL;
  FILE *fout_d = NULL;
  struct stat buf;
  if (stat(likelihoods_name, &buf) < 0)
  {
    fprintf(stderr, "the file %s was not found\n", likelihoods_name);
    errcode = -1; goto end;
  }
  if (buf.st_size % nbytes_per_pos)
  {
    fprintf(stderr, "%s should have %d double precision floats per position\n",
        likelihoods_name, ptm->nstates);
    errcode = -1; goto end;
  }
  if (!(fin_l = fopen(likelihoods_name, "rb")))
  {
    fprintf(stderr, "failed to open the likelihoods file for reading\n");
    errcode = -1; goto end;
  }
  if (!(fout_d = fopen(posterior_name, "wb")))
  {
    fprintf(stderr, "failed to open the posterior file for writing\n");
    errcode = -1; goto end;
  }
  nobs = buf.st_size / nbytes_per_pos;
  fwdbwd_somedisk(ptm, nobs, fin_l, fout_d);
end:
  fsafeclose(fin_l);
  fsafeclose(fout_d);
  return errcode;
}

int do_forward(const struct TM *ptm,
    const char *likelihoods_name, const char *forward_name,
    const char *scaling_name)
{
  int errcode = 0;
  FILE *fin_l = NULL;
  FILE *fout_f = NULL;
  FILE *fout_s = NULL;
  if (!(fin_l = fopen(likelihoods_name, "rb")))
  {
    fprintf(stderr, "failed to open the likelihoods file for reading\n");
    errcode = -1; goto end;
  }
  if (!(fout_f = fopen(forward_name, "wb")))
  {
    fprintf(stderr, "failed to open the forward vector file for writing\n");
    errcode = -1; goto end;
  }
  if (!(fout_s = fopen(scaling_name, "wb")))
  {
    fprintf(stderr, "failed to open the scaling factor file for writing\n");
    errcode = -1; goto end;
  }
  forward_alldisk(ptm, fin_l, fout_f, fout_s);
end:
  fsafeclose(fin_l);
  fsafeclose(fout_f);
  fsafeclose(fout_s);
  return errcode;
}

int do_backward(const struct TM *ptm,
    const char *likelihoods_name, const char *scaling_name,
    const char *backward_name)
{
  int errcode = 0;
  FILE *fin_l = NULL;
  FILE *fin_s = NULL;
  FILE *fout_b = NULL;
  if (!(fin_l = fopen(likelihoods_name, "rb")))
  {
    fprintf(stderr, "failed to open the likelihoods file for reading\n");
    errcode = -1; goto end;
  }
  if (!(fin_s = fopen(scaling_name, "rb")))
  {
    fprintf(stderr, "failed to open the scaling factor file for reading\n");
    errcode = -1; goto end;
  }
  if (!(fout_b = fopen(backward_name, "wb")))
  {
    fprintf(stderr, "failed to open the backward vector file for writing\n");
    errcode = -1; goto end;
  }
  backward_alldisk(ptm, fin_l, fin_s, fout_b);
end:
  fsafeclose(fin_l);
  fsafeclose(fin_s);
  fsafeclose(fout_b);
  return errcode;
}

int do_posterior(int nstates,
    const char *forward_name, const char *scaling_name,
    const char *backward_name, const char *posterior_name)
{
  int errcode = 0;
  FILE *fin_f = NULL;
  FILE *fin_s = NULL;
  FILE *fin_b = NULL;
  FILE *fout_d = NULL;
  if (!(fin_f = fopen(forward_name, "rb")))
  {
    fprintf(stderr, "failed to open the forward vector file for reading\n");
    errcode = -1; goto end;
  }
  if (!(fin_s = fopen(scaling_name, "rb")))
  {
    fprintf(stderr, "failed to open the scaling factor file for reading\n");
    errcode = -1; goto end;
  }
  if (!(fin_b = fopen(backward_name, "rb")))
  {
    fprintf(stderr, "failed to open the backward vector file for reading\n");
    errcode = -1; goto end;
  }
  if (!(fout_d = fopen(posterior_name, "wb")))
  {
    fprintf(stderr, "failed to open the posterior vector file for writing\n");
    errcode = -1; goto end;
  }
  posterior_alldisk(nstates, fin_f, fin_s, fin_b, fout_d);
end:
  fsafeclose(fin_f);
  fsafeclose(fin_s);
  fsafeclose(fin_b);
  fsafeclose(fout_d);
  return errcode;
}
