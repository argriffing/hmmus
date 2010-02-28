#include <stdio.h>
#include <stdlib.h>
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

int forward(const struct TM *ptm, FILE *fin_l, FILE *fout_f, FILE *fout_s)
{
  /*
   * Run the forward algorithm.
   * @param ptm: address of a transition matrix
   * @param fin_l: file of likelihood vectors
   * @param fout_f: file of forward vectors
   * @param fout_s: file of scaling factors
   */
  int nstates = ptm->nstates;
  double *f_tmp;
  double *f_curr = malloc(nstates*sizeof(double));
  double *f_prev = malloc(nstates*sizeof(double));
  if (!f_curr || !f_prev)
  {
    fprintf(stderr, "failed to allocate an array\n");
    return -1;
  }
  double p;
  double tprob;
  double scaling_factor;
  int isource, isink;
  int64_t pos=0;
  while (fread(f_curr, sizeof(double), nstates, fin_l))
  {
    /* create the unscaled forward vector */
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
    /* scale the forward vector */
    scaling_factor = sum(f_curr, nstates);
    if (scaling_factor == 0.0)
    {
      fprintf(stderr, "scaling factor 0.0 at pos %lld\n", pos);
      return -1;
    }
    for (isink=0; isink<nstates; isink++)
    {
      f_curr[isink] /= scaling_factor;
    }
    fwrite(f_curr, sizeof(double), nstates, fout_f);
    fwrite(&scaling_factor, sizeof(double), 1, fout_s);
    pos++;
    /* swap f_prev and f_curr */
    f_tmp = f_curr; f_curr = f_prev; f_prev = f_tmp;
  }
  free(f_curr);
  free(f_prev);
  return 0;
}

int backward(const struct TM *ptm, FILE *fin_l, FILE *fin_s, FILE *fout_b)
{
  /*
   * @param ptm: pointer to the transition matrix struct
   * @param fin_l: file of the likelihood vectors open for reading
   * @param fin_s: file of scaling vectors open for reading
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
  int isource, isink;
  int i;
  double scaling_factor;
  double p;
  int64_t pos = 0;
  do
  {
    /* read the likelihood vector and the scaling vector */
    nbytes = fread(l_curr, sizeof(double), nstates, fin_l);
    nbytes = fread(&scaling_factor, sizeof(double), 1, fin_s);
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

int posterior(int nstates, FILE *fi_f, FILE *fi_s, FILE *fi_b, FILE *fo_d)
{
  /*
   * @param nstates: the number of hidden states
   * @param fi_f: file of forward vectors open for reading
   * @param fi_s: file of scaling vectors open for reading
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

int fwdbwd_somedisk(const struct TM *ptm, FILE *fin_l, FILE *fout_d)
{
}

int do_fwdbwd_somedisk(const struct TM *ptm,
    const char *likelihoods_name, const char *posterior_name)
{
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
  forward(ptm, fin_l, fout_f, fout_s);
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
    fprintf(stderr, "failed to open the scaling vector file for reading\n");
    errcode = -1; goto end;
  }
  if (!(fout_b = fopen(backward_name, "wb")))
  {
    fprintf(stderr, "failed to open the backward vector file for writing\n");
    errcode = -1; goto end;
  }
  backward(ptm, fin_l, fin_s, fout_b);
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
    fprintf(stderr, "failed to open the scaling vector file for reading\n");
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
  posterior(nstates, fin_f, fin_s, fin_b, fout_d);
end:
  fsafeclose(fin_f);
  fsafeclose(fin_s);
  fsafeclose(fin_b);
  fsafeclose(fout_d);
  return errcode;
}
