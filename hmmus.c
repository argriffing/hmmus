#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>
#include <sys/stat.h>

struct TM
{
  /*
   * a transition matrix
   * @member order: number of states
   * @member value: row major matrix elements
   * @member initial_distn: initial distribution of states
   */
  int order;
  double *value;
  double *initial_distn;
};

int TM_init(struct TM *p, int order)
{
  /* constructor */
  p->order = order;
  p->value = malloc(order*order*sizeof(double));
  p->initial_distn = malloc(order*sizeof(double));
  return 0;
}

int TM_del(struct TM *p)
{
  /* destructor */
  free(p->value);
  free(p->initial_distn);
  return 0;
}

double sum(double *v, int n)
{
  int i;
  double total=0;
  for (i=0; i<n; i++) total += v[i];
  return total;
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
  int nstates = ptm->order;
  double *f_tmp;
  double *f_curr = malloc(nstates*sizeof(double));
  double *f_prev = malloc(nstates*sizeof(double));
  if (!f_curr || !f_prev)
  {
    fprintf(stderr, "failed to allocate an array\n");
    return 1;
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
          tprob = ptm->value[isource*nstates + isink];
          p += f_prev[isource] * tprob;
        }
        f_curr[isink] *= p;
      }
    } else {
      for (isink=0; isink<nstates; isink++)
      {
        f_curr[isink] *= ptm->initial_distn[isink];
      }
    }
    /* scale the forward vector */
    scaling_factor = sum(f_curr, nstates);
    if (scaling_factor == 0.0)
    {
      fprintf(stderr, "scaling factor 0.0 at pos %lld\n", pos);
      return 1;
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

int backward(struct TM *ptm, FILE *fin_l, FILE *fin_s, FILE *fout_b)
{
  /*
   * @param ptm: pointer to the transition matrix struct
   * @param fin_l: file of the likelihood vectors open for reading
   * @param fin_s: file of scaling vectors open for reading
   * @param fout_b: file of backward vectors open for writing
   */
  int nhidden = ptm->order;
  /* seek to near the end of the likelihood and scaling files */
  int result = 0;
  result |= fseek(fin_l, -nhidden*sizeof(double), SEEK_END);
  result |= fseek(fin_s, -sizeof(double), SEEK_END);
  if (result)
  {
    fprintf(stderr, "seek error\n");
    return 1;
  }
  /* read the likelihood and scaling files in reverse */
  double *ptmp;
  double *b_curr = malloc(nhidden*sizeof(double));
  double *b_prev = malloc(nhidden*sizeof(double));
  double *l_curr = malloc(nhidden*sizeof(double));
  double *l_prev = malloc(nhidden*sizeof(double));
  int isource, isink;
  int i;
  double scaling_factor;
  double p;
  int64_t pos = 0;
  do
  {
    /* read the likelihood vector and the scaling vector */
    fread(l_curr, sizeof(double), nhidden, fin_l);
    fread(&scaling_factor, sizeof(double), 1, fin_s);
    if (pos)
    {
      for (i=0; i<nhidden; i++) b_curr[i] = 0.0;
      for (isource=0; isource<nhidden; isource++)
      {
        for (isink=0; isink<nhidden; isink++)
        {
          p = ptm->value[isource*nhidden + isink];
          p *= l_prev[isink] * b_prev[isink];
          b_curr[isource] += p;
        }
      }
    } else {
      for (i=0; i<nhidden; i++) b_curr[i] = 1.0;
    }
    for (isource=0; isource<nhidden; isource++)
    {
      b_curr[isource] /= scaling_factor;
    }
    /* write the vector */
    fwrite(b_curr, sizeof(double), nhidden, fout_b);
    /* swap buffers and increment the position */
    ptmp = b_curr; b_curr = b_prev; b_prev = ptmp;
    ptmp = l_curr; l_curr = l_prev; l_prev = ptmp;
    pos++;
    /* seek back and break if we go too far */
    result = 0;
    result |= fseek(fin_l, -2*nhidden*sizeof(double), SEEK_CUR);
    result |= fseek(fin_s, -2*sizeof(double), SEEK_CUR);
  } while (!result);
  /* clean up */
  free(b_curr);
  free(b_prev);
  free(l_curr);
  free(l_prev);
}

int posterior(struct TM *ptm, FILE *fi_f, FILE *fi_s, FILE *fi_b, FILE *fo_d)
{
  /*
   * @param ptm: pointer to the transition matrix struct
   * @param fi_f: file of forward vectors open for reading
   * @param fi_s: file of scaling vectors open for reading
   * @param fi_b: file of backward vectors open for reading
   * @param fo_d: file of posterior probability vectors open for writing
   */
  int nhidden = ptm->order;
  int result;
  /* seek to near the end of the backward file */
  result = fseek(fi_b, -nhidden*sizeof(double), SEEK_END);
  if (result)
  {
    fprintf(stderr, "seek error\n");
    return 1;
  }
  /* multiply stuff together and write to the output file */
  int i;
  double scaling_factor;
  double *forward = malloc(nhidden*sizeof(double));
  double *backward = malloc(nhidden*sizeof(double));
  do
  {
    fread(forward, sizeof(double), nhidden, fi_f);
    fread(backward, sizeof(double), nhidden, fi_b);
    fread(&scaling_factor, sizeof(double), 1, fi_s);
    /* construct and write the output vector */
    for (i=0; i<nhidden; i++)
    {
      forward[i] *= scaling_factor * backward[i];
    }
    fwrite(forward, sizeof(double), nhidden, fo_d);
    /* go back a bit */
    result = fseek(fi_b, -2*nhidden*sizeof(double), SEEK_CUR);
  } while (!result);
  /* clean up */
  free(forward);
  free(backward);
}

double* get_doubles(int ndoubles, const char *filename)
{
  /*
   * Read some double precision floating point numbers from a binary file.
   */
  FILE *fin = fopen(filename, "rb");
  if (!fin)
  {
    fprintf(stderr, "failed to open the file ");
    fprintf(stderr, "%s for reading", filename);
    return NULL;
  }
  struct stat buf;
  stat(filename, &buf);
  if (buf.st_size != ndoubles*sizeof(double))
  {
    fprintf(stderr, "unexpected number of bytes ");
    fprintf(stderr, "in the file %s", filename);
    return NULL;
  }
  double *arr = malloc(ndoubles*sizeof(double));
  fread(arr, sizeof(double), ndoubles, fin);
  fclose(fin);
  return arr;
}

int main(int argc, char* argv[])
{
  FILE *fin_l, *fin_f, *fin_s, *fin_b;
  FILE *fout_f, *fout_s, *fout_b, *fout_d;
  int nstates=2;
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
  /* run the forward algorithm */
  fin_l = fopen("likelihoods.bin", "rb");
  if (!fin_l)
  {
    fprintf(stderr, "failed to open the likelihoods file for reading\n");
    return 1;
  }
  fout_f = fopen("test.forward", "wb");
  if (!fout_f)
  {
    fprintf(stderr, "failed to open the forward vector file for writing\n");
    return 1;
  }
  fout_s = fopen("test.scaling", "wb");
  if (!fout_s)
  {
    fprintf(stderr, "failed to open the scaling factor file for writing\n");
    return 1;
  }
  forward(&tm, fin_l, fout_f, fout_s);
  fclose(fin_l);
  fclose(fout_f);
  fclose(fout_s);
  /* run the backward algorithm */
  fin_l = fopen("likelihoods.bin", "rb");
  if (!fin_l)
  {
    fprintf(stderr, "failed to open the likelihoods file for reading\n");
    return 1;
  }
  fin_s = fopen("test.scaling", "rb");
  if (!fin_s)
  {
    fprintf(stderr, "failed to open the scaling vector file for reading\n");
    return 1;
  }
  fout_b = fopen("test.backward", "wb");
  if (!fout_b)
  {
    fprintf(stderr, "failed to open the backward vector file for writing\n");
    return 1;
  }
  backward(&tm, fin_l, fin_s, fout_b);
  fclose(fin_l);
  fclose(fin_s);
  fclose(fout_b);
  /* run the posterior decoding algorithm */
  fin_f = fopen("test.forward", "rb");
  if (!fin_f)
  {
    fprintf(stderr, "failed to open the forward vector file for reading\n");
    return 1;
  }
  fin_s = fopen("test.scaling", "rb");
  if (!fin_s)
  {
    fprintf(stderr, "failed to open the scaling vector file for reading\n");
    return 1;
  }
  fin_b = fopen("test.backward", "rb");
  if (!fin_b)
  {
    fprintf(stderr, "failed to open the backward vector file for reading\n");
    return 1;
  }
  fout_d = fopen("test.posterior", "wb");
  if (!fout_d)
  {
    fprintf(stderr, "failed to open the posterior vector file for writing\n");
    return 1;
  }
  posterior(&tm, fin_f, fin_s, fin_b, fout_d);
  fclose(fin_f);
  fclose(fin_s);
  fclose(fin_b);
  fclose(fout_d);
  /* clean up */
  TM_del(&tm);
  return 0;
}
