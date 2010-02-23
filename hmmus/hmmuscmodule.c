#include <Python.h>

#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>
#include <sys/stat.h>

#include "notpythonc.h"

double* get_doubles(int ndoubles, const char *filename)
{
  /*
   * Read some double precision floating point numbers from a binary file.
   */
  struct stat buf;
  int result = stat(filename, &buf);
  if (result < 0)
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

int TM_init_from_names(struct TM *p, int nstates,
    const char *distribution_name, const char *transitions_name)
{
  double *distribution = get_doubles(nstates, "distribution.bin");
  if (!distribution) return 1;
  double *transitions = get_doubles(nstates*nstates, "transitions.bin");
  if (!transitions) return 1;
  p->order = nstates;
  p->value = transitions;
  p->initial_distn = distribution;
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
  size_t nbytes;
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
    nbytes = fread(l_curr, sizeof(double), nhidden, fin_l);
    nbytes = fread(&scaling_factor, sizeof(double), 1, fin_s);
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
  return 0;
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
  size_t nbytes;
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
  double *arr_f = malloc(nhidden*sizeof(double));
  double *arr_b = malloc(nhidden*sizeof(double));
  do
  {
    nbytes = fread(arr_f, sizeof(double), nhidden, fi_f);
    nbytes = fread(arr_b, sizeof(double), nhidden, fi_b);
    nbytes = fread(&scaling_factor, sizeof(double), 1, fi_s);
    /* construct and write the output vector */
    for (i=0; i<nhidden; i++)
    {
      arr_f[i] *= scaling_factor * arr_b[i];
    }
    fwrite(arr_f, sizeof(double), nhidden, fo_d);
    /* go back a bit */
    result = fseek(fi_b, -2*nhidden*sizeof(double), SEEK_CUR);
  } while (!result);
  /* clean up */
  free(arr_f);
  free(arr_b);
  return 0;
}



int do_forward(struct TM *ptm,
    const char *likelihoods_name, const char *forward_name,
    const char *scaling_name)
{
  FILE *fin_l = fopen(likelihoods_name, "rb");
  if (!fin_l)
  {
    fprintf(stderr, "failed to open the likelihoods file for reading\n");
    return 1;
  }
  FILE *fout_f = fopen(forward_name, "wb");
  if (!fout_f)
  {
    fprintf(stderr, "failed to open the forward vector file for writing\n");
    return 1;
  }
  FILE *fout_s = fopen(scaling_name, "wb");
  if (!fout_s)
  {
    fprintf(stderr, "failed to open the scaling factor file for writing\n");
    return 1;
  }
  forward(ptm, fin_l, fout_f, fout_s);
  fclose(fin_l);
  fclose(fout_f);
  fclose(fout_s);
  return 0;
}

int do_backward(struct TM *ptm,
    const char *likelihoods_name, const char *scaling_name,
    const char *backward_name)
{
  FILE *fin_l = fopen(likelihoods_name, "rb");
  if (!fin_l)
  {
    fprintf(stderr, "failed to open the likelihoods file for reading\n");
    return 1;
  }
  FILE *fin_s = fopen(scaling_name, "rb");
  if (!fin_s)
  {
    fprintf(stderr, "failed to open the scaling vector file for reading\n");
    return 1;
  }
  FILE *fout_b = fopen(backward_name, "wb");
  if (!fout_b)
  {
    fprintf(stderr, "failed to open the backward vector file for writing\n");
    return 1;
  }
  backward(ptm, fin_l, fin_s, fout_b);
  fclose(fin_l);
  fclose(fin_s);
  fclose(fout_b);
  return 0;
}

int do_posterior(struct TM *ptm,
    const char *forward_name, const char *scaling_name,
    const char *backward_name, const char *posterior_name)
{
  FILE *fin_f = fopen(forward_name, "rb");
  if (!fin_f)
  {
    fprintf(stderr, "failed to open the forward vector file for reading\n");
    return 1;
  }
  FILE *fin_s = fopen(scaling_name, "rb");
  if (!fin_s)
  {
    fprintf(stderr, "failed to open the scaling vector file for reading\n");
    return 1;
  }
  FILE *fin_b = fopen(backward_name, "rb");
  if (!fin_b)
  {
    fprintf(stderr, "failed to open the backward vector file for reading\n");
    return 1;
  }
  FILE *fout_d = fopen(posterior_name, "wb");
  if (!fout_d)
  {
    fprintf(stderr, "failed to open the posterior vector file for writing\n");
    return 1;
  }
  posterior(ptm, fin_f, fin_s, fin_b, fout_d);
  fclose(fin_f);
  fclose(fin_s);
  fclose(fin_b);
  fclose(fout_d);
  return 0;
}

int not_main(int argc, char* argv[])
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




/*
 * Define some application specific stuff before defining Python extension stuff.
 */

/*
 * Normalize an angle to the real interval [0, 2pi).
 */
double norm(double theta_in)
{
  double theta = fmod(theta_in, 2*M_PI);
  if (theta < 0)
  {
    theta += 2*M_PI;
  }
  return theta;
}

/*
 * Define an angle interval.
 */
typedef struct Interval {
  double low;
  double high;
} Interval;

/*
 * Modify the current interval by adding another interval.
 * The union of the ranges is assumed to be contiguous and to span less than 2*pi radians.
 * @param mutable: a pointer to an interval that will be modified
 * @param other: a pointer to an overlapping or contiguous interval that will not be modified
 */
void update_interval(Interval *mutable, Interval *other)
{
  /* Try each combination of low and high angles to find the one that gives the widest interval. */
  double low = mutable->low;
  double high = mutable->high;
  double magnitude = norm(high - low);
  double best_magnitude = magnitude;
  double best_low = low;
  double best_high = high;
  low = mutable->low;
  high = other->high;
  magnitude = norm(high - low);
  if (best_magnitude < magnitude)
  {
    best_magnitude = magnitude;
    best_low = low;
    best_high = high;
  }
  low = other->low;
  high = mutable->high;
  magnitude = norm(high - low);
  if (best_magnitude < magnitude)
  {
    best_magnitude = magnitude;
    best_low = low;
    best_high = high;
  }
  low = other->low;
  high = other->high;
  magnitude = norm(high - low);
  if (best_magnitude < magnitude)
  {
    best_magnitude = magnitude;
    best_low = low;
    best_high = high;
  }
  mutable->low = best_low;
  mutable->high = best_high;
}

/*
 * blen: branch length
 * x: x coordinate
 * y: y coordinate
 * id: node identifier
 * nneighbors: neighbor count (includes parent and children)
 * neighbors: neighbors (includes parent and children)
 * parent: one of the neighbors, or NULL if the node is the root
 */
typedef struct Node {
  double blen;
  double x;
  double y;
  long id;
  long nneighbors;
  struct Node **neighbors;
  struct Node *parent;
} Node;

void reroot(Node *new_root)
{
  Node *next_parent = NULL;
  Node *current = new_root;
  while (current->parent)
  {
    Node *parent = current->parent;
    current->parent = next_parent;
    next_parent = current;
    current = parent;
  }
  current->parent = next_parent;
}


/*
 * @param root: the root of the subtree
 * @param nodes: an allocated array of node pointers to be filled, or NULL
 * @param current_node_count: the number of node pointers filled so far
 * @return: the number of node pointers filled so far
 * Note that by setting the nodes parameter to NULL this function
 * can be used to count the number of nodes in the subtree.
 */
long _fill_preorder_nodes(Node *root, Node **nodes, long current_node_count)
{
  if (root)
  {
    if (nodes)
    {
      nodes[current_node_count] = root;
    }
    current_node_count++;
    int i;
    for (i=0; i<root->nneighbors; i++)
    {
      Node *neighbor = root->neighbors[i];
      if (neighbor != root->parent)
      {
        current_node_count = _fill_preorder_nodes(neighbor, nodes, current_node_count);
      }
    }
  }
  return current_node_count;
}

/*
 * @param root: the root of the subtree
 * @param pnodes: a pointer to an array of node pointers
 * @param pcount: a pointer to the length of the array
 * This function allocates and fills the array pointed to by pnodes.
 * The length of the array is recorded in the variable pointed to by pcount.
 */
void get_preorder_nodes(Node *root, Node ***pnodes, long *pcount)
{
  *pcount = _fill_preorder_nodes(root, NULL, 0);
  if (*pcount)
  {
    *pnodes = (Node **) malloc(*pcount * sizeof(Node *));
    _fill_preorder_nodes(root, *pnodes, 0);
  }
}


static PyObject *
hello_world(PyObject *self, PyObject *args)
{
  const char *command;
  int sts = 42;
  if (!PyArg_ParseTuple(args, "s", &command))
    return NULL;
  printf("ohai %s", command);
  return Py_BuildValue("i", sts);
}

static PyObject *
posterior_python(PyObject *self, PyObject *args)
{
  int nstates;
  const char *distn_name;
  const char *transitions_name;
  const char *forward_name;
  const char *scaling_name;
  const char *backward_name;
  const char *posterior_name;
  int ok = PyArg_ParseTuple(args, "issssss", &nstates,
      &distn_name, &transitions_name, &forward_name, &scaling_name,
      &backward_name, &posterior_name);
  if (!ok) return NULL;
  printf("nstates: %d\n", nstates);
  printf("distribution filename: %s\n", distn_name);
  printf("transitions filename: %s\n", transitions_name);
  printf("forward array filename: %s\n", forward_name);
  printf("scaling array filename: %s\n", scaling_name);
  printf("backward array filename: %s\n", backward_name);
  printf("posterior array filename: %s\n", posterior_name);
  return Py_BuildValue("i", 42);
}

static PyObject *
four_times(PyObject *self, PyObject *args)
{
  int value;
  if (!PyArg_ParseTuple(args, "i", &value))
    return NULL;
  return Py_BuildValue("i", twice(value)*2);
}

static PyMethodDef HmmuscMethods[] = {
  {"hello", hello_world, METH_VARARGS, "Say hi."},
  {"four_times", four_times, METH_VARARGS, "Multiply an integer by four."},
  {"posterior", posterior_python, METH_VARARGS, "Posterior decoding."},
  {NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC
inithmmusc(void) 
{
  (void) Py_InitModule("hmmusc", HmmuscMethods);
}

