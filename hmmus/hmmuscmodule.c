#include <Python.h>

#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>
#include <sys/stat.h>

/* REMOVE */
#include <math.h>

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


/*
 * Define Python extension stuff.
 */


/* REMOVE */
typedef struct {
    PyObject_HEAD
    long myvalue;
    Node *root;
    Node *cursor;
        /* Type-specific fields go here. */
} DayObject;

/* REMOVE */
/* constructor */
static int
Day_init(DayObject *self, PyObject *args, PyObject *kwds)
{
  self->myvalue = 0;
  self->root = NULL;
  self->cursor = NULL;
  return 0;
}

/* REMOVE */
/* destructor */
static void 
Day_tp_dealloc(DayObject *self)
{
  /* 
   * Delete stuff.
   */
  long node_count = 0;
  Node **nodes = NULL;
  get_preorder_nodes(self->root, &nodes, &node_count);
  int i;
  for (i=0; i<node_count; i++)
  {
    free(nodes[i]);
  }
  free(nodes);
  PyObject_Del(self);
}

/* REMOVE */
/* custom method: a silly hello world function */
static PyObject *
Day_myinc(DayObject *self, PyObject *unused)
{
  return PyInt_FromLong(self->myvalue++);
}

/* REMOVE */
/* custom method: get the x value of the current node */
static PyObject *
Day_get_x(DayObject *self, PyObject *unused)
{
  if (!self->cursor)
  {
    PyErr_SetString(PyExc_RuntimeError, "no node is selected");
    return NULL;
  }
  return PyFloat_FromDouble(self->cursor->x);
}

/* REMOVE */
/* custom method: set the x value of the current node */
static PyObject *
Day_set_x(DayObject *self, PyObject *args)
{
  double x = 0;
  int ok = PyArg_ParseTuple(args, "d", &x);
  if (!ok)
  {
    return NULL;
  }
  if (!self->cursor)
  {
    PyErr_SetString(PyExc_RuntimeError, "no node is selected");
    return NULL;
  }
  double old_x = self->cursor->x;
  self->cursor->x = x;
  return PyFloat_FromDouble(old_x);
}

/* REMOVE */
/* custom method: get the y value of the current node */
static PyObject *
Day_get_y(DayObject *self, PyObject *unused)
{
  if (!self->cursor)
  {
    PyErr_SetString(PyExc_RuntimeError, "no node is selected");
    return NULL;
  }
  return PyFloat_FromDouble(self->cursor->y);
}

/* REMOVE */
/* custom method: set the y value of the current node */
static PyObject *
Day_set_y(DayObject *self, PyObject *args)
{
  double y = 0;
  int ok = PyArg_ParseTuple(args, "d", &y);
  if (!ok)
  {
    return NULL;
  }
  if (!self->cursor)
  {
    PyErr_SetString(PyExc_RuntimeError, "no node is selected");
    return NULL;
  }
  double old_y = self->cursor->y;
  self->cursor->y = y;
  return PyFloat_FromDouble(old_y);
}

/* REMOVE */
/* custom method: set the branch length of the current node */
static PyObject *
Day_set_branch_length(DayObject *self, PyObject *args)
{
  double blen = 0;
  int ok = PyArg_ParseTuple(args, "d", &blen);
  if (!ok)
  {
    return NULL;
  }
  if (!self->cursor)
  {
    PyErr_SetString(PyExc_RuntimeError, "no node is selected");
    return NULL;
  }
  double old_blen = self->cursor->blen;
  self->cursor->blen = blen;
  return PyFloat_FromDouble(old_blen);
}

/* REMOVE */
/* custom method: move the cursor to the node with the given id */
static PyObject *
Day_select_node(DayObject *self, PyObject *args)
{
  /* get the target id and verify that the tree exists */
  long id = 0;
  int ok = PyArg_ParseTuple(args, "l", &id);
  if (!ok)
  {
    return NULL;
  }
  if (!self->root)
  {
    PyErr_SetString(PyExc_RuntimeError, "the tree is empty");
    return NULL;
  }
  /* get the list of all nodes in the tree */
  long count = 0;
  Node **nodes = NULL;
  get_preorder_nodes(self->root, &nodes, &count);
  /* find the target node by its id */
  Node *node = NULL;
  int i;
  for (i=0; i<count; i++)
  {
    if (nodes[i]->id == id)
    {
      node = nodes[i];
      break;
    }
  }
  /* free the list of nodes but not the nodes themselves */
  free(nodes);
  /* if the id was not found then fail */
  if (!node)
  {
    PyErr_SetString(PyExc_ValueError, "no node with the given id was found");
    return NULL;
  }
  /* set the cursor to the target node and return None */
  self->cursor = node;
  PyObject *result = Py_None;
  Py_INCREF(result);
  return result;
}

/* REMOVE */
/* custom method: add a node to the tree */
static PyObject *
Day_begin_node(DayObject *self, PyObject *args)
{
  /* create the new child node with the given id */
  Node *neighbor = (Node *) malloc(sizeof(Node));
  int ok = PyArg_ParseTuple(args, "l", &neighbor->id);
  if (!ok)
  {
    return NULL;
  }
  neighbor->x = 0.0;
  neighbor->y = 0.0;
  neighbor->blen = 0.0;
  neighbor->nneighbors = 0;
  neighbor->neighbors = NULL;
  neighbor->parent = NULL;
  if (self->cursor)
  {
    /* create the link from the child to the parent */
    neighbor->parent = self->cursor;
    neighbor->nneighbors = 1;
    neighbor->neighbors = (Node **) malloc(sizeof(Node *));
    neighbor->neighbors[0] = self->cursor;
    /* create the link from the parent to the child */
    long nneighbors = self->cursor->nneighbors + 1;
    Node **neighbors = (Node **) malloc(nneighbors * sizeof(Node *));
    int i;
    for (i=0; i<nneighbors-1; i++)
    {
      neighbors[i] = self->cursor->neighbors[i];
    }
    neighbors[i] = neighbor;
    free(self->cursor->neighbors);
    self->cursor->nneighbors = nneighbors;
    self->cursor->neighbors = neighbors;
  } else {
    self->root = neighbor;
  }
  self->cursor = neighbor;
  return PyInt_FromLong(self->cursor->id);
}

/* REMOVE */
/* custom method: move the node cursor to the parent node if possible */
static PyObject *
Day_end_node(DayObject *self, PyObject *unused)
{
  if (!self->root)
  {
    PyErr_SetString(PyExc_RuntimeError, "no root node was created");
    return NULL;
  }
  if (!self->cursor)
  {
    PyErr_SetString(PyExc_RuntimeError, "all nodes have already been ended");
    return NULL;
  }
  self->cursor = self->cursor->parent;
  PyObject *result = Py_None;
  Py_INCREF(result);
  return result;
}

/* REMOVE */
/* custom method: reroot the tree at the cursor */
static PyObject *
Day_reroot(DayObject *self, PyObject *unused)
{
  /* if there is no cursor or no tree then fail */
  if (!self->cursor)
  {
    PyErr_SetString(PyExc_RuntimeError, "no node is selected");
    return NULL;
  }
  if (!self->root)
  {
    PyErr_SetString(PyExc_RuntimeError, "no root node was found");
    return NULL;
  }
  /* reroot at the cursor */
  reroot(self->cursor);
  self->root = self->cursor;
  /* return None */
  PyObject *result = Py_None;
  Py_INCREF(result);
  return result;
}

/* REMOVE */
/* custom method: equalize daylight at the root */
static PyObject *
Day_equalize(DayObject *self, PyObject *unused)
{
  /* do some basic error checking */
  if (!self->root)
  {
    PyErr_SetString(PyExc_RuntimeError, "no root node was found");
    return NULL;
  }
  if (self->root->nneighbors < 2)
  {
    PyErr_SetString(PyExc_RuntimeError, "equalization requires at least two neighbors");
    return NULL;
  }
  /* utility variables */
  int i;
  int j;
  /* get the list of nodes in each subtree */
  Node ***node_lists = (Node ***) malloc(self->root->nneighbors * sizeof(Node **));
  long *node_list_lengths = malloc(self->root->nneighbors * sizeof(int));
  for (i=0; i<self->root->nneighbors; i++)
  {
    Node *neighbor = self->root->neighbors[i];
    get_preorder_nodes(neighbor, &node_lists[i], &node_list_lengths[i]);
  }
  /* get the center x and y values */
  double cx = self->root->x;
  double cy = self->root->y;
  /* get the intervals in each subtree */
  Interval *intervals = (Interval *) malloc(self->root->nneighbors * sizeof(Interval));
  for (i=0; i<self->root->nneighbors; i++)
  {
    /* get the angle from the tree root to the subtree root */
    Node *subtree_root = node_lists[i][0];
    double theta = norm(atan2(subtree_root->y - cy, subtree_root->x - cx));
    /* initialize the degenerate interval to just the angle */
    Interval interval = {theta, theta};
    /* expand the interval using the preorder interval list */
    Interval addend;
    for (j=1; j<node_list_lengths[i]; j++)
    {
      Node *node = node_lists[i][j];
      double a1 = norm(atan2(node->y - cy, node->x - cx));
      double a2 = norm(atan2(node->parent->y - cy, node->parent->x - cx));
      if (norm(a2 - a1) < norm(a1 - a2))
      {
        addend.low = a1;
        addend.high = a2;
      } else {
        addend.low = a2;
        addend.high = a1;
      }
      update_interval(&interval, &addend);
    }
    intervals[i] = interval;
  }
  /* get the amount of total daylight */
  double total_occlusion = 0;
  for (i=0; i<self->root->nneighbors; i++)
  {
    total_occlusion += norm(intervals[i].high - intervals[i].low);
  }
  /* rotate the subtrees if they can be spread out so they do not overlap */
  if (total_occlusion < 2*M_PI)
  {
    double daylight_per_subtree = (2*M_PI - total_occlusion) / self->root->nneighbors;
    double observed_cumulative_angle = 0;
    double expected_cumulative_angle = 0;
    for (i=0; i<self->root->nneighbors; i++)
    {
      double theta = expected_cumulative_angle - observed_cumulative_angle;
      if (theta)
      {
        double ct = cos(theta);
        double st = sin(theta);
        for (j=0; j<node_list_lengths[i]; j++)
        {
          Node *node = node_lists[i][j];
          double nx = cx + (node->x - cx) * ct - (node->y - cy) * st;
          double ny = cy + (node->x - cx) * st + (node->y - cy) * ct;
          node->x = nx;
          node->y = ny;
        }
      }
      double current_high = intervals[i].high;
      double next_low = intervals[(i+1) % self->root->nneighbors].low;
      observed_cumulative_angle += norm(next_low - current_high);
      expected_cumulative_angle += daylight_per_subtree;
    }
  }
  /* free the interval list */
  free(intervals);
  /* free the node lists and their lengths */
  free(node_list_lengths);
  for (i=0; i<self->root->nneighbors; i++)
  {
    free(node_lists[i]);
  }
  free(node_lists);
  /* if the total occlusion is too much then report it, otherwise return None */
  if (total_occlusion < 2*M_PI)
  {
    PyObject *result = Py_None;
    Py_INCREF(result);
    return result;
  } else {
    PyErr_SetString(PyExc_RuntimeError, "subtrees span at least 360 degrees");
    return NULL;
  }
}

/* REMOVE */
/* custom method: get the number of nodes in the tree */
static PyObject *
Day_get_node_count(DayObject *self, PyObject *unused)
{
  long count = 0;
  if (self->root)
  {
    count = _fill_preorder_nodes(self->root, NULL, 0);
  }
  return PyInt_FromLong(count);
}

/* REMOVE */
/* custom method: get the id of the root node */
static PyObject *
Day_get_root_id(DayObject *self, PyObject *unused)
{
  if (!self->root)
  {
    PyErr_SetString(PyExc_RuntimeError, "no root node was found");
    return NULL;
  }
  return PyInt_FromLong(self->root->id);
}

/* REMOVE */
/* custom method: get the number of subtrees of the current node */
static PyObject *
Day_get_subtree_count(DayObject *self, PyObject *unused)
{
  long count = 0;
  if (self->cursor)
  {
    int i;
    for (i=0; i<self->cursor->nneighbors; i++)
    {
      Node *neighbor = self->cursor->neighbors[i];
      if (neighbor != self->cursor->parent)
      {
        count++;
      }
    }
  }
  return PyInt_FromLong(count);
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

static PyMethodDef HmmuscMethods[] = {
  {"hello", hello_world, METH_VARARGS, "Say hi."},
  {"posterior", posterior_python, METH_VARARGS, "Posterior decoding."},
  {NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC
inithmmusc(void) 
{
  (void) Py_InitModule("hmmusc", HmmuscMethods);
  /*
  PyObject* m;
  m = Py_InitModule3("hmmusc", NULL, "Extend python with HMM functions.");
  if (m == NULL)
    return;
  */
}

