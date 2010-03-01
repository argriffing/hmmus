#include <Python.h>

#include "hmmguts/hmmguts.h"

static PyObject *HmmuscError;

/*
 * Input is a tuple of python floats.
 * The length of the array is filled.
 * Returned is the pointer to the new array of doubles.
 */
double *
double_vector_helper(PyObject *myfloattuple, Py_ssize_t *pn)
{
  PyObject *myfloat;
  Py_ssize_t i;
  *pn = PyTuple_Size(myfloattuple);
  if (!(*pn)) return NULL;
  double *doubles = malloc((*pn)*sizeof(double));
  for (i=0; i<(*pn); i++)
  {
    myfloat = PyTuple_GetItem(myfloattuple, i);
    doubles[i] = PyFloat_AsDouble(myfloat);
  }
  return doubles;
}

/* 
 * Initialize the transition matrix object
 * given a pytuple of initial distribution floats
 * and a pytuple of transition matrix floats.
 * A return value of -1 means there was an error.
 */
int TM_init_from_pytuples(struct TM *ptm, PyObject *vtuple, PyObject *mtuple)
{
  ptm->nstates = 0;
  ptm->distn = NULL;
  ptm->trans = NULL;
  Py_ssize_t ndistribution;
  Py_ssize_t ntransitions;
  ptm->distn = double_vector_helper(vtuple, &ndistribution);
  if (!ptm->distn)
  {
    PyErr_SetString(HmmuscError, "initial distribution init error");
    goto fail;
  }
  ptm->trans = double_vector_helper(mtuple, &ntransitions);
  if (!ptm->trans)
  {
    PyErr_SetString(HmmuscError, "transition matrix init error");
    goto fail;
  }
  ptm->nstates = (int) ndistribution;
  if (ndistribution*ndistribution != ntransitions)
  {
    PyErr_SetString(HmmuscError, "transition matrix compatibility error");
    goto fail;
  }
  return 0;
fail:
  TM_del(ptm);
  return -1;
}

static PyObject *
forward_python(PyObject *self, PyObject *args)
{
  PyObject *vtuple;
  PyObject *mtuple;
  const char *likelihoods_name;
  const char *forward_name;
  const char *scaling_name;
  /* read the args */
  int ok = PyArg_ParseTuple(args, "OOsss",
      &vtuple, &mtuple,
      &likelihoods_name, &forward_name, &scaling_name);
  if (!ok) return NULL;
  /* run the hmm algorithm */
  struct TM tm;
  if (TM_init_from_pytuples(&tm, vtuple, mtuple) < 0) return NULL;
  if (do_forward(&tm, likelihoods_name, forward_name, scaling_name))
  {
    PyErr_SetString(HmmuscError, "forward algorithm error");
    TM_del(&tm);
    return NULL;
  }
  TM_del(&tm);
  return Py_BuildValue("i", 0);
}

static PyObject *
backward_python(PyObject *self, PyObject *args)
{
  PyObject *vtuple;
  PyObject *mtuple;
  const char *likelihoods_name;
  const char *scaling_name;
  const char *backward_name;
  /* read the args */
  int ok = PyArg_ParseTuple(args, "OOsss",
      &vtuple, &mtuple,
      &likelihoods_name, &scaling_name, &backward_name);
  if (!ok) return NULL;
  /* run the hmm algorithm */
  struct TM tm;
  if (TM_init_from_pytuples(&tm, vtuple, mtuple) < 0) return NULL;
  if (do_backward(&tm, likelihoods_name, scaling_name, backward_name))
  {
    PyErr_SetString(HmmuscError, "backward algorithm error");
    TM_del(&tm);
    return NULL;
  }
  TM_del(&tm);
  return Py_BuildValue("i", 0);
}

static PyObject *
posterior_python(PyObject *self, PyObject *args)
{
  PyObject *vtuple;
  PyObject *mtuple;
  const char *f_name;
  const char *s_name;
  const char *b_name;
  const char *p_name;
  /* read the args */
  int ok = PyArg_ParseTuple(args, "OOssss",
      &vtuple, &mtuple,
      &f_name, &s_name, &b_name, &p_name);
  if (!ok) return NULL;
  /* run the hmm algorithm */
  struct TM tm;
  if (TM_init_from_pytuples(&tm, vtuple, mtuple) < 0) return NULL;
  if (do_posterior(tm.nstates, f_name, s_name, b_name, p_name))
  {
    PyErr_SetString(HmmuscError, "posterior algorithm error");
    TM_del(&tm);
    return NULL;
  }
  TM_del(&tm);
  return Py_BuildValue("i", 0);
}

static PyObject *
fwdbwd_somedisk_python(PyObject *self, PyObject *args)
{
  PyObject *vtuple;
  PyObject *mtuple;
  const char *l_name;
  const char *d_name;
  /* read the args */
  int ok = PyArg_ParseTuple(args, "OOss",
      &vtuple, &mtuple,
      &l_name, &d_name);
  if (!ok) return NULL;
  /* run the hmm algorithm */
  struct TM tm;
  if (TM_init_from_pytuples(&tm, vtuple, mtuple) < 0) return NULL;
  if (do_fwdbwd_somedisk(&tm, l_name, d_name))
  {
    PyErr_SetString(HmmuscError, "fwdbwd_somedisk algorithm error");
    TM_del(&tm);
    return NULL;
  }
  TM_del(&tm);
  return Py_BuildValue("i", 0);
}

static PyObject *
fwdbwd_nodisk_python(PyObject *self, PyObject *args)
{
  struct TM tm;
  tm.nstates = 0;
  tm.distn = NULL;
  tm.trans = NULL;
  PyObject *dtuple = NULL;
  double *d_big = NULL;
  /* read the args */
  PyObject *vtuple;
  PyObject *mtuple;
  PyObject *ltuple;
  int ok = PyArg_ParseTuple(args, "OOO",
      &vtuple, &mtuple, &ltuple);
  if (!ok) goto end;
  /* read the distribution vector and transition matrix */
  if (TM_init_from_pytuples(&tm, vtuple, mtuple) < 0) goto end;
  /* read the vector of likelihoods */
  Py_ssize_t nlikelihoods;
  double *l_big = double_vector_helper(ltuple, &nlikelihoods);
  if (!l_big)
  {
    PyErr_SetString(HmmuscError, "likelihood init error");
    goto end;
  }
  if (nlikelihoods % tm.nstates != 0)
  {
    PyErr_SetString(HmmuscError, "likelihood shape error");
    goto end;
  }
  size_t nobs = nlikelihoods / tm.nstates;
  d_big = malloc(nobs * tm.nstates * sizeof(double));
  if (fwdbwd_nodisk(&tm, nobs, l_big, d_big))
  {
    PyErr_SetString(HmmuscError, "fwdbwd_nodisk algorithm error");
    goto end;
  }
  dtuple = PyTuple_New(nlikelihoods);
  Py_ssize_t i;
  for (i=0; i<nlikelihoods; i++)
  {
    PyTuple_SetItem(dtuple, i, PyFloat_FromDouble(d_big[i]));
  }
end:
  free(d_big);
  TM_del(&tm);
  return dtuple;
}

/*
 * An experimental function that uses a buffer interface.
 */
static PyObject *
hello_buffer_python(PyObject *self, PyObject *args)
{
  /* read the args */
  int ok = PyArg_ParseTuple(args, "");
  if (!ok) return NULL;
  /* say hi */
  int i;
  double hi[2] = {2.71828, 3.14159};
  PyObject *mytuple = PyTuple_New(2);
  for (i=0; i<2; i++)
  {
    PyTuple_SetItem(mytuple, i, PyFloat_FromDouble(hi[i]));
  }
  return mytuple;
}

static PyMethodDef HmmuscMethods[] = {
  {"hello_buffer", hello_buffer_python, METH_VARARGS,
    "an experimental function that uses the buffer interface"},
  {"forward", forward_python, METH_VARARGS,
    "Forward algorithm."},
  {"backward", backward_python, METH_VARARGS,
    "Backward algorithm."},
  {"posterior", posterior_python, METH_VARARGS,
    "Posterior decoding."},
  {"fwdbwd_somedisk", fwdbwd_somedisk_python, METH_VARARGS,
    "Forward-backward algorithm with intermediate arrays in RAM."},
  {"fwdbwd_nodisk", fwdbwd_nodisk_python, METH_VARARGS,
    "Forward-backward algorithm with all arrays in RAM."},
  {NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC
inithmmusc(void) 
{
  PyObject *m = Py_InitModule("hmmusc", HmmuscMethods);
  if (!m) return;

  HmmuscError = PyErr_NewException("hmmusc.error", NULL, NULL);
  Py_INCREF(HmmuscError);
  PyModule_AddObject(m, "error", HmmuscError);
}

