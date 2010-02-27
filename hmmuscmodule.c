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
  Py_ssize_t ndistribution;
  Py_ssize_t ntransitions;
  ptm->initial_distn = double_vector_helper(vtuple, &ndistribution);
  ptm->value = double_vector_helper(mtuple, &ntransitions);
  ptm->order = (int) ndistribution;
  if (!ptm->value || !ptm->initial_distn)
  {
    PyErr_SetString(HmmuscError, "transition matrix init error (NULL)");
    TM_del(ptm);
    return -1;
  }
  if (ndistribution*ndistribution != ntransitions)
  {
    PyErr_SetString(HmmuscError, "transition matrix init error (k*k != n)");
    TM_del(ptm);
    return -1;
  }
  return 0;
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
  if (TM_init_from_pytuples(&tm, vtuple, mtuple)) return NULL;
  if (do_posterior(&tm, f_name, s_name, b_name, p_name))
  {
    PyErr_SetString(HmmuscError, "posterior algorithm error");
    TM_del(&tm);
    return NULL;
  }
  TM_del(&tm);
  return Py_BuildValue("i", 0);
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
  if (TM_init_from_pytuples(&tm, vtuple, mtuple)) return NULL;
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
  if (TM_init_from_pytuples(&tm, vtuple, mtuple)) return NULL;
  if (do_backward(&tm, likelihoods_name, scaling_name, backward_name))
  {
    PyErr_SetString(HmmuscError, "backward algorithm error");
    TM_del(&tm);
    return NULL;
  }
  TM_del(&tm);
  return Py_BuildValue("i", 0);
}

static PyMethodDef HmmuscMethods[] = {
  {"posterior", posterior_python, METH_VARARGS, "Posterior decoding."},
  {"backward", backward_python, METH_VARARGS, "Backward algorithm."},
  {"forward", forward_python, METH_VARARGS, "Forward algorithm."},
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

