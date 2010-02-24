#include <Python.h>

#include "hmmguts/hmmguts.h"

static PyObject *HmmuscError;

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
  int result;
  int ok = PyArg_ParseTuple(args, "issssss",
      &nstates, &distn_name, &transitions_name,
      &forward_name, &scaling_name, &backward_name, &posterior_name);
  if (!ok) return NULL;
  struct TM tm;
  result = TM_init_from_names(&tm, nstates, distn_name, transitions_name);
  if (result)
  {
    PyErr_SetString(HmmuscError, "transition matrix init error");
    return NULL;
  }
  result = do_posterior(&tm, forward_name, scaling_name, backward_name,
      posterior_name);
  if (result)
  {
    PyErr_SetString(HmmuscError, "posterior algorithm error");
    return NULL;
  }
  TM_del(&tm);
  return Py_BuildValue("i", 0);
}

static PyObject *
forward_python(PyObject *self, PyObject *args)
{
  int nstates;
  const char *distn_name;
  const char *transitions_name;
  const char *likelihoods_name;
  const char *forward_name;
  const char *scaling_name;
  int result;
  int ok = PyArg_ParseTuple(args, "isssss",
      &nstates, &distn_name, &transitions_name,
      &likelihoods_name, &forward_name, &scaling_name);
  if (!ok) return NULL;
  struct TM tm;
  result = TM_init_from_names(&tm, nstates, distn_name, transitions_name);
  if (result)
  {
    PyErr_SetString(HmmuscError, "transition matrix init error");
    return NULL;
  }
  result = do_forward(&tm, likelihoods_name, forward_name, scaling_name);
  if (result)
  {
    PyErr_SetString(HmmuscError, "forward algorithm error");
    return NULL;
  }
  TM_del(&tm);
  return Py_BuildValue("i", 0);
}

static PyObject *
backward_python(PyObject *self, PyObject *args)
{
  int nstates;
  const char *distn_name;
  const char *transitions_name;
  const char *likelihoods_name;
  const char *scaling_name;
  const char *backward_name;
  int result;
  int ok = PyArg_ParseTuple(args, "isssss",
      &nstates, &distn_name, &transitions_name,
      &likelihoods_name, &scaling_name, &backward_name);
  if (!ok) return NULL;
  struct TM tm;
  result = TM_init_from_names(&tm, nstates, distn_name, transitions_name);
  if (result)
  {
    PyErr_SetString(HmmuscError, "transition matrix init error");
    return NULL;
  }
  result = do_backward(&tm, likelihoods_name, scaling_name, backward_name);
  if (result)
  {
    PyErr_SetString(HmmuscError, "backward algorithm error");
    return NULL;
  }
  TM_del(&tm);
  return Py_BuildValue("i", 0);
}

static PyMethodDef HmmuscMethods[] = {
  {"hello", hello_world, METH_VARARGS, "Say hi."},
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

