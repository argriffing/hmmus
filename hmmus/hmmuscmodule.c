#include <Python.h>

#include "hmmguts/hmmguts.h"
#include "notpythonc.h"

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

