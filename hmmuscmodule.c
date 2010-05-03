#include <Python.h>

#include "hmmguts/hmmguts.h"

static PyObject *HmmuscError;

/* On error, set an appropriate exception and return -1.
 * Function arguments are pointers to new-style buffers.
 */
int check_buffers(Py_buffer *distn, Py_buffer *trans,
    Py_buffer *like, Py_buffer *post) {
  /* check data format for each buffer */
  if (distn->format[0] != 'd') {
    PyErr_SetString(HmmuscError,
        "the distribution vector has the wrong data type");
    return -1;
  }
  if (trans->format[0] != 'd') {
    PyErr_SetString(HmmuscError,
        "the transition matrix has the wrong data type");
    return -1;
  }
  if (like->format[0] != 'd') {
    PyErr_SetString(HmmuscError,
        "the likelihoods matrix has the wrong data type");
    return -1;
  }
  if (post->format[0] != 'd') {
    PyErr_SetString(HmmuscError,
        "the posterior matrix has the wrong data type");
    return -1;
  }
  /* buffers should be non-null */
  if (distn->buf == NULL) {
    PyErr_SetString(HmmuscError,
        "the distribution vector data buffer is NULL");
    return -1;
  }
  if (trans->buf == NULL) {
    PyErr_SetString(HmmuscError,
        "the transition matrix data buffer is NULL");
    return -1;
  }
  if (like->buf == NULL) {
    PyErr_SetString(HmmuscError,
        "the likelihoods vector data buffer is NULL");
    return -1;
  }
  if (post->buf == NULL) {
    PyErr_SetString(HmmuscError,
        "the posterior matrix data buffer is NULL");
    return -1;
  }
  return 0;
}

/* On error, set an appropriate exception and return -1.
 * Array dimensions should already have been checked.
 * Function arguments are new-style buffer shapes.
 */
int check_shape_compatibility(Py_ssize_t *distn, Py_ssize_t *trans,
    Py_ssize_t *like, Py_ssize_t *post)
{
  if (trans[0] != trans[1]) {
    PyErr_SetString(HmmuscError,
        "the transition matrix should be square");
    return -1;
  }
  int nstates = distn[0];
  if (trans[0] != nstates) {
    PyErr_SetString(HmmuscError,
        "the transition matrix should have the same number of states "
        "as the distribution vector");
    return -1;
  }
  if (like[1] != nstates) {
    PyErr_SetString(HmmuscError,
        "the likelihoods matrix should have the same number of states "
        "as the distribution vector");
    return -1;
  }
  int npositions = like[0];
  if (post[1] != nstates) {
    PyErr_SetString(HmmuscError,
        "the posterior matrix should have the same number of states "
        "as the distribution vector");
    return -1;
  }
  if (post[0] != npositions) {
    PyErr_SetString(HmmuscError,
        "the posterior matrix should have the same number of positions "
        "as the likelihoods matrix");
    return -1;
  }
  return 0;
}

/* On error, set an appropriate exception and return -1.
 */
int check_interface(PyObject *obj_distn, PyObject *obj_trans,
    PyObject *obj_like, PyObject *obj_post)
{
  int distn_fail = !PyObject_CheckBuffer(obj_distn);
  int trans_fail = !PyObject_CheckBuffer(obj_distn);
  int like_fail = !PyObject_CheckBuffer(obj_distn);
  int post_fail = !PyObject_CheckBuffer(obj_distn);
  int all_fail = distn_fail && trans_fail && like_fail && post_fail;
  if (all_fail) {
    PyErr_SetString(HmmuscError,
        "these objects should support the buffer interface");
    return -1;
  }
  if (distn_fail) {
    PyErr_SetString(HmmuscError,
        "the distribution object should support the buffer interface");
    return -1;
  }
  if (trans_fail) {
    PyErr_SetString(HmmuscError,
        "the transition matrix object should support the buffer interface");
    return -1;
  }
  if (like_fail) {
    PyErr_SetString(HmmuscError,
        "the likelihoods object should support the buffer interface");
    return -1;
  }
  if (post_fail) {
    PyErr_SetString(HmmuscError,
        "the posterior object should support the buffer interface");
    return -1;
  }
  return 0;
}

/* Try to use the buffer interface.
 * Python args are:
 * distribution_in
 * transitions_in
 * likelihoods_in
 * posterior_out
 * TODO: allow the arrays to be non-contiguous
 */
static PyObject *
newbuf_fwdbwd_nodisk_python(PyObject *self, PyObject *args)
{
  int except = 0;
  /* buffer views */
  Py_buffer distn_view;
  Py_buffer trans_view;
  Py_buffer like_view;
  Py_buffer post_view;
  int got_distn_view = 0;
  int got_trans_view = 0;
  int got_like_view = 0;
  int got_post_view = 0;
  /* read the args */
  PyObject *obj_distn;
  PyObject *obj_trans;
  PyObject *obj_like;
  PyObject *obj_post;
  if (!PyArg_ParseTuple(args, "OOOO",
        &obj_distn, &obj_trans, &obj_like, &obj_post)) {
    except = 1; goto end;
  }
  /* all objects should support the buffer interface */
  if (check_interface(obj_distn, obj_trans, obj_like, obj_post) < 0) {
    except = 1; goto end;
  }
  /* get the buffer view for each object */
  int flags = PyBUF_ND | PyBUF_FORMAT | PyBUF_C_CONTIGUOUS | PyBUF_WRITABLE;
  if (PyObject_GetBuffer(obj_distn, &distn_view, flags) < 0) {
    except = 1; goto end;
  } else {
    got_distn_view = 1;
  }
  if (PyObject_GetBuffer(obj_trans, &trans_view, flags) < 0) {
    except = 1; goto end;
  } else {
    got_trans_view = 1;
  }
  if (PyObject_GetBuffer(obj_like, &like_view, flags) < 0) {
    except = 1; goto end;
  } else {
    got_like_view = 1;
  }
  if (PyObject_GetBuffer(obj_post, &post_view, flags) < 0) {
    except = 1; goto end;
  } else {
    got_post_view = 1;
  }
  /* the distribution buffer should be one dimensional */
  if (distn_view.ndim != 1) {
    PyErr_SetString(HmmuscError, "the distribution buffer should be 1D");
    except = 1; goto end;
  }
  /* the other three buffers should be two dimensional */
  if (trans_view.ndim != 2) {
    PyErr_SetString(HmmuscError, "the transition matrix buffer should be 2D");
    except = 1; goto end;
  }
  if (like_view.ndim != 2) {
    PyErr_SetString(HmmuscError, "the likelihoods buffer should be 2D");
    except = 1; goto end;
  }
  if (post_view.ndim != 2) {
    PyErr_SetString(HmmuscError, "the posterior buffer should be 2D");
    except = 1; goto end;
  }
  /* the buffers should have compatible shapes */
  if (check_shape_compatibility(distn_view.shape, trans_view.shape,
      like_view.shape, post_view.shape) < 0) {
    except = 1; goto end;
  }
  /* the buffers should have the right format */
  if (check_buffers(&distn_view, &trans_view,
        &like_view, &post_view) < 0) {
    except = 1; goto end;
  }
  /* get some information */
  int nstates = distn_view.shape[0];
  int npositions = like_view.shape[0];
  /* fill the posterior matrix */
  struct TM tm;
  tm.nstates = nstates;
  tm.distn = (double *) distn_view.buf;
  tm.trans = (double *) trans_view.buf;
  if (fwdbwd_nodisk(&tm, npositions,
      (double *) like_view.buf, (double *) post_view.buf) < 0) {
    PyErr_SetString(HmmuscError,
        "failed to run the forward-backward algorithm");
    except = 1; goto end;
  }
end:
  /* clean up the buffer views */
  if (got_distn_view) {
    PyBuffer_Release(&distn_view);
  }
  if (got_trans_view) {
    PyBuffer_Release(&trans_view);
  }
  if (got_like_view) {
    PyBuffer_Release(&like_view);
  }
  if (got_post_view) {
    PyBuffer_Release(&post_view);
  }
  /* return an appropriate value */
  if (except) {
    return NULL;
  } else {
    return Py_BuildValue("i", 42);
  }
}

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
  {"newbuf_fwdbwd_nodisk", newbuf_fwdbwd_nodisk_python, METH_VARARGS,
    "Forward-backward algorithm with all arrays in RAM, "
    "using the new-style buffer interface."},
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

