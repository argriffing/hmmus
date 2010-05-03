#include <Python.h>

#include "hmmguts/hmmguts.h"

static PyObject *HmmusbufError;

/* 
 * Initialize the transition matrix object
 * given a numpy array of initial distribution floats
 * and a numpy array of transition matrix floats.
 * The members of the TM struct will be assigned
 * to point to the data in the input buffers,
 * so these members should not be freed.
 * A return value of -1 means there was an error.
 */
int TM_init_from_buffers(struct TM *ptm, Py_buffer *distn, Py_buffer *trans)
{
  /* initialize the TM struct */
  ptm->nstates = 0;
  ptm->distn = NULL;
  ptm->trans = NULL;
  /* check for the existence of data in each buffer */
  if (distn->buf == NULL) {
    PyErr_SetString(HmmusbufError,
        "the distribution vector data buffer is NULL");
    return -1;
  }
  if (trans->buf == NULL) {
    PyErr_SetString(HmmusbufError,
        "the transition matrix data buffer is NULL");
    return -1;
  }
  /* check the data format of each buffer */
  if (distn->format[0] != 'd') {
    PyErr_SetString(HmmusbufError,
        "the distribution vector has the wrong data type");
    return -1;
  }
  if (trans->format[0] != 'd') {
    PyErr_SetString(HmmusbufError,
        "the transition matrix has the wrong data type");
    return -1;
  }
  /* the distribution vector should be one dimensional */
  if (distn->ndim != 1) {
    PyErr_SetString(HmmusbufError, "the distribution vector should be 1D");
    except = 1; goto end;
  }
  /* the transition matrix should be two dimensional */
  if (trans->ndim != 2) {
    PyErr_SetString(HmmusbufError, "the transition matrix should be 2D");
    except = 1; goto end;
  }
  /* the transition matrix should be square */
  if (trans->shape[0] != trans->shape[1]) {
    PyErr_SetString(HmmusbufError,
        "the transition matrix should be square");
    return -1;
  }
  /* the distribution vector and transition matrix should be compatible */
  if (trans->shape[0] != distn->shape[0]) {
    PyErr_SetString(HmmusbufError,
        "the transition matrix should have the same number of states "
        "as the distribution vector");
    return -1;
  }
  /* fill the TM struct */
  ptm->nstates = distn->shape[0];
  ptm->distn = (double *) distn->buf;
  ptm->trans = (double *) trans->buf;
  return 0;
}

/* Collect new-style buffer stuff here.
 */
struct HMBUFA {
  Py_buffer distn;
  Py_buffer trans;
  int has_distn;
  int has_trans;
  TM tm;
};

/* Collect new-style buffer stuff here.
 */
struct HMBUFB {
  Py_buffer like;
  Py_buffer post;
  int has_like;
  int has_post;
  struct HMBUFA hmbufa;
};

int HMBUFA_init(struct HMBUFA *p, PyObject *distn, PyObject *trans)
{
}

int HMBUFA_destroy(struct HMBUFA *p)
{
  if (p->has_distn) {
    PyBuffer_Release(&p->distn);
    p->has_distn = 0;
  }
  if (p->has_trans) {
    PyBuffer_Release(&p->trans);
    p->has_trans = 0;
  }
  p->tm.nstates = 0;
  p->tm.distn = 0;
  p->tm.trans = 0;
  return 0;
}

int HMBUFB_init(struct HMBUFB *p, PyObject *distn, PyObject *trans,
    PyObject *like, PyObject *post)
{
  if (HMBUFA_init(&p->hmbufa) < 0) {
    return -1;
  }
}

int HMBUFB_destroy(struct HMBUFB *p)
{
  if (p->has_like) {
    PyBuffer_Release(&p->like);
    p->has_like = 0;
  }
  if (p->has_post) {
    PyBuffer_Release(&p->post);
    p->has_post = 0;
  }
  return HMBUFA_destroy(&p->hmbufa);
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
    PyErr_SetString(HmmusbufError,
        "these objects should support the buffer interface");
    return -1;
  }
  if (distn_fail) {
    PyErr_SetString(HmmusbufError,
        "the distribution object should support the buffer interface");
    return -1;
  }
  if (trans_fail) {
    PyErr_SetString(HmmusbufError,
        "the transition matrix object should support the buffer interface");
    return -1;
  }
  if (like_fail) {
    PyErr_SetString(HmmusbufError,
        "the likelihoods object should support the buffer interface");
    return -1;
  }
  if (post_fail) {
    PyErr_SetString(HmmusbufError,
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
  /* initialize a TM struct */
  struct TM tm;
  if (TM_init_from_buffers(&tm, &distn_view, &trans_view) < 0) {
    except = 1; goto end;
  }
  /* check dimensionality */
  if (like_view.ndim != 2) {
    PyErr_SetString(HmmusbufError, "the likelihoods buffer should be 2D");
    except = 1; goto end;
  }
  if (post_view.ndim != 2) {
    PyErr_SetString(HmmusbufError, "the posterior buffer should be 2D");
    except = 1; goto end;
  }
  /* the buffers should have compatible shapes */
  if (like_view.shape[1] != tm.nstates) {
    PyErr_SetString(HmmusbufError,
        "the likelihoods matrix should have the same number of states "
        "as the distribution vector");
    except = 1; goto end;
  }
  int npositions = like_view.shape[0];
  if (post_view.shape[1] != nstates) {
    PyErr_SetString(HmmusbufError,
        "the posterior matrix should have the same number of states "
        "as the distribution vector");
    except = 1; goto end;
  }
  if (post_view.shape[0] != npositions) {
    PyErr_SetString(HmmusbufError,
        "the posterior matrix should have the same number of positions "
        "as the likelihoods matrix");
    except = 1; goto end;
  }
  /* check data format for each buffer */
  if (like_view.format[0] != 'd') {
    PyErr_SetString(HmmusbufError,
        "the likelihoods matrix has the wrong data type");
    return -1;
  }
  if (post_view.format[0] != 'd') {
    PyErr_SetString(HmmusbufError,
        "the posterior matrix has the wrong data type");
    return -1;
  }
  /* buffers should be non-null */
  if (like_view.buf == NULL) {
    PyErr_SetString(HmmusbufError,
        "the likelihoods vector data buffer is NULL");
    return -1;
  }
  if (post_view.buf == NULL) {
    PyErr_SetString(HmmusbufError,
        "the posterior matrix data buffer is NULL");
    return -1;
  }
  /* fill the posterior matrix */
  if (fwdbwd_nodisk(&tm, like_view.shape[0],
      (double *) like_view.buf, (double *) post_view.buf) < 0) {
    PyErr_SetString(HmmusbufError,
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
    PyErr_SetString(HmmusbufError, "forward algorithm error");
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
    PyErr_SetString(HmmusbufError, "backward algorithm error");
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
    PyErr_SetString(HmmusbufError, "posterior algorithm error");
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
    PyErr_SetString(HmmusbufError, "fwdbwd_somedisk algorithm error");
    TM_del(&tm);
    return NULL;
  }
  TM_del(&tm);
  return Py_BuildValue("i", 0);
}

static PyMethodDef HmmuscMethods[] = {
  {"forward", forward_python, METH_VARARGS,
    "Forward algorithm."},
  {"backward", backward_python, METH_VARARGS,
    "Backward algorithm."},
  {"posterior", posterior_python, METH_VARARGS,
    "Posterior decoding."},
  {"fwdbwd_somedisk", fwdbwd_somedisk_python, METH_VARARGS,
    "Forward-backward algorithm with intermediate arrays in RAM."},
  {"newbuf_fwdbwd_nodisk", newbuf_fwdbwd_nodisk_python, METH_VARARGS,
    "Forward-backward algorithm with all arrays in RAM, "
    "using the new-style buffer interface."},
  {NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC
inithmmusbuf(void) 
{
  PyObject *m = Py_InitModule("hmmusbuf", HmmusbufMethods);
  if (!m) return;

  HmmusbufError = PyErr_NewException("hmmusbuf.error", NULL, NULL);
  Py_INCREF(HmmusbufError);
  PyModule_AddObject(m, "error", HmmusbufError);
}

