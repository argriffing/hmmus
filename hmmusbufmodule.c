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
    return -1;
  }
  /* the transition matrix should be two dimensional */
  if (trans->ndim != 2) {
    PyErr_SetString(HmmusbufError, "the transition matrix should be 2D");
    return -1;
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

/* Set an error message and return -1 if an error is found.
 */
int check_interfaces(int n, PyObject **pobjects, char **names) {
  int i;
  /* get the number of objects with a buffer interface */
  int nvalid = 0;
  for (i=0; i<n; ++i) {
    if (PyObject_CheckBuffer(pobjects[i])) {
      ++nvalid;
    }
  }
  if (!nvalid) {
    PyErr_SetString(HmmusbufError,
        "these objects should support the buffer interface");
    return -1;
  }
  /* report the first nonconformant object if any */
  char msg[1000];
  char fmtstr[] = "%s should support the buffer interface";
  for (i=0; i<n; ++i) {
    if (!PyObject_CheckBuffer(pobjects[i])) {
      sprintf(msg, fmtstr, names[i]);
      PyErr_SetString(HmmusbufError, msg);
      return -1;
    }
  }
  return 0;
}

/* Collect new-style buffer stuff here.
 */
struct HMBUFA {
  struct TM tm;
  Py_buffer distn;
  Py_buffer trans;
  int has_distn;
  int has_trans;
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

int HMBUFA_clear(struct HMBUFA *p)
{
  p->tm.nstates = 0;
  p->tm.distn = 0;
  p->tm.trans = 0;
  p->has_distn = 0;
  p->has_trans = 0;
  return 0;
}

int HMBUFA_init(struct HMBUFA *p, PyObject *distn, PyObject *trans)
{
  HMBUFA_clear(p);
  /* check the interfaces */
  PyObject *pyobjects[] = {distn, trans};
  char *names[] = {"the distribution vector", "the transition matrix"};
  if (check_interfaces(2, pyobjects, names) < 0) {
    return -1;
  }
  /* get the buffer view for each object */
  int flags = PyBUF_ND | PyBUF_FORMAT | PyBUF_C_CONTIGUOUS | PyBUF_WRITABLE;
  if (PyObject_GetBuffer(distn, &p->distn, flags) < 0) {
    return -1;
  } else {
    p->has_distn = 1;
  }
  if (PyObject_GetBuffer(trans, &p->trans, flags) < 0) {
    return -1;
  } else {
    p->has_trans = 1;
  }
  /* init the TM using the buffer views */
  if (TM_init_from_buffers(&p->tm, &p->distn, &p->trans) < 0) {
    return -1;
  }
  return 0;
}

int HMBUFA_destroy(struct HMBUFA *p)
{
  if (p->has_distn) {
    PyBuffer_Release(&p->distn);
  }
  if (p->has_trans) {
    PyBuffer_Release(&p->trans);
  }
  return 0;
}

int HMBUFB_clear(struct HMBUFB *p)
{
  p->has_like = 0;
  p->has_post = 0;
  HMBUFA_clear(&p->hmbufa);
  return 0;
}

int HMBUFB_init(struct HMBUFB *p, PyObject *distn, PyObject *trans,
    PyObject *like, PyObject *post)
{
  HMBUFB_clear(p);
  /* check the interfaces */
  PyObject *pyobjects[] = {distn, trans, like, post};
  char *names[] = {
    "the distribution vector", "the transition matrix",
    "the likelihoods matrix", "the posterior matrix"};
  if (check_interfaces(4, pyobjects, names) < 0) {
    return -1;
  }
  /* initialize the distribution and transition data */
  if (HMBUFA_init(&p->hmbufa, distn, trans) < 0) {
    return -1;
  }
  int nstates = p->hmbufa.tm.nstates;
  /* get the buffer view for the likelihood and posterior objects */
  int flags = PyBUF_ND | PyBUF_FORMAT | PyBUF_C_CONTIGUOUS | PyBUF_WRITABLE;
  if (PyObject_GetBuffer(like, &p->like, flags) < 0) {
    return -1;
  } else {
    p->has_like = 1;
  }
  if (PyObject_GetBuffer(post, &p->post, flags) < 0) {
    return -1;
  } else {
    p->has_post = 1;
  }
  /* check dimensionality */
  if (p->like.ndim != 2) {
    PyErr_SetString(HmmusbufError, "the likelihoods buffer should be 2D");
    return -1;
  }
  if (p->post.ndim != 2) {
    PyErr_SetString(HmmusbufError, "the posterior buffer should be 2D");
    return -1;
  }
  /* the buffers should have compatible shapes */
  if (p->like.shape[1] != nstates) {
    PyErr_SetString(HmmusbufError,
        "the likelihoods matrix should have the same number of states "
        "as the distribution vector");
    return -1;
  }
  int npositions = p->like.shape[0];
  if (p->post.shape[1] != nstates) {
    PyErr_SetString(HmmusbufError,
        "the posterior matrix should have the same number of states "
        "as the distribution vector");
    return -1;
  }
  if (p->post.shape[0] != npositions) {
    PyErr_SetString(HmmusbufError,
        "the posterior matrix should have the same number of positions "
        "as the likelihoods matrix");
    return -1;
  }
  /* check data format for each buffer */
  if (p->like.format[0] != 'd') {
    PyErr_SetString(HmmusbufError,
        "the likelihoods matrix has the wrong data type");
    return -1;
  }
  if (p->post.format[0] != 'd') {
    PyErr_SetString(HmmusbufError,
        "the posterior matrix has the wrong data type");
    return -1;
  }
  /* buffers should be non-null */
  if (p->like.buf == NULL) {
    PyErr_SetString(HmmusbufError,
        "the likelihoods vector data buffer is NULL");
    return -1;
  }
  if (p->post.buf == NULL) {
    PyErr_SetString(HmmusbufError,
        "the posterior matrix data buffer is NULL");
    return -1;
  }
  return 0;
}

int HMBUFB_destroy(struct HMBUFB *p)
{
  if (p->has_like) {
    PyBuffer_Release(&p->like);
  }
  if (p->has_post) {
    PyBuffer_Release(&p->post);
  }
  return HMBUFA_destroy(&p->hmbufa);
}


static PyObject *
forward_python(PyObject *self, PyObject *args)
{
  int except = 0;
  /* init an object */
  struct HMBUFA hmbufa;
  HMBUFA_clear(&hmbufa);
  /* read the args */
  PyObject *distn;
  PyObject *trans;
  const char *likelihoods_name;
  const char *forward_name;
  const char *scaling_name;
  if (!PyArg_ParseTuple(args, "OOsss",
      &distn, &trans,
      &likelihoods_name, &forward_name, &scaling_name)) {
    except = 1; goto end;
  }
  /* do extensive error checking */
  if (HMBUFA_init(&hmbufa, distn, trans) < 0) {
    except = 1; goto end;
  }
  /* run the hmm algorithm */
  if (do_forward(&hmbufa.tm, likelihoods_name, forward_name, scaling_name))
  {
    PyErr_SetString(HmmusbufError, "forward algorithm error");
    except = 1; goto end;
  }
end:
  HMBUFA_destroy(&hmbufa);
  if (except) {
    return NULL;
  } else {
    return Py_BuildValue("i", 42);
  }
}

static PyObject *
backward_python(PyObject *self, PyObject *args)
{
  int except = 0;
  /* init an object */
  struct HMBUFA hmbufa;
  HMBUFA_clear(&hmbufa);
  /* read the args */
  PyObject *distn;
  PyObject *trans;
  const char *likelihoods_name;
  const char *scaling_name;
  const char *backward_name;
  if (!PyArg_ParseTuple(args, "OOsss",
      &distn, &trans,
      &likelihoods_name, &scaling_name, &backward_name)) {
    except = 1; goto end;
  }
  /* do extensive error checking */
  if (HMBUFA_init(&hmbufa, distn, trans) < 0) {
    except = 1; goto end;
  }
  /* run the hmm algorithm */
  if (do_backward(&hmbufa.tm, likelihoods_name, scaling_name, backward_name))
  {
    PyErr_SetString(HmmusbufError, "backward algorithm error");
    except = 1; goto end;
  }
end:
  HMBUFA_destroy(&hmbufa);
  if (except) {
    return NULL;
  } else {
    return Py_BuildValue("i", 42);
  }
}

static PyObject *
posterior_python(PyObject *self, PyObject *args)
{
  int except = 0;
  /* init an object */
  struct HMBUFA hmbufa;
  HMBUFA_clear(&hmbufa);
  /* read the args */
  PyObject *distn;
  PyObject *trans;
  const char *f_name;
  const char *s_name;
  const char *b_name;
  const char *p_name;
  if (!PyArg_ParseTuple(args, "OOssss",
      &distn, &trans,
      &f_name, &s_name, &b_name, &p_name)) {
    except = 1; goto end;
  }
  /* do extensive error checking */
  if (HMBUFA_init(&hmbufa, distn, trans) < 0) {
    except = 1; goto end;
  }
  /* run the hmm algorithm */
  if (do_posterior(hmbufa.tm.nstates, f_name, s_name, b_name, p_name))
  {
    PyErr_SetString(HmmusbufError, "posterior algorithm error");
    except = 1; goto end;
  }
end:
  HMBUFA_destroy(&hmbufa);
  if (except) {
    return NULL;
  } else {
    return Py_BuildValue("i", 42);
  }
}

static PyObject *
fwdbwd_somedisk_python(PyObject *self, PyObject *args)
{
  int except = 0;
  /* init an object */
  struct HMBUFA hmbufa;
  HMBUFA_clear(&hmbufa);
  /* read the args */
  PyObject *distn;
  PyObject *trans;
  const char *l_name;
  const char *d_name;
  if (!PyArg_ParseTuple(args, "OOss",
      &distn, &trans,
      &l_name, &d_name)) {
    except = 1; goto end;
  }
  /* do extensive error checking */
  if (HMBUFA_init(&hmbufa, distn, trans) < 0) {
    except = 1; goto end;
  }
  /* run the hmm algorithm */
  if (do_fwdbwd_somedisk(&hmbufa.tm, l_name, d_name))
  {
    PyErr_SetString(HmmusbufError, "fwdbwd_somedisk algorithm error");
    except = 1; goto end;
  }
end:
  HMBUFA_destroy(&hmbufa);
  if (except) {
    return NULL;
  } else {
    return Py_BuildValue("i", 42);
  }
}

/* Try to use the buffer interface.
 * Python args are the numpy arrays:
 * distribution_in
 * transitions_in
 * likelihoods_in
 * posterior_out
 */
static PyObject *
fwdbwd_nodisk_python(PyObject *self, PyObject *args)
{
  int except = 0;
  /* init an object */
  struct HMBUFB hmbufb;
  HMBUFB_clear(&hmbufb);
  /* read the args */
  PyObject *distn, *trans, *like, *post;
  if (!PyArg_ParseTuple(args, "OOOO", &distn, &trans, &like, &post)) {
    except = 1; goto end;
  }
  /* do extensive error checking */
  if (HMBUFB_init(&hmbufb, distn, trans, like, post) < 0) {
    except = 1; goto end;
  }
  /* fill the posterior matrix */
  if (fwdbwd_nodisk(&hmbufb.hmbufa.tm, hmbufb.like.shape[0],
      (double *) hmbufb.like.buf, (double *) hmbufb.post.buf) < 0) {
    PyErr_SetString(HmmusbufError,
        "failed to run the forward-backward algorithm");
    except = 1; goto end;
  }
end:
  HMBUFB_destroy(&hmbufb);
  /* return an appropriate value */
  if (except) {
    return NULL;
  } else {
    return Py_BuildValue("i", 42);
  }
}


static PyMethodDef HmmusbufMethods[] = {
  {"forward", forward_python, METH_VARARGS,
    "Forward algorithm "
    "using the new-style buffer interface."},
  {"backward", backward_python, METH_VARARGS,
    "Backward algorithm "
    "using the new-style buffer interface."},
  {"posterior", posterior_python, METH_VARARGS,
    "Posterior decoding"
    "using the new-style buffer interface."},
  {"fwdbwd_somedisk", fwdbwd_somedisk_python, METH_VARARGS,
    "Forward-backward algorithm with intermediate arrays in RAM, "
    "using the new-style buffer interface."},
  {"fwdbwd_nodisk", fwdbwd_nodisk_python, METH_VARARGS,
    "Forward-backward algorithm with all arrays in RAM, "
    "using the new-style buffer interface."},
  {NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC
inithmmusbuf(void) 
{
  PyObject *m = Py_InitModule("hmmusbuf", HmmusbufMethods);
  if (!m) return;
  /* init the error object */
  HmmusbufError = PyErr_NewException("hmmusbuf.error", NULL, NULL);
  Py_INCREF(HmmusbufError);
  PyModule_AddObject(m, "error", HmmusbufError);
}

