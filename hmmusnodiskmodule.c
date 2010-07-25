#include <Python.h>

#include "hmmguts/hmmguts.h"

static PyObject *HmmusnodiskError;

struct baum {
  Py_buffer v; /* big observation vector */
  Py_buffer l; /* big likelihood matrix */
  Py_buffer f; /* big forward matrix */
  Py_buffer s; /* big scaling vector */
  Py_buffer b; /* big backward matrix */
  Py_buffer d; /* big posterior matrix */
  Py_buffer distn; /* small input distribution vector */
  Py_buffer trans; /* small input transition matrix */
  Py_buffer emiss; /* small input emission matrix */
  Py_buffer trans_expect; /* small output transition expectation matrix */
  Py_buffer emiss_expect; /* small output emission expectation matrix */
  int nstates; /* # hidden states; -1 if unknown; -2 if error */
  int nalpha; /* # unique emission symbols; -1 if unknown; -2 if error */
  int nobs; /* # observations; -1 if unknown; -2 if error */
};

int baum_init(struct baum *p)
{
  p->v.buf = NULL;
  p->l.buf = NULL;
  p->f.buf = NULL;
  p->s.buf = NULL;
  p->b.buf = NULL;
  p->d.buf = NULL;
  p->distn.buf = NULL;
  p->trans.buf = NULL;
  p->emiss.buf = NULL;
  p->trans_expect.buf = NULL;
  p->emiss_expect.buf = NULL;
  p->nstates = -1;
  p->nalpha = -1;
  p->nobs = -1;
}

int baum_set_nobs(struct baum *p, int nobs)
{
  if (p->nobs == -1) {
    p->nobs = nobs;
    return 0;
  } else if (p->nobs == nobs) {
    return 0;
  } else {
    PyErr_SetString(HmmusnodiskError,
        "The inputs imply incompatible numbers of observations.");
    return -1;
  }
}

int baum_set_nstates(struct baum *p, int nstates)
{
  if (p->nstates == -1) {
    p->nstates = nstates;
    return 0;
  } else if (p->nstates == nstates) {
    return 0;
  } else {
    PyErr_SetString(HmmusnodiskError,
        "The inputs imply incompatible numbers of hidden states.");
    return -1;
  }
}

int baum_set_nalpha(struct baum *p, int nalpha)
{
  if (p->nalpha == -1) {
    p->nalpha = nalpha;
    return 0;
  } else if (p->nalpha == nalpha) {
    return 0;
  } else {
    PyErr_SetString(HmmusnodiskError,
        "The inputs imply incompatible finite observation alphabet sizes.");
    return -1;
  }
}

int check_buffer_interface(PyObject *pobj, const char *name)
{
  char msg[1000];
  if (!PyObject_CheckBuffer(pobj)) {
    sprintf(msg, "%s should support the buffer interface.", name);
    PyErr_SetString(HmmusnodiskError, msg);
    return -1;
  }
  return 0;
}

int get_buffer(PyObject *pobj, Py_buffer *pbuf)
{
  int flags = PyBUF_ND | PyBUF_FORMAT | PyBUF_C_CONTIGUOUS | PyBUF_WRITABLE;
  return PyObject_GetBuffer(pobj, pbuf, flags);
}

int check_ndim(Py_buffer *pbuf, int ndim, const char *name)
{
  char msg[1000];
  if (pbuf->ndim != ndim)
  {
    sprintf(msg, "%s should be %dd.", name, ndim);
    PyErr_SetString(HmmusnodiskError, msg);
    return -1;
  }
  return 0;
}

int check_datatype_byte(Py_buffer *pbuf, const char *name)
{
  char msg[1000];
  const char fmt = pbuf->format;
  if (fmt == NULL) {
    sprintf(msg, "%s has an undefined data type.", name);
    PyErr_SetString(HmmusnodiskError, msg);
    return -1;
  } else if (fmt[0] == 'b' || fmt[0] == 'B') {
    return 0;
  } else {
    sprintf(msg, "%s should hold 8-bit elements.", name);
    PyErr_SetString(HmmusnodiskError, msg);
    return -1;
  }
}

int check_datatype_double(Py_buffer *pbuf, const char *name)
{
  char msg[1000];
  const char fmt = pbuf->format;
  if (fmt == NULL) {
    sprintf(msg, "%s has an undefined data type.", name);
    PyErr_SetString(HmmusnodiskError, msg);
    return -1;
  } else if (fmt[0] == 'd') {
    return 0;
  } else {
    sprintf(msg, "%s should hold 64-bit float elements.", name);
    PyErr_SetString(HmmusnodiskError, msg);
    return -1;
  }
}

int baum_set_v(struct baum *p, PyObject *pobj)
{
  char name[] = "The observation vector";
  if (check_buffer_interface(pobj, name) < 0) return -1;
  if (get_buffer(pobj, &p->v) < 0) return -1;
  if (check_ndim(&p->v, 1, name) < 0) return -1;
  if (check_datatype_byte(&p->v, name) < 0) return -1;
  if (baum_set_nobs(p, p->v.shape[0]) < 0) return -1;
  return 0;
}

int baum_set_l(struct baum *p, PyObject *pobj)
{
  char name[] = "The likelihood matrix";
  if (check_buffer_interface(pobj, name) < 0) return -1;
  if (get_buffer(pobj, &p->l) < 0) return -1;
  if (check_ndim(&p->l, 2, name) < 0) return -1;
  if (check_datatype_double(&p->l, name) < 0) return -1;
  if (baum_set_nobs(p, p->l.shape[0]) < 0) return -1;
  if (baum_set_nstates(p, p->l.shape[1]) < 0) return -1;
  return 0;
}

int baum_set_f(struct baum *p, PyObject *pobj)
{
  char name[] = "The forward matrix";
  if (check_buffer_interface(pobj, name) < 0) return -1;
  if (get_buffer(pobj, &p->f) < 0) return -1;
  if (check_ndim(&p->f, 2, name) < 0) return -1;
  if (check_datatype_double(&p->f, name) < 0) return -1;
  if (baum_set_nobs(p, p->f.shape[0]) < 0) return -1;
  if (baum_set_nstates(p, p->f.shape[1]) < 0) return -1;
  return 0;
}

int baum_get_nstates(struct baum *p, int *pnstates)
{
  int retval = baum_get_nstates(p);
  if (retval == -1) {
    PyErr_SetString(HmmusnodiskError,
        "The number of hidden states could not be determined.");
    return -1;
  } else if (retval == -2) {
    PyErr_SetString(HmmusnodiskError,
        "The input arrays have incompatible numbers of hidden states.");
    return -1;
  } else {
    *pnstates = retval;
    return 0;
  }
}


int get_emission_buffer()
{
  /* assert that the objects support the buffer interface */
  PyObject *pyobjects[] = {emission_obj};
  char *names[] = {"the emission matrix"};
  if (check_interfaces(1, pyobjects, names) < 0) {
    except = 1; goto end;
  }
  /* get the buffer view for each object */
  int flags = PyBUF_ND | PyBUF_FORMAT | PyBUF_C_CONTIGUOUS | PyBUF_WRITABLE;
  if (PyObject_GetBuffer(emission_obj, &emission_buffer, flags) < 0) {
    except = 1; goto end;
  } else {
    got_emission_buffer = 1;
  }
  /* check the buffer shapes */
  if (emission_buffer.ndim != 2)
  {
    PyErr_SetString(HmmusbufError,
        "the emission matrix should be two dimensional");
    except = 1; goto end;
  }
}

static PyObject *
finite_alphabet_likelihoods_python(PyObject *self, PyObject *args)
{
  int except = 0;
  /* init buffer info */
  int got_emission_buffer = 0;
  Py_buffer emission_buffer;
  /* read the args */
  PyObject *emission_obj;
  const char *v_name;
  const char *l_name;
  if (!PyArg_ParseTuple(args, "Oss",
        &emission_obj, &v_name, &l_name)) {
    except = 1; goto end;
  }
  /* assert that the objects support the buffer interface */
  PyObject *pyobjects[] = {emission_obj};
  char *names[] = {"the emission matrix"};
  if (check_interfaces(1, pyobjects, names) < 0) {
    except = 1; goto end;
  }
  /* get the buffer view for each object */
  int flags = PyBUF_ND | PyBUF_FORMAT | PyBUF_C_CONTIGUOUS | PyBUF_WRITABLE;
  if (PyObject_GetBuffer(emission_obj, &emission_buffer, flags) < 0) {
    except = 1; goto end;
  } else {
    got_emission_buffer = 1;
  }
  /* check the buffer shapes */
  if (emission_buffer.ndim != 2)
  {
    PyErr_SetString(HmmusbufError,
        "the emission matrix should be two dimensional");
    except = 1; goto end;
  }
  /* run the algorithm */
  int nstates = emission_buffer.shape[0];
  int nalpha = emission_buffer.shape[1];
  if (do_finite_alphabet_likelihoods(nstates, nalpha, emission_buffer.buf,
        v_name, l_name))
  {
    PyErr_SetString(HmmusbufError, "finite_alphabet_likelihoods error");
    except = 1; goto end;
  }
end:
  /* cleanup */
  if (got_emission_buffer) {
    PyBuffer_Release(&emission_buffer);
  }
  /* return an appropriate value */
  if (except) {
    return NULL;
  } else {
    return Py_BuildValue("i", 42);
  }
}

static PyObject *
state_expectations_python(PyObject *self, PyObject *args)
{
  int except = 0;
  /* init buffer info */
  int got_expectations_buffer = 0;
  Py_buffer expectations_buffer;
  /* read the args */
  PyObject *expectations_obj;
  const char *d_name;
  if (!PyArg_ParseTuple(args, "Os",
        &expectations_obj, &d_name)) {
    except = 1; goto end;
  }
  /* assert that the objects support the buffer interface */
  PyObject *pyobjects[] = {expectations_obj};
  char *names[] = {"the expectations vector"};
  if (check_interfaces(1, pyobjects, names) < 0) {
    except = 1; goto end;
  }
  /* get the buffer view for each object */
  int flags = PyBUF_ND | PyBUF_FORMAT | PyBUF_C_CONTIGUOUS | PyBUF_WRITABLE;
  if (PyObject_GetBuffer(expectations_obj, &expectations_buffer, flags) < 0) {
    except = 1; goto end;
  } else {
    got_expectations_buffer = 1;
  }
  /* check the buffer shapes */
  if (expectations_buffer.ndim != 1)
  {
    PyErr_SetString(HmmusbufError,
        "the expectations vector should be one dimensional");
    except = 1; goto end;
  }
  /* run the algorithm */
  int nstates = expectations_buffer.shape[0];
  if (do_state_expectations(nstates, expectations_buffer.buf, d_name))
  {
    PyErr_SetString(HmmusbufError, "state_expectations error");
    except = 1; goto end;
  }
end:
  /* cleanup */
  if (got_expectations_buffer) {
    PyBuffer_Release(&expectations_buffer);
  }
  /* return an appropriate value */
  if (except) {
    return NULL;
  } else {
    return Py_BuildValue("i", 42);
  }
}

static PyObject *
transition_expectations_python(PyObject *self, PyObject *args)
{
  int except = 0;
  /* init buffer info */
  int got_expectations_buffer = 0;
  int got_trans_buffer = 0;
  Py_buffer expectations_buffer;
  Py_buffer trans_buffer;
  /* read the args */
  PyObject *expectations_obj;
  PyObject *trans_obj;
  const char *l_name;
  const char *f_name;
  const char *b_name;
  if (!PyArg_ParseTuple(args, "OOsss",
        &trans_obj, &expectations_obj, 
        &l_name, &f_name, &b_name)) {
    except = 1; goto end;
  }
  /* assert that the objects support the buffer interface */
  PyObject *pyobjects[] = {trans_obj, expectations_obj};
  char *names[] = {"the transition matrix", "the expectations matrix"};
  if (check_interfaces(2, pyobjects, names) < 0) {
    except = 1; goto end;
  }
  /* get the buffer view for each object */
  int flags = PyBUF_ND | PyBUF_FORMAT | PyBUF_C_CONTIGUOUS | PyBUF_WRITABLE;
  if (PyObject_GetBuffer(trans_obj, &trans_buffer, flags) < 0) {
    except = 1; goto end;
  } else {
    got_trans_buffer = 1;
  }
  if (PyObject_GetBuffer(expectations_obj, &expectations_buffer, flags) < 0) {
    except = 1; goto end;
  } else {
    got_expectations_buffer = 1;
  }
  /* check the buffer shapes */
  if (trans_buffer.ndim != 2) {
    PyErr_SetString(HmmusbufError,
        "the transition matrix should be two dimensional");
    except = 1; goto end;
  }
  if (expectations_buffer.ndim != 2) {
    PyErr_SetString(HmmusbufError,
        "the expectations matrix should be two dimensional");
    except = 1; goto end;
  }
  if (trans_buffer.shape[0] != trans_buffer.shape[1])
  {
    PyErr_SetString(HmmusbufError,
        "the transition matrix should be square");
    except = 1; goto end;
  }
  if (expectations_buffer.shape[0] != expectations_buffer.shape[1])
  {
    PyErr_SetString(HmmusbufError,
        "the expectations matrix should be square");
    except = 1; goto end;
  }
  if (trans_buffer.shape[0] != expectations_buffer.shape[0])
  {
    PyErr_SetString(HmmusbufError,
        "the transition and expectations matrices should be the same size");
    except = 1; goto end;
  }
  /* run the algorithm */
  int nstates = expectations_buffer.shape[0];
  if (do_transition_expectations(nstates,
        trans_buffer.buf, expectations_buffer.buf,
        l_name, f_name, b_name))
  {
    PyErr_SetString(HmmusbufError, "transition_expectations error");
    except = 1; goto end;
  }
end:
  /* cleanup */
  if (got_trans_buffer) {
    PyBuffer_Release(&trans_buffer);
  }
  if (got_expectations_buffer) {
    PyBuffer_Release(&expectations_buffer);
  }
  /* return an appropriate value */
  if (except) {
    return NULL;
  } else {
    return Py_BuildValue("i", 42);
  }
}

static PyObject *
emission_expectations_python(PyObject *self, PyObject *args)
{
  int except = 0;
  /* init buffer info */
  int got_expectations_buffer = 0;
  Py_buffer expectations_buffer;
  /* read the args */
  PyObject *expectations_obj;
  const char *v_name;
  const char *d_name;
  if (!PyArg_ParseTuple(args, "Oss",
        &expectations_obj, &v_name, &d_name)) {
    except = 1; goto end;
  }
  /* assert that the objects support the buffer interface */
  PyObject *pyobjects[] = {expectations_obj};
  char *names[] = {"the expectations matrix"};
  if (check_interfaces(1, pyobjects, names) < 0) {
    except = 1; goto end;
  }
  /* get the buffer view for each object */
  int flags = PyBUF_ND | PyBUF_FORMAT | PyBUF_C_CONTIGUOUS | PyBUF_WRITABLE;
  if (PyObject_GetBuffer(expectations_obj, &expectations_buffer, flags) < 0) {
    except = 1; goto end;
  } else {
    got_expectations_buffer = 1;
  }
  /* check the buffer shapes */
  if (expectations_buffer.ndim != 2)
  {
    PyErr_SetString(HmmusbufError,
        "the expectations matrix should be two dimensional");
    except = 1; goto end;
  }
  /* run the algorithm */
  int nstates = expectations_buffer.shape[0];
  int nalpha = expectations_buffer.shape[1];
  if (do_emission_expectations(nstates, nalpha, expectations_buffer.buf,
        v_name, d_name))
  {
    PyErr_SetString(HmmusbufError, "emissions_expectations error");
    except = 1; goto end;
  }
end:
  /* cleanup */
  if (got_expectations_buffer) {
    PyBuffer_Release(&expectations_buffer);
  }
  /* return an appropriate value */
  if (except) {
    return NULL;
  } else {
    return Py_BuildValue("i", 42);
  }
}

static PyObject *
sequence_log_likelihood_python(PyObject *self, PyObject *args)
{
  int except = 0;
  double log_likelihood = 0.0;
  /* read the args */
  const char *s_name;
  if (!PyArg_ParseTuple(args, "s", &s_name))
  {
    except = 1; goto end;
  }
  /* get the log likelihood */
  if (do_sequence_log_likelihood(&log_likelihood, s_name))
  {
    PyErr_SetString(HmmusbufError, "sequence_log_likelihood error");
    except = 1; goto end;
  }
end:
  if (except) {
    return NULL;
  } else {
    return Py_BuildValue("d", log_likelihood);
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


static PyMethodDef HmmusnodiskMethods[] = {
  {"finite_alphabet_likelihoods",
    finite_alphabet_likelihoods_python, METH_VARARGS,
    "Compute the likelihoods at each position of the observation vector."},
  {"forward_backward",
    forward_backward_python, METH_VARARGS,
    "Do the forward-backward algorithm."},
  {"transition_expectations",
    transition_expectations_python, METH_VARARGS,
    "Compute the expected count of each transition."},
  {"emission_expectations",
    emission_expectations_python, METH_VARARGS,
    "Compute emission expectations for each hidden state."},
  {"sequence_log_likelihood",
    sequence_log_likelihood_python, METH_VARARGS,
    "Compute the log likelihood of the observation sequence."},
  {NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC
inithmmusnodisk(void) 
{
  PyObject *m = Py_InitModule("hmmusnodisk", HmmusnodiskMethods);
  if (!m) return;
  /* init the error object */
  HmmusnodiskError = PyErr_NewException("hmmusnodisk.error", NULL, NULL);
  Py_INCREF(HmmusnodiskError);
  PyModule_AddObject(m, "error", HmmusnodiskError);
}
