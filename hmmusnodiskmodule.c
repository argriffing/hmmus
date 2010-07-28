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
  int nstates; /* # hidden states or -1 if unknown */
  int nalpha; /* # unique emission symbols or -1 if unknown */
  int nobs; /* # observations or -1 if unknown */
  PyObject *v_obj;
  PyObject *l_obj;
  PyObject *f_obj;
  PyObject *s_obj;
  PyObject *b_obj;
  PyObject *d_obj;
  PyObject *distn_obj;
  PyObject *trans_obj;
  PyObject *emiss_obj;
  PyObject *trans_expect_obj;
  PyObject *emiss_expect_obj;
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
  p->v_obj = NULL;
  p->l_obj = NULL;
  p->f_obj = NULL;
  p->s_obj = NULL;
  p->b_obj = NULL;
  p->d_obj = NULL;
  p->distn_obj = NULL;
  p->trans_obj = NULL;
  p->emiss_obj = NULL;
  p->trans_expect_obj = NULL;
  p->emiss_expect_obj = NULL;
  return 0;
}

int safe_buffer_release(Py_buffer *pbuf)
{
  if (pbuf->buf) PyBuffer_Release(pbuf);
  return 0;
}

int baum_destroy(struct baum *p)
{
  safe_buffer_release(&p->v);
  safe_buffer_release(&p->l);
  safe_buffer_release(&p->f);
  safe_buffer_release(&p->s);
  safe_buffer_release(&p->b);
  safe_buffer_release(&p->d);
  safe_buffer_release(&p->distn);
  safe_buffer_release(&p->trans);
  safe_buffer_release(&p->emiss);
  safe_buffer_release(&p->trans_expect);
  safe_buffer_release(&p->emiss_expect);
  return 0;
}

int baum_set_nobs(struct baum *p, int nobs)
{
  char msg[1000];
  char s[] = "Args have incompatible numbers of observations: %d vs %d";
  if (p->nobs == -1) {
    p->nobs = nobs;
    return 0;
  } else if (p->nobs == nobs) {
    return 0;
  } else {
    sprintf(msg, s, p->nobs, nobs);
    PyErr_SetString(HmmusnodiskError, msg);
    return -1;
  }
}

int baum_set_nstates(struct baum *p, int nstates)
{
  char msg[1000];
  char s[] = "Args have incompatible numbers of hidden states: %d vs %d";
  if (p->nstates == -1) {
    p->nstates = nstates;
    return 0;
  } else if (p->nstates == nstates) {
    return 0;
  } else {
    sprintf(msg, s, p->nstates, nstates);
    PyErr_SetString(HmmusnodiskError, msg);
    return -1;
  }
}

int baum_set_nalpha(struct baum *p, int nalpha)
{
  char msg[1000];
  char s[] = "Args have incompatible alphabet sizes: %d vs %d";
  if (p->nalpha == -1) {
    p->nalpha = nalpha;
    return 0;
  } else if (p->nalpha == nalpha) {
    return 0;
  } else {
    sprintf(msg, s, p->nalpha, nalpha);
    PyErr_SetString(HmmusnodiskError, msg);
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
  const char *fmt = pbuf->format;
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
  const char *fmt = pbuf->format;
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

int baum_set_s(struct baum *p, PyObject *pobj)
{
  char name[] = "The scaling matrix";
  if (check_buffer_interface(pobj, name) < 0) return -1;
  if (get_buffer(pobj, &p->s) < 0) return -1;
  if (check_ndim(&p->s, 1, name) < 0) return -1;
  if (check_datatype_double(&p->s, name) < 0) return -1;
  if (baum_set_nobs(p, p->s.shape[0]) < 0) return -1;
  return 0;
}

int baum_set_b(struct baum *p, PyObject *pobj)
{
  char name[] = "The backward matrix";
  if (check_buffer_interface(pobj, name) < 0) return -1;
  if (get_buffer(pobj, &p->b) < 0) return -1;
  if (check_ndim(&p->b, 2, name) < 0) return -1;
  if (check_datatype_double(&p->b, name) < 0) return -1;
  if (baum_set_nobs(p, p->b.shape[0]) < 0) return -1;
  if (baum_set_nstates(p, p->b.shape[1]) < 0) return -1;
  return 0;
}

int baum_set_d(struct baum *p, PyObject *pobj)
{
  char name[] = "The posterior matrix";
  if (check_buffer_interface(pobj, name) < 0) return -1;
  if (get_buffer(pobj, &p->d) < 0) return -1;
  if (check_ndim(&p->d, 2, name) < 0) return -1;
  if (check_datatype_double(&p->d, name) < 0) return -1;
  if (baum_set_nobs(p, p->d.shape[0]) < 0) return -1;
  if (baum_set_nstates(p, p->d.shape[1]) < 0) return -1;
  return 0;
}

int baum_set_distn(struct baum *p, PyObject *pobj)
{
  char name[] = "The stationary distribution matrix";
  if (check_buffer_interface(pobj, name) < 0) return -1;
  if (get_buffer(pobj, &p->distn) < 0) return -1;
  if (check_ndim(&p->distn, 1, name) < 0) return -1;
  if (check_datatype_double(&p->distn, name) < 0) return -1;
  if (baum_set_nstates(p, p->distn.shape[0]) < 0) return -1;
  return 0;
}

int baum_set_trans(struct baum *p, PyObject *pobj)
{
  char name[] = "The transition matrix";
  if (check_buffer_interface(pobj, name) < 0) return -1;
  if (get_buffer(pobj, &p->trans) < 0) return -1;
  if (check_ndim(&p->trans, 2, name) < 0) return -1;
  if (check_datatype_double(&p->trans, name) < 0) return -1;
  if (baum_set_nstates(p, p->trans.shape[0]) < 0) return -1;
  if (baum_set_nstates(p, p->trans.shape[1]) < 0) return -1;
  return 0;
}

int baum_set_emiss(struct baum *p, PyObject *pobj)
{
  char name[] = "The emission probability matrix";
  if (check_buffer_interface(pobj, name) < 0) return -1;
  if (get_buffer(pobj, &p->emiss) < 0) return -1;
  if (check_ndim(&p->emiss, 2, name) < 0) return -1;
  if (check_datatype_double(&p->emiss, name) < 0) return -1;
  if (baum_set_nstates(p, p->emiss.shape[0]) < 0) return -1;
  if (baum_set_nalpha(p, p->emiss.shape[1]) < 0) return -1;
  return 0;
}

int baum_set_trans_expect(struct baum *p, PyObject *pobj)
{
  char name[] = "The transition expectation matrix";
  if (check_buffer_interface(pobj, name) < 0) return -1;
  if (get_buffer(pobj, &p->trans_expect) < 0) return -1;
  if (check_ndim(&p->trans_expect, 2, name) < 0) return -1;
  if (check_datatype_double(&p->trans_expect, name) < 0) return -1;
  if (baum_set_nstates(p, p->trans_expect.shape[0]) < 0) return -1;
  if (baum_set_nstates(p, p->trans_expect.shape[1]) < 0) return -1;
  return 0;
}

int baum_set_emiss_expect(struct baum *p, PyObject *pobj)
{
  char name[] = "The emission expectation matrix";
  if (check_buffer_interface(pobj, name) < 0) return -1;
  if (get_buffer(pobj, &p->emiss_expect) < 0) return -1;
  if (check_ndim(&p->emiss_expect, 2, name) < 0) return -1;
  if (check_datatype_double(&p->emiss_expect, name) < 0) return -1;
  if (baum_set_nstates(p, p->emiss_expect.shape[0]) < 0) return -1;
  if (baum_set_nalpha(p, p->emiss_expect.shape[1]) < 0) return -1;
  return 0;
}

int baum_read_buffers(struct baum *p)
{
  if (p->v_obj != NULL)
    if (baum_set_v(p, p->v_obj) < 0) return -1;
  if (p->l_obj != NULL)
    if (baum_set_l(p, p->l_obj) < 0) return -1;
  if (p->f_obj != NULL)
    if (baum_set_f(p, p->f_obj) < 0) return -1;
  if (p->s_obj != NULL)
    if (baum_set_s(p, p->s_obj) < 0) return -1;
  if (p->b_obj != NULL)
    if (baum_set_b(p, p->b_obj) < 0) return -1;
  if (p->d_obj != NULL)
    if (baum_set_d(p, p->d_obj) < 0) return -1;
  if (p->distn_obj != NULL)
    if (baum_set_distn(p, p->distn_obj) < 0) return -1;
  if (p->trans_obj != NULL)
    if (baum_set_trans(p, p->trans_obj) < 0) return -1;
  if (p->emiss_obj != NULL)
    if (baum_set_emiss(p, p->emiss_obj) < 0) return -1;
  if (p->trans_expect_obj != NULL)
    if (baum_set_trans_expect(p, p->trans_expect_obj) < 0) return -1;
  if (p->emiss_expect_obj != NULL)
    if (baum_set_emiss_expect(p, p->emiss_expect_obj) < 0) return -1;
  return 0;
}

static PyObject *
forward_python(PyObject *self, PyObject *args)
{
  int except = 0;
  struct baum bm;
  baum_init(&bm);
  if (!PyArg_ParseTuple(args, "OOOOO",
        &bm.distn_obj, &bm.trans_obj, &bm.l_obj, &bm.f_obj, &bm.s_obj)) {
    except = 1; goto end;
  }
  if (baum_read_buffers(&bm) < 0) {
    except = 1; goto end;
  }
  struct TM tm;
  tm.nstates = bm.nstates;
  tm.distn = bm.distn.buf;
  tm.trans = bm.trans.buf;
  if (forward_nodisk(&tm, bm.nobs, bm.l.buf, bm.f.buf, bm.s.buf) < 0) {
    PyErr_SetString(HmmusnodiskError, "forward algorithm error");
    except = 1; goto end;
  }
end:
  baum_destroy(&bm);
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
  struct baum bm;
  baum_init(&bm);
  if (!PyArg_ParseTuple(args, "OOOOO",
        &bm.distn_obj, &bm.trans_obj, &bm.l_obj, &bm.s_obj, &bm.b_obj)) {
    except = 1; goto end;
  }
  if (baum_read_buffers(&bm) < 0) {
    except = 1; goto end;
  }
  struct TM tm;
  tm.nstates = bm.nstates;
  tm.distn = bm.distn.buf;
  tm.trans = bm.trans.buf;
  if (backward_nodisk(&tm, bm.nobs, bm.l.buf, bm.s.buf, bm.b.buf) < 0) {
    PyErr_SetString(HmmusnodiskError, "backward algorithm error");
    except = 1; goto end;
  }
end:
  baum_destroy(&bm);
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
  struct baum bm;
  baum_init(&bm);
  if (!PyArg_ParseTuple(args, "OOOO",
        &bm.f_obj, &bm.s_obj, &bm.b_obj, &bm.d_obj)) {
    except = 1; goto end;
  }
  if (baum_read_buffers(&bm) < 0) {
    except = 1; goto end;
  }
  if (posterior_nodisk(bm.nstates, bm.nobs,
        bm.f.buf, bm.s.buf, bm.b.buf, bm.d.buf) < 0) {
    PyErr_SetString(HmmusnodiskError, "posterior algorithm error");
    except = 1; goto end;
  }
end:
  baum_destroy(&bm);
  if (except) {
    return NULL;
  } else {
    return Py_BuildValue("i", 42);
  }
}

static PyObject *
finite_alphabet_likelihoods_python(PyObject *self, PyObject *args)
{
  int except = 0;
  struct baum bm;
  baum_init(&bm);
  if (!PyArg_ParseTuple(args, "OOO",
        &bm.emiss_obj, &bm.v_obj, &bm.l_obj)) {
    except = 1; goto end;
  }
  if (baum_read_buffers(&bm) < 0) {
    except = 1; goto end;
  }
  if (finite_alphabet_likelihoods_nodisk(bm.nstates, bm.nalpha, bm.nobs,
        bm.emiss.buf, bm.v.buf, bm.l.buf) < 0) {
    PyErr_SetString(HmmusnodiskError, "finite_alphabet_likelihoods error");
    except = 1; goto end;
  }
end:
  baum_destroy(&bm);
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
  struct baum bm;
  baum_init(&bm);
  if (!PyArg_ParseTuple(args, "OOOOO",
        &bm.trans_obj, &bm.trans_expect_obj,
        &bm.l_obj, &bm.f_obj, &bm.b_obj)) {
    except = 1; goto end;
  }
  if (baum_read_buffers(&bm) < 0) {
    except = 1; goto end;
  }
  if (transition_expectations_nodisk(bm.nstates, bm.nobs,
        bm.trans.buf, bm.trans_expect.buf,
        bm.l.buf, bm.f.buf, bm.b.buf)) {
    PyErr_SetString(HmmusnodiskError, "transition_expectations error");
    except = 1; goto end;
  }
end:
  baum_destroy(&bm);
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
  struct baum bm;
  baum_init(&bm);
  if (!PyArg_ParseTuple(args, "OOO",
        &bm.emiss_expect_obj, &bm.v_obj, &bm.d_obj)) {
    except = 1; goto end;
  }
  if (baum_read_buffers(&bm) < 0) {
    except = 1; goto end;
  }
  if (emission_expectations_nodisk(bm.nstates, bm.nalpha, bm.nobs,
        bm.emiss_expect.buf, bm.v.buf, bm.d.buf)) {
    PyErr_SetString(HmmusnodiskError, "emissions_expectations error");
    except = 1; goto end;
  }
end:
  baum_destroy(&bm);
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
  struct baum bm;
  baum_init(&bm);
  if (!PyArg_ParseTuple(args, "O", &bm.s_obj)) {
    except = 1; goto end;
  }
  if (baum_read_buffers(&bm) < 0) {
    except = 1; goto end;
  }
  if (sequence_log_likelihood_nodisk(&log_likelihood, bm.nobs,
        bm.s.buf) < 0) {
    PyErr_SetString(HmmusnodiskError, "sequence_log_likelihood error");
    except = 1; goto end;
  }
end:
  baum_destroy(&bm);
  if (except) {
    return NULL;
  } else {
    return Py_BuildValue("d", log_likelihood);
  }
}

char probability_to_symbol(double probability)
{
  if (probability < 0.1) return '0';
  if (probability < 0.2) return '1';
  if (probability < 0.3) return '2';
  if (probability < 0.4) return '3';
  if (probability < 0.5) return '4';
  if (probability < 0.6) return '5';
  if (probability < 0.7) return '6';
  if (probability < 0.8) return '7';
  if (probability < 0.9) return '8';
  return '9';
}

static PyObject *
pretty_print_posterior_python(PyObject *self, PyObject *args)
{
  int except = 0;
  /* baum object for only the posterior */
  struct baum bm;
  baum_init(&bm);
  /* declare the output file C object initialized to NULL for safe closing */
  FILE *fout = NULL;
  /* declare the vanilla args */
  const char *obs;
  int ncols;
  const char *filename;
  /* read the args */
  if (!PyArg_ParseTuple(args, "sOis",
        &obs, &bm.d_obj, &ncols, &filename)) {
    except = 1; goto end;
  }
  if (baum_read_buffers(&bm) < 0) {
    except = 1; goto end;
  }
  /* open the file for writing */
  fout = fopen(filename, "wt");
  if (fout == NULL) {
    PyErr_SetString(HmmusnodiskError, "failed to open the output file");
    except = 1; goto end;
  }
  /* begin doing the interesting stuff */
  int nwholegroups = bm.nobs / ncols;
  int nremainder = bm.nobs % ncols;
  int ngroups = nwholegroups + (nremainder ? 1 : 0);
  int igroup;
  int icol;
  int istate;
  double probability;
  for (igroup=0; igroup<ngroups; ++igroup)
  {
    if (igroup) fputc('\n', fout);
    int current_ncols = ncols;
    if (igroup == ngroups-1 && nremainder != 0) current_ncols = nremainder;
    /* write the observation line */
    for (icol=0; icol<current_ncols; ++icol) {
      fputc(obs[igroup*ncols + icol], fout);
    }
    fputc('\n', fout);
    /* write the posterior probability lines per state */
    for (istate=0; istate<bm.nstates; ++istate) {
      for (icol=0; icol<current_ncols; ++icol) {
        int obs_offset = igroup*ncols + icol;
        probability = ((double *) bm.d.buf)[obs_offset*bm.nstates + istate];
        fputc(probability_to_symbol(probability), fout);
      }
      fputc('\n', fout);
    }
  }
end:
  baum_destroy(&bm);
  fsafeclose(fout);
  if (except) {
    return NULL;
  } else {
    return Py_BuildValue("i", 42);
  }
}

static PyObject *
pretty_print_posterior_decoding_python(PyObject *self, PyObject *args)
{
  int except = 0;
  /* baum object for only the posterior */
  struct baum bm;
  baum_init(&bm);
  /* declare the output file C object initialized to NULL for safe closing */
  FILE *fout = NULL;
  /* declare the vanilla args */
  const char *obs;
  int ncols;
  const char *filename;
  /* read the args */
  if (!PyArg_ParseTuple(args, "sOis",
        &obs, &bm.d_obj, &ncols, &filename)) {
    except = 1; goto end;
  }
  if (baum_read_buffers(&bm) < 0) {
    except = 1; goto end;
  }
  /* the number of states must be small */
  if (bm.nstates > 10) {
    PyErr_SetString(HmmusnodiskError, "too many states");
    except = 1; goto end;
  }
  /* open the file for writing */
  fout = fopen(filename, "wt");
  if (fout == NULL) {
    PyErr_SetString(HmmusnodiskError, "failed to open the output file");
    except = 1; goto end;
  }
  /* begin doing the interesting stuff */
  int nwholegroups = bm.nobs / ncols;
  int nremainder = bm.nobs % ncols;
  int ngroups = nwholegroups + (nremainder ? 1 : 0);
  int igroup;
  int icol;
  int istate;
  double probability;
  for (igroup=0; igroup<ngroups; ++igroup)
  {
    if (igroup) fputc('\n', fout);
    int current_ncols = ncols;
    if (igroup == ngroups-1 && nremainder != 0) current_ncols = nremainder;
    /* write the observation line */
    for (icol=0; icol<current_ncols; ++icol) {
      fputc(obs[igroup*ncols + icol], fout);
    }
    fputc('\n', fout);
    /* write the posterior decoding line per state */
    for (icol=0; icol<current_ncols; ++icol) {
      int obs_offset = igroup*ncols + icol;
      int best_state = -1;
      double best_prob = -1.0;
      for (istate=0; istate<bm.nstates; ++istate) {
        probability = ((double *) bm.d.buf)[obs_offset*bm.nstates + istate];
        if (probability > best_prob || best_state < 0) {
          best_state = istate;
          best_prob = probability;
        }
      }
      fputc('0' + best_state, fout);
    }
    fputc('\n', fout);
  }
end:
  baum_destroy(&bm);
  fsafeclose(fout);
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
  {"forward",
    forward_python, METH_VARARGS,
    "Do the forward algorithm."},
  {"backward",
    backward_python, METH_VARARGS,
    "Do the backward algorithm."},
  {"posterior",
    posterior_python, METH_VARARGS,
    "Do probabilistic decoding."},
  {"transition_expectations",
    transition_expectations_python, METH_VARARGS,
    "Compute the expected count of each transition."},
  {"emission_expectations",
    emission_expectations_python, METH_VARARGS,
    "Compute emission expectations for each hidden state."},
  {"sequence_log_likelihood",
    sequence_log_likelihood_python, METH_VARARGS,
    "Compute the log likelihood of the observation sequence."},
  {"pretty_print_posterior",
    pretty_print_posterior_python, METH_VARARGS,
    "Write an ascii representation of the probabilistic posterior to a file."},
  {"pretty_print_posterior_decoding",
    pretty_print_posterior_decoding_python, METH_VARARGS,
    "Write an ascii representation of the posterior decoding to a file."},
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
