#ifndef HMMGUTS
#define HMMGUTS

struct TM
{
  /*
   * a transition matrix
   * @member nstates: number of states
   * @member trans: row major matrix elements
   * @member distn: initial distribution of states
   */
  int nstates;
  double *distn;
  double *trans;
};

double* get_doubles(int ndoubles, const char *filename);

int TM_init(struct TM *p, int nstates);

int TM_init_from_names(struct TM *p,
    const char *distribution_name, const char *transitions_name);

int TM_del(struct TM *p);

double sum(double *v, int n);


int forward_innerloop(size_t pos, const struct TM *ptm,
    const double *l_curr, const double *f_prev,
    double *f_curr, double *s_curr);

int backward_innerloop(size_t pos, const struct TM *ptm,
    const double *l_prev, const double *l_curr,
    double scaling_factor, const double *b_prev, double *b_curr);


int forward_alldisk(const struct TM *ptm,
    FILE *fin_l, FILE *fout_f, FILE *fout_s);

int backward_alldisk(const struct TM *ptm,
    FILE *fin_l, FILE *fin_s, FILE *fout_b);

int posterior_alldisk(int nstates,
    FILE *fi_f, FILE *fi_s, FILE *fi_b, FILE *fo_d);


int forward_somedisk(const struct TM *ptm, FILE *fin_l,
    double *f_big, double *s_big);

int backward_somedisk(const struct TM *ptm, size_t nobs, FILE *fin_l,
    const double *s_big, double *p_big);

int posterior_somedisk(int nstates, size_t nobs,
    const double *f_big, const double *s_big, const double *b_big,
    FILE *fout_d);

int fwdbwd_somedisk(const struct TM *ptm, size_t nobs,
    FILE *fin_l, FILE *fout_d);


int forward_nodisk(const struct TM *ptm, size_t nobs,
    const double *l_big, double *f_big, double *s_big);

int backward_nodisk(const struct TM *ptm, size_t nobs,
    const double *l_big, const double *s_big, double *p_big);

int posterior_nodisk(int nstates, size_t nobs,
    const double *f_big, const double *s_big, const double *b_big,
    double *d_big);

int fwdbwd_nodisk(const struct TM *ptm, size_t nobs,
    const double *l_big, double *d_big);


int do_forward(const struct TM *ptm,
    const char *likelihoods_name, const char *forward_name,
    const char *scaling_name);

int do_backward(const struct TM *ptm,
    const char *likelihoods_name, const char *scaling_name,
    const char *backward_name);

int do_posterior(int nstates,
    const char *forward_name, const char *scaling_name,
    const char *backward_name, const char *posterior_name);

int do_fwdbwd_somedisk(const struct TM *ptm,
    const char *likelihoods_name, const char *posterior_name);


#endif
