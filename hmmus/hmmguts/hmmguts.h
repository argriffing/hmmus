#ifndef HMMGUTS
#define HMMGUTS

struct TM
{
  /*
   * a transition matrix
   * @member order: number of states
   * @member value: row major matrix elements
   * @member initial_distn: initial distribution of states
   */
  int order;
  double *value;
  double *initial_distn;
};

double* get_doubles(int ndoubles, const char *filename);

int TM_init(struct TM *p, int order);

int TM_init_from_names(struct TM *p, int nstates,
    const char *distribution_name, const char *transitions_name);

int TM_del(struct TM *p);

double sum(double *v, int n);

int forward(struct TM *ptm, FILE *fin_l, FILE *fout_f, FILE *fout_s);

int backward(struct TM *ptm, FILE *fin_l, FILE *fin_s, FILE *fout_b);

int posterior(struct TM *ptm, FILE *fi_f, FILE *fi_s, FILE *fi_b, FILE *fo_d);

int do_forward(struct TM *ptm,
    const char *likelihoods_name, const char *forward_name,
    const char *scaling_name);

int do_backward(struct TM *ptm,
    const char *likelihoods_name, const char *scaling_name,
    const char *backward_name);

int do_posterior(struct TM *ptm,
    const char *forward_name, const char *scaling_name,
    const char *backward_name, const char *posterior_name);

#endif
