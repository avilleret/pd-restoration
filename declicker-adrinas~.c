#include "m_pd.h"
#include <stdlib.h> // free()
#include "adrinas.h"

/**
 * declicker~
 *
 * The code below is mainly copied and pasted from declicker effect from Audacity source at git commit 3651d0a.
 */

static t_class *declicker_adrinas_tilde_class;

#define WINDOW_SIZE 8192

typedef struct _declicker_adrinas_tilde {
  t_object  x_obj;
  t_sample f;

  t_outlet*x_out;

  double K;    /* First threshold */
  int b;       /* Second threshold */
  int p;       /* Order of the model */
  int Nw;      /* Window length */
  double* buffer;
  int bufsize;

} t_declicker_adrinas_tilde;

t_int *declicker_adrinas_tilde_perform(t_int *w)
{
  /* the first element is a pointer to the dataspace of this object */
  t_declicker_adrinas_tilde *x = (t_declicker_adrinas_tilde *)(w[1]);
  /* here is a pointer to the t_sample arrays that hold the input signal */
  t_sample  *in =    (t_sample *)(w[2]);
  /* here comes the signalblock that will hold the output signal */
  t_sample  *out =    (t_sample *)(w[3]);
  /* all signalblocks are of the same length */
  int        blocksize =           (int)(w[4]);

  if (x->bufsize != blocksize){
      if (x->buffer) free(x->buffer);
      x->buffer = malloc(blocksize * sizeof(double));
      x->bufsize = blocksize;
  }

  int i;
/*
#pragma omp parallel for private(i) shared(in,out)
  for (i=0;i<blocksize;i++)
      x->buffer[i] = in[i];
*/

  int* burst = adrinas_float(in, blocksize, x->p, x->K, x->b, x->Nw);
  if (burst) free(burst);
#pragma omp parallel for private(i) shared(in,out)
  for (i=0;i<blocksize;i++)
      out[i] = in[i];

  /* return a pointer to the dataspace for the next dsp-object */
  return (w+5);
}


void declicker_adrinas_tilde_dsp(t_declicker_adrinas_tilde *x, t_signal **sp)
{
  /* add declicker_adrinas_tilde_perform() to the DSP-tree;
   * the declicker_adrinas_tilde_perform() will expect "4" arguments (packed into an
   * t_int-array), which are:
   * the objects data-space, 2 signal vectors (which happen to be
   * 1 input and 1 output signals) and the length of the
   * signal vectors (all vectors are of the same length)
   */
  dsp_add(declicker_adrinas_tilde_perform, 4, x,
          sp[0]->s_vec, sp[1]->s_vec, sp[0]->s_n);
}

void declicker_adrinas_tilde_threshold(t_declicker_adrinas_tilde* x, t_float f1, t_float f2){
  x->K = f1;
  x->b = f2;
}

void declicker_adrinas_tilde_model_order(t_declicker_adrinas_tilde* x, t_float f){
  x->p = f;
}

void declicker_adrinas_tilde_window_width(t_declicker_adrinas_tilde* x, t_float f){
  x->Nw = f;
}

void declicker_adrinas_tilde_free(t_declicker_adrinas_tilde *x)
{
  if (x->buffer) free(x->buffer);
  outlet_free(x->x_out);
}

void *declicker_adrinas_tilde_new()
{
  t_declicker_adrinas_tilde *x = (t_declicker_adrinas_tilde *)pd_new(declicker_adrinas_tilde_class);

  /* create a new signal-outlet */
  x->x_out = outlet_new(&x->x_obj, &s_signal);

  x->K = 2.0;       /* First threshold */
  x->b = 20;        /* Second threshold */
  x->p = 3*100+2;   /* Order of the model */
  x->Nw = 8*x->p;   /* Window length */
  x->buffer = NULL;
  x->bufsize = 0;

  return (void *)x;
}

/**
 * define the function-space of the class
 * within a single-object external the name of this function is very special
 */
void setup_declicker0x2dadrinas_tilde(void) {
  declicker_adrinas_tilde_class = class_new(gensym("declicker-adrinas~"),
                                    (t_newmethod)declicker_adrinas_tilde_new,
                                    (t_method)declicker_adrinas_tilde_free,
                                    sizeof(t_declicker_adrinas_tilde),
                                    CLASS_DEFAULT,
                                    0);

  /* whenever the audio-engine is turned on, the "declicker_adrinas_tilde_dsp()"
   * function will get called
   */
  class_addmethod(declicker_adrinas_tilde_class,
                  (t_method)declicker_adrinas_tilde_dsp, gensym("dsp"), 0);

  class_addmethod(declicker_adrinas_tilde_class,
                  (t_method)declicker_adrinas_tilde_threshold, gensym("thresholds"), A_FLOAT, A_FLOAT, 0);

  class_addmethod(declicker_adrinas_tilde_class,
                  (t_method)declicker_adrinas_tilde_model_order, gensym("model_order"), A_FLOAT, 0);

  class_addmethod(declicker_adrinas_tilde_class,
                  (t_method)declicker_adrinas_tilde_window_width, gensym("window_width"), A_FLOAT, 0);

  CLASS_MAINSIGNALIN(declicker_adrinas_tilde_class, t_declicker_adrinas_tilde, f);
}
