#include "m_pd.h"
#include <stdlib.h> // malloc and free

/**
 * declicker~
 *
 * The code below is mainly copied and pasted from declicker effect from Audacity source at git commit 3651d0a.
 */

static t_class *declicker_tilde_class;

typedef struct _declicker_tilde {
  t_object  x_obj;
  t_sample f;

  t_outlet*x_out;

  int x_clickWidth;
  int x_threshold;

  t_sample* x_ms_seq;
  t_sample* x_b2;
  int x_length;

} t_declicker_tilde;

t_int *declicker_tilde_perform(t_int *w)
{
  /* the first element is a pointer to the dataspace of this object */
  t_declicker_tilde *x = (t_declicker_tilde *)(w[1]);
  /* here is a pointer to the t_sample arrays that hold the input signal */
  t_sample  *in =    (t_sample *)(w[2]);
  /* here comes the signalblock that will hold the output signal */
  t_sample  *out =    (t_sample *)(w[3]);
  /* all signalblocks are of the same length */
  int          len =           (int)(w[4]);
  if (len != x->x_length){
    if (x->x_ms_seq) free(x->x_ms_seq);
    if (x->x_b2) free(x->x_b2);
    x->x_ms_seq = malloc(len * sizeof(t_sample));
    x->x_b2 = malloc(len * sizeof(t_sample));
    x->x_length = len;
  }

  int i;
  int j;
  int left = 0;

  float msw;
  int ww;
  int sep = len / 4 + 1;
  int s2 = sep/2;

  for( i=0; i<len; i++)
    x->x_b2[i] = in[i]*in[i];

  /* Shortcut for rms - multiple passes through b2, accumulating
    * as we go.
    */
  for(i=0;i<len;i++)
    x->x_ms_seq[i]=x->x_b2[i];

  for(i=1; i < sep; i *= 2) {
    for(j=0;j<len-i; j++)
      x->x_ms_seq[j] += x->x_ms_seq[j+i];
  }

  /* Cheat by truncating sep to next-lower power of two... */
  sep = i;

  for( i=0; i<len-sep; i++ ) {
    x->x_ms_seq[i] /= sep;
  }
  /* ww runs from about 4 to x->x_clickWidth.  wrc is the reciprocal;
   * chosen so that integer roundoff doesn't clobber us.
   */
  int wrc;
  for(wrc=x->x_clickWidth/4; wrc>=1; wrc /= 2) {
    ww = x->x_clickWidth/wrc;

    for( i=0; i<len-sep; i++ ){
      msw = 0;
      for( j=0; j<ww; j++) {
        msw += x->x_b2[i+s2+j];
      }
      msw /= ww;

      if(msw >= x->x_threshold * x->x_ms_seq[i]/10) {
        if( left == 0 ) {
          left = i+s2;
        }
      } else {
        if(left != 0 && i-left+s2 <= ww*2) {
          float lv = in[left];
          float rv = in[i+ww+s2];
          for(j=left; j<i+ww+s2; j++) {
            in[j]= (rv*(j-left) + lv*(i+ww+s2-j))/(float)(i+ww+s2-left);
            x->x_b2[j] = in[j]*in[j];
          }
          left=0;
        } else if(left != 0) {
          left = 0;
        }
      }
    }
  }

  /* return a pointer to the dataspace for the next dsp-object */
  return (w+5);
}


/**
 * register a special perform-routine at the dsp-engine
 * this function gets called whenever the DSP is turned ON
 * the name of this function is registered in declicker_tilde_setup()
 */
void declicker_tilde_dsp(t_declicker_tilde *x, t_signal **sp)
{
  /* add declicker_tilde_perform() to the DSP-tree;
   * the declicker_tilde_perform() will expect "4" arguments (packed into an
   * t_int-array), which are:
   * the objects data-space, 2 signal vectors (which happen to be
   * 1 input and 1 output signals) and the length of the
   * signal vectors (all vectors are of the same length)
   */
  dsp_add(declicker_tilde_perform, 4, x,
          sp[0]->s_vec, sp[1]->s_vec, sp[0]->s_n);
}

void declicker_tilde_threshold(t_declicker_tilde* x, t_float f){
  x->x_threshold = f > 0. ? f : 0.;
}

void declicker_tilde_width(t_declicker_tilde* x, t_float f){
  x->x_clickWidth = f > 0. ? f : 0.;
}

void declicker_tilde_free(t_declicker_tilde *x)
{
  outlet_free(x->x_out);
}

void *declicker_tilde_new()
{
  t_declicker_tilde *x = (t_declicker_tilde *)pd_new(declicker_tilde_class);

  /* create a new signal-outlet */
  x->x_out = outlet_new(&x->x_obj, &s_signal);

  x->x_threshold = 200;
  x->x_clickWidth = 20;

  x->x_ms_seq = NULL;
  x->x_b2 = NULL;

  return (void *)x;
}

/**
 * define the function-space of the class
 * within a single-object external the name of this function is very special
 */
void declicker_tilde_setup(void) {
  declicker_tilde_class = class_new(gensym("declicker~"),
                                    (t_newmethod)declicker_tilde_new,
                                    (t_method)declicker_tilde_free,
                                    sizeof(t_declicker_tilde),
                                    CLASS_DEFAULT,
                                    0);

  /* whenever the audio-engine is turned on, the "declicker_tilde_dsp()"
   * function will get called
   */
  class_addmethod(declicker_tilde_class,
                  (t_method)declicker_tilde_dsp, gensym("dsp"), 0);

  class_addmethod(declicker_tilde_class,
                  (t_method)declicker_tilde_threshold, gensym("threshold"), A_FLOAT, 0);

  class_addmethod(declicker_tilde_class,
                  (t_method)declicker_tilde_width, gensym("width"), A_FLOAT, 0);

  CLASS_MAINSIGNALIN(declicker_tilde_class, t_declicker_tilde, f);
}
