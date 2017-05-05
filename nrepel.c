/*
noise-repellent -- Noise Reduction LV2

Copyright 2016 Luciano Dato <lucianodato@gmail.com>

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/
*/


#include <stdlib.h>
#include <string.h>
#include <fftw3.h>
#include <time.h>
#include <stdio.h>

#include "m_pd.h"
#include "extra_functions.h"
#include "spectral_processing.h"
#include "estimate_noise_spectrum.h"

//STFT default values (These are standard values)
#define WINDOW_COMBINATION 0          //0 HANN-HANN 1 HAMMING-HANN 2 BLACKMAN-HANN
#define OVERLAP_FACTOR 4              //4 is 75% overlap

///---------------------------------------------------------------------

static t_class *denoiser2_tilde_class;

typedef enum {
    NREPEL_CAPTURE = 0,
    NREPEL_N_AUTO = 1,
    NREPEL_AMOUNT = 2,
    NREPEL_SCALE = 3,
    NREPEL_STRENGTH = 4,
    NREPEL_SMOOTHING = 5,
    NREPEL_FREQUENCY_SMOOTHING = 6,
    NREPEL_LATENCY = 7,
    NREPEL_WHITENING = 8,
    NREPEL_MAKEUP = 9,
    NREPEL_RESET = 10,
    NREPEL_NOISE_LISTEN = 11,
    NREPEL_ENABLE = 12,
    NREPEL_INPUT = 13,
    NREPEL_OUTPUT = 14,
} PortIndex;

typedef struct _denoiser2_tilde {
    t_object x_obj;
    t_outlet* outlet;
    t_sample f;
    bool debug;

    float samp_rate;                  //Sample rate received from the host

    //Parameters for the algorithm (user input)
    bool  capture_state;            //Capture Noise state (Manual-Off-Auto)
    float amount_of_reduction;      //Amount of noise to reduce in dB
    float snr_influence;       	  	//Scale of reduction for nonlinear_power_sustraction
    float reduction_strenght;        //Second Oversustraction factor
    // float* report_latency;            //Latency necessary
    bool   reset_print;               //Reset Noise switch
    float noise_listen;              //For noise only listening
    float residual_whitening;        //Whitening of the residual spectrum
    float time_smoothing;            //constant that set the time smoothing coefficient
    bool  auto_state;                //autocapture switch
    float frequency_smoothing;       //Smoothing over frequency
    // float* masking;                   //Activate masking threshold
    float  enable;                    //For soft bypass (click free bypass)
    float makeup_gain;

    //Control variables
    bool noise_thresholds_availables; //indicate whether a noise print is available or no


    //Parameters values and arrays for the STFT
    int fft_size;                     //FFTW input size
    int fft_size_2;                   //FFTW half input size
    int window_combination;           //Window combination for the STFT
    float overlap_factor;             //oversampling factor for overlap calculations
    float overlap_scale_factor;       //Scaling factor for conserving the final amplitude
    int hop;                          //Hop size for the STFT
    float* window_input;              //Input Window values
    float* window_output;             //Input Window values
    float window_count;              //Count windows for mean computing
    float tau;                        //time constant for soft bypass
    float wet_dry_target;             //softbypass target for softbypass
    float wet_dry;                    //softbypass coeff
    float reduction_coeff;            //Gain to apply to the residual noise

    //Buffers for processing and outputting
    int input_latency;
    float* in_fifo;                   //internal input buffer
    float* out_fifo;                  //internal output buffer
    float* output_accum;              //FFT output accumulator
    int read_ptr;                     //buffers read pointer

    //FFTW related arrays
    float* input_fft_buffer;
    float* output_fft_buffer;
    fftwf_plan forward;
    fftwf_plan backward;

    //Arrays and variables for getting bins info
    float real_p,imag_n,mag,p2;
    float* fft_magnitude;             //magnitude spctrum
    float* fft_magnitude_prev;        //magnitude spectrum of the previous frame
    float* fft_p2;                    //power spectrum
    float* fft_p2_prev;               //power spectum of previous frame
    float* noise_thresholds_p2;       //captured noise print power spectrum
    float* noise_thresholds_magnitude;//captured noise print magnitude spectrum

    float* Gk;                        //gain to be applied

    //Loizou algorithm
    float* auto_thresholds;           //Reference threshold for louizou algorithm
    float* prev_noise_thresholds;
    float* s_pow_spec;
    float* prev_s_pow_spec;
    float* p_min;
    float* prev_p_min;
    float* speech_p_p;
    float* prev_speech_p_p;

    // clock_t start, end;
    // double cpu_time_used;
} t_denoiser2_tilde;

void denoiser2_tilde_free(t_denoiser2_tilde* x){
    free(x->in_fifo);
    free(x->out_fifo);
    free(x->output_accum);
    free(x->window_input);
    free(x->window_output);
    free(x->input_fft_buffer);
    free(x->output_fft_buffer);
    free(x->fft_magnitude);
    free(x->fft_magnitude_prev);
    free(x->fft_p2);
    free(x->fft_p2_prev);
    free(x->noise_thresholds_p2);
    free(x->noise_thresholds_magnitude);
    free(x->Gk);
    free(x->auto_thresholds);
    free(x->prev_noise_thresholds);
    free(x->s_pow_spec);
    free(x->prev_s_pow_spec);
    free(x->p_min);
    free(x->prev_p_min);
    free(x->speech_p_p);
    free(x->prev_speech_p_p);
    outlet_free(x->outlet);
}

void denoiser2_tilde_allocate_buffer(t_denoiser2_tilde* x){
    //Initialize variables
    x->fft_size_2 = x->fft_size/2;
    x->hop = x->fft_size/x->overlap_factor;
    x->input_latency = x->fft_size - x->hop;
    x->read_ptr = x->input_latency; //the initial position because we are that many samples ahead
    x->tau = (1.f - exp (-2.f * M_PI * 25.f * 64.f  / x->samp_rate));
    x->window_count = 0.f;
    x->noise_thresholds_availables = false;

    free(x->in_fifo);
    free(x->out_fifo);
    free(x->output_accum);
    free(x->window_input);
    free(x->window_output);
    free(x->input_fft_buffer);
    free(x->output_fft_buffer);
    free(x->fft_magnitude);
    free(x->fft_magnitude_prev);
    free(x->fft_p2);
    free(x->fft_p2_prev);
    free(x->noise_thresholds_p2);
    free(x->noise_thresholds_magnitude);
    free(x->Gk);
    free(x->auto_thresholds);
    free(x->prev_noise_thresholds);
    free(x->s_pow_spec);
    free(x->prev_s_pow_spec);
    free(x->p_min);
    free(x->prev_p_min);
    free(x->speech_p_p);
    free(x->prev_speech_p_p);

    x->in_fifo = NULL;
    x->out_fifo = NULL;
    x->output_accum = NULL;

    x->window_input =  NULL;
    x->window_output = NULL;

    x->input_fft_buffer = NULL;
    x->output_fft_buffer = NULL;

    x->fft_magnitude = NULL;
    x->fft_magnitude_prev = NULL;
    x->fft_p2 = NULL;
    x->fft_p2_prev = NULL;
    x->noise_thresholds_p2 = NULL;
    x->noise_thresholds_magnitude = NULL;

    x->Gk = NULL;

    x->auto_thresholds = NULL;
    x->prev_noise_thresholds = NULL;
    x->s_pow_spec = NULL;
    x->prev_s_pow_spec = NULL;
    x->p_min = NULL;
    x->prev_p_min = NULL;
    x->speech_p_p = NULL;
    x->prev_speech_p_p = NULL;

    x->in_fifo = (float*)calloc(x->fft_size,sizeof(float));
    x->out_fifo = (float*)calloc(x->fft_size,sizeof(float));
    x->output_accum = (float*)calloc(x->fft_size,sizeof(float));

    x->window_input = (float*)calloc(x->fft_size,sizeof(float));
    x->window_output = (float*)calloc(x->fft_size,sizeof(float));

    x->input_fft_buffer = (float*)calloc(x->fft_size,sizeof(float));
    x->output_fft_buffer = (float*)calloc(x->fft_size,sizeof(float));
    x->forward = fftwf_plan_r2r_1d(x->fft_size, x->input_fft_buffer, x->output_fft_buffer, FFTW_R2HC, FFTW_ESTIMATE);
    x->backward = fftwf_plan_r2r_1d(x->fft_size, x->output_fft_buffer, x->input_fft_buffer, FFTW_HC2R, FFTW_ESTIMATE);

    x->fft_magnitude = (float*)calloc((x->fft_size_2+1),sizeof(float));
    x->fft_magnitude_prev = (float*)calloc((x->fft_size_2+1),sizeof(float));
    x->fft_p2 = (float*)calloc((x->fft_size_2+1),sizeof(float));
    x->fft_p2_prev = (float*)calloc((x->fft_size_2+1),sizeof(float));
    x->noise_thresholds_p2 = (float*)calloc((x->fft_size_2+1),sizeof(float));
    x->noise_thresholds_magnitude = (float*)calloc((x->fft_size_2+1),sizeof(float));

    x->Gk = (float*)calloc((x->fft_size_2+1),sizeof(float));

    x->auto_thresholds = (float*)calloc((x->fft_size_2+1),sizeof(float));
    x->prev_noise_thresholds = (float*)calloc((x->fft_size_2+1),sizeof(float));
    x->s_pow_spec = (float*)calloc((x->fft_size_2+1),sizeof(float));
    x->prev_s_pow_spec = (float*)calloc((x->fft_size_2+1),sizeof(float));
    x->p_min = (float*)calloc((x->fft_size_2+1),sizeof(float));
    x->prev_p_min = (float*)calloc((x->fft_size_2+1),sizeof(float));
    x->speech_p_p = (float*)calloc((x->fft_size_2+1),sizeof(float));
    x->prev_speech_p_p = (float*)calloc((x->fft_size_2+1),sizeof(float));

    //Window combination initialization (pre processing window post processing window)
    fft_pre_and_post_window(x->window_input,
                                                    x->window_output,
                                                    x->fft_size,
                                                    x->window_combination,
                                                    &x->overlap_scale_factor);

    //Set initial gain as unity
    memset(x->Gk, 1, (x->fft_size_2+1)*sizeof(float));

    //Compute auto mode initial thresholds
    compute_auto_thresholds(x->auto_thresholds, x->fft_size, x->fft_size_2, x->samp_rate);
}

void * denoiser2_tilde_new(){
    //Actual struct declaration
    t_denoiser2_tilde *x = (t_denoiser2_tilde*) pd_new(denoiser2_tilde_class);

    x->samp_rate = 0;
    x->fft_size = 0;
    x->in_fifo = NULL;
    x->out_fifo = NULL;
    x->output_accum = NULL;

    x->window_input =  NULL;
    x->window_output = NULL;

    x->input_fft_buffer = NULL;
    x->output_fft_buffer = NULL;

    x->fft_magnitude = NULL;
    x->fft_magnitude_prev = NULL;
    x->fft_p2 = NULL;
    x->fft_p2_prev = NULL;
    x->noise_thresholds_p2 = NULL;
    x->noise_thresholds_magnitude = NULL;

    x->Gk = NULL;

    x->auto_thresholds = NULL;
    x->prev_noise_thresholds = NULL;
    x->s_pow_spec = NULL;
    x->prev_s_pow_spec = NULL;
    x->p_min = NULL;
    x->prev_p_min = NULL;
    x->speech_p_p = NULL;
    x->prev_speech_p_p = NULL;

    // default parameters values
    x->capture_state = false;
    x->auto_state = false;
    x->amount_of_reduction = 0.;
    x->snr_influence = 0.;
    x->reduction_strenght = 1.0;
    x->time_smoothing = 0.;
    x->frequency_smoothing = 0.;
    x->residual_whitening = 0.;
    x->makeup_gain = 0.;
    x->reset_print = false;
    x->noise_listen = false;
    x->enable = true;

    x->window_combination = WINDOW_COMBINATION;
    x->overlap_factor = OVERLAP_FACTOR;

    x->wet_dry = 1.f;

    /* create a new signal-outlet */
    x->outlet = outlet_new(&x->x_obj, &s_signal);

    x->debug = false;

    return (void*) x;
}

void debug_print(t_denoiser2_tilde* x){
    if (x->debug){
        printf("debug: \n");
        for (int i=0;i<x->fft_size_2+1;i++){
            printf("%.4f %.4f | ", x->noise_thresholds_p2[i], x->noise_thresholds_magnitude[i]);
            if ((i+1)%64 == 0) printf("\n");
        }
        printf("\n");
    }
}

t_int *denoiser2_tilde_perform(t_int *w){
    t_denoiser2_tilde* nrepel = (t_denoiser2_tilde*)(w[1]);
    t_sample* in  = (t_sample*)(w[2]);
    t_sample* out = (t_sample*)(w[3]);
    int n_samples = (int)(w[4]);

    // //Time execution measurement
    // nrepel->start = clock();
    // //--------------

    //handy variables
    int k;
    int pos;

    //Inform latency at run call
    // *(nrepel->report_latency) = (float) nrepel->input_latency;

    //Softbypass targets in case of disabled or enabled
    if(nrepel->enable == 0.f){ //if disabled
        nrepel->wet_dry_target = 0.f;
    } else { //if enabled
        nrepel->wet_dry_target = 1.f;
    }

    //Interpolate parameters over time softly to bypass without clicks or pops
    // nrepel->wet_dry += nrepel->tau * (nrepel->wet_dry_target - nrepel->wet_dry) + FLT_MIN;

    //Reset button state (if on)
    if ( nrepel->reset_print ) {
        memset(nrepel->noise_thresholds_p2, 0, (nrepel->fft_size_2+1)*sizeof(float));
        memset(nrepel->noise_thresholds_magnitude, 0, (nrepel->fft_size_2+1)*sizeof(float));
        memset(nrepel->Gk, 1, (nrepel->fft_size_2+1)*sizeof(float));
        nrepel->window_count = 0.f;

        memset(nrepel->prev_noise_thresholds, 0, (nrepel->fft_size_2+1)*sizeof(float));
        memset(nrepel->s_pow_spec, 0, (nrepel->fft_size_2+1)*sizeof(float));
        memset(nrepel->prev_s_pow_spec, 0, (nrepel->fft_size_2+1)*sizeof(float));
        memset(nrepel->p_min, 0, (nrepel->fft_size_2+1)*sizeof(float));
        memset(nrepel->prev_p_min, 0, (nrepel->fft_size_2+1)*sizeof(float));
        memset(nrepel->speech_p_p, 0, (nrepel->fft_size_2+1)*sizeof(float));
        memset(nrepel->prev_speech_p_p, 0, (nrepel->fft_size_2+1)*sizeof(float));

        nrepel->noise_thresholds_availables = false;

        nrepel->reset_print = false;
    }

    //main loop for processing
    for (pos = 0; pos < n_samples; pos++){

        //Store samples int the input buffer
        nrepel->in_fifo[nrepel->read_ptr] = in[pos];
        //Output samples in the output buffer (even zeros introduced by latency)
        out[pos] = nrepel->out_fifo[nrepel->read_ptr - nrepel->input_latency];
        //Now move the read pointer
        nrepel->read_ptr++;

        //Once the buffer is full we can do stuff
        if (nrepel->read_ptr >= nrepel->fft_size){
            //Reset the input buffer position
            nrepel->read_ptr = nrepel->input_latency;

            //Apply windowing
            for (k = 0; k < nrepel->fft_size; k++){
                nrepel->input_fft_buffer[k] = nrepel->in_fifo[k] * nrepel->window_input[k];
            }

            //----------FFT Analysis------------

            //Do transform
            fftwf_execute(nrepel->forward);

            //-----------GET INFO FROM BINS--------------

            //Store previous magnitude and power values
            for (k = 0; k <= nrepel->fft_size_2; k++){
                nrepel->fft_p2_prev[k] = nrepel->fft_p2[k]; //store previous value for smoothing
                nrepel->fft_magnitude_prev[k] = nrepel->fft_magnitude[k]; //store previous value for smoothing
            }

            //Get the positive spectrum and compute the magnitude
            for (k = 0; k <= nrepel->fft_size_2; k++){
                //Get the half complex spectrum reals and complex
                nrepel->real_p = nrepel->output_fft_buffer[k];
                nrepel->imag_n = nrepel->output_fft_buffer[nrepel->fft_size-k];

                //Get the magnitude and power spectrum
                if(k < nrepel->fft_size){
                    nrepel->p2 = (nrepel->real_p*nrepel->real_p + nrepel->imag_n*nrepel->imag_n);
                    nrepel->mag = sqrtf(nrepel->p2);//sqrt(real^2+imag^2)
                } else {
                    //Nyquist - this is due to half complex transform look at http://www.fftw.org/doc/The-Halfcomplex_002dformat-DFT.html
                    nrepel->p2 = nrepel->real_p*nrepel->real_p;
                    nrepel->mag = nrepel->real_p;
                }
                //Store values in magnitude and power arrays
                nrepel->fft_p2[k] = nrepel->p2;
                nrepel->fft_magnitude[k] = nrepel->mag;
            }

            /////////////////////SPECTRAL PROCESSING//////////////////////////

            //If the spectrum is not silence
            if(!is_empty(nrepel->fft_magnitude,nrepel->fft_size_2)){
                //If autolearn is selected allways estimate noise_thresholds using Loizou
                if(nrepel->auto_state) {
                    auto_capture_noise(nrepel->fft_p2,//this is supposed to be the power spectrum in Loizou method
                                       nrepel->fft_size_2,
                                       nrepel->noise_thresholds_p2,
                                       nrepel->noise_thresholds_magnitude,
                                       nrepel->auto_thresholds,
                                       nrepel->prev_noise_thresholds,
                                       nrepel->s_pow_spec,
                                       nrepel->prev_s_pow_spec,
                                       nrepel->p_min,
                                       nrepel->prev_p_min,
                                       nrepel->speech_p_p,
                                       nrepel->prev_speech_p_p);

                    nrepel->noise_thresholds_availables = true;
                    // debug_print(nrepel);
                }

                /*If selected estimate noise spectrum is based on selected portion of signal
                 *do not process the signal
                 */
                if(nrepel->capture_state) { //MANUAL
                    get_noise_statistics(nrepel->fft_p2,
                                         nrepel->fft_magnitude,
                                         nrepel->fft_size_2,
                                         nrepel->noise_thresholds_p2,
                                         nrepel->noise_thresholds_magnitude,
                                         &nrepel->window_count);

                    nrepel->noise_thresholds_availables = true;
                    // debug_print(nrepel);
                } else {
                    //If there is a noise profile reduce noise
                    if (nrepel->noise_thresholds_availables == true) {
                        // TODO something is wrong here because there is no sound
                        //Gain Calculation
                        spectral_gain_computing(nrepel->fft_p2,
                                                nrepel->fft_p2_prev,
                                                nrepel->fft_magnitude,
                                                nrepel->fft_magnitude_prev,
                                                nrepel->time_smoothing,
                                                nrepel->snr_influence,
                                                nrepel->noise_thresholds_p2,
                                                nrepel->noise_thresholds_magnitude,
                                                nrepel->fft_size_2,
                                                nrepel->reduction_strenght,
                                                nrepel->Gk,
                                                nrepel->frequency_smoothing);

                        //Gain Application
                        gain_application(nrepel->amount_of_reduction,
                                         nrepel->fft_size_2,
                                         nrepel->fft_size,
                                         nrepel->output_fft_buffer,
                                         nrepel->Gk,
                                         nrepel->makeup_gain,
                                         nrepel->wet_dry,
                                         nrepel->residual_whitening,
                                         nrepel->noise_listen,
                                         nrepel->debug);

                    }
                }
            }

            ///////////////////////////////////////////////////////////

            //Normalize values to obtain correct magnitude and power values
            for (k = 0; k < nrepel->fft_size; k++){
                nrepel->output_fft_buffer[k] /= nrepel->fft_size;
            }


            //------------FFT Synthesis-------------

            //Do inverse transform
            fftwf_execute(nrepel->backward);

            //------------OVERLAPADD-------------

            //Accumulate (Overlapadd)
            for(k = 0; k < nrepel->fft_size; k++){
                nrepel->output_accum[k] += nrepel->window_output[k]*nrepel->input_fft_buffer[k]/( nrepel->overlap_scale_factor * nrepel->overlap_factor);
            }

            //Output samples up to the hop size (using makeup gain setted by the user)
            for (k = 0; k < nrepel->hop; k++){
                nrepel->out_fifo[k] = nrepel->output_accum[k];
            }

            //shift FFT accumulator the hop size
            memmove(nrepel->output_accum, nrepel->output_accum + nrepel->hop, nrepel->fft_size*sizeof(float));

            //Make sure that the non overlaping section is 0
            for (k = (nrepel->fft_size-nrepel->hop); k < nrepel->fft_size; k++){
                nrepel->output_accum[k] = 0.f;
            }

            //move input FIFO
            for (k = 0; k < nrepel->input_latency; k++){
                nrepel->in_fifo[k] = nrepel->in_fifo[k+nrepel->hop];
            }
            //-------------------------------
        }//if
    }//main loop

    // //Time measurement
    // nrepel->end = clock();
    // nrepel->cpu_time_used = ((double) (nrepel->end - nrepel->start)) / CLOCKS_PER_SEC;
    //
    // //To string
    // char buffer[50];
    // sprintf(buffer,"%lf",nrepel->cpu_time_used);
    // strcat(buffer,"\n");
    //
    // //Saving results to a file
    // FILE *fp;
    //
    // fp = fopen("resuts.txt", "a");
    // fputs(buffer, fp);
    // fclose(fp);

    return (w+5);
}

void denoiser2_tilde_dsp(t_denoiser2_tilde *x, t_signal **sp)
{
    logpost(x,2,"samplerate: %0.2f, vector size: %d", sp[0]->s_sr, sp[0]->s_n);
    if (sp[0]->s_sr != x->samp_rate){
        x->samp_rate=sp[0]->s_sr;
        x->tau = (1.f - exp (-2.f * M_PI * 25.f * 64.f  / x->samp_rate));
        compute_auto_thresholds(x->auto_thresholds, x->fft_size, x->fft_size_2, x->samp_rate);
    }
    if (sp[0]->s_n != x->fft_size){
        x->fft_size = sp[0]->s_n;
        denoiser2_tilde_allocate_buffer(x);
    }
  dsp_add(denoiser2_tilde_perform, 4, x,
          sp[0]->s_vec, sp[1]->s_vec, sp[0]->s_n);
}

void denoiser2_tilde_param_mess(t_denoiser2_tilde* x, t_symbol* s, int ac, t_atom* av){
    if(strcmp(s->s_name,"noise_capture") == 0){
        if (ac > 0 && av[0].a_type == A_FLOAT)
            x->capture_state = atom_getfloat(av)>0;
        else pd_error(x,"wrong argument to message '%s'", s->s_name);
    } else if (strcmp(s->s_name, "noise_auto_capture") == 0) {
        if (ac > 0 && av[0].a_type == A_FLOAT)
            x->auto_state = atom_getfloat(av)>0;
        else pd_error(x,"wrong argument to message '%s'", s->s_name);
    } else if (strcmp(s->s_name, "reduction_amount") == 0) {
        if (ac > 0 && av[0].a_type == A_FLOAT){
            float val = atom_getfloat(av);
            val = val > 48.f ? 48.f : val;
            val = val < 0.f ? 0.f : val;
            x->amount_of_reduction = val;
        } else pd_error(x,"wrong argument to message '%s'", s->s_name);
    } else if (strcmp(s->s_name, "snr_influence") == 0) {
        if (ac > 0 && av[0].a_type == A_FLOAT){
            float val = atom_getfloat(av);
            val = val > 3.f ? 3.f : val;
            val = val < 0.f ? 0.f : val;
            x->snr_influence = val;
        } else pd_error(x,"wrong argument to message '%s'", s->s_name);
    } else if (strcmp(s->s_name, "reduction_strenght") == 0) {
        if (ac > 0 && av[0].a_type == A_FLOAT){
            float val = atom_getfloat(av);
            val = val > 6.f ? 6.f : val;
            val = val < 0.5f ? 0.5f : val;
            x->reduction_strenght = val;
        } else pd_error(x,"wrong argument to message '%s'", s->s_name);
    } else if (strcmp(s->s_name, "time_smoothing") == 0) {
        if (ac > 0 && av[0].a_type == A_FLOAT){
            float val = atom_getfloat(av);
            val = val > 1.f ? 1.f : val;
            val = val < 0.f ? 0.f : val;
            x->time_smoothing = val;
        } else pd_error(x,"wrong argument to message '%s'", s->s_name);
    } else if (strcmp(s->s_name, "frequency_smoothing") == 0) {
        if (ac > 0 && av[0].a_type == A_FLOAT){
            float val = atom_getfloat(av);
            val = val > 20.f ? 20.f : val;
            val = val < 0.f ? 0.f : val;
            x->frequency_smoothing = val;
        } else pd_error(x,"wrong argument to message '%s'", s->s_name);
    } else if (strcmp(s->s_name, "residual_whitening") == 0) {
        if (ac > 0 && av[0].a_type == A_FLOAT){
            float val = atom_getfloat(av);
            val = val > 1.f ? 1.f : val;
            val = val < 0.f ? 0.f : val;
            x->residual_whitening = val;
        } else pd_error(x,"wrong argument to message '%s'", s->s_name);
    } else if (strcmp(s->s_name, "makeup_gain") == 0) {
        if (ac > 0 && av[0].a_type == A_FLOAT){
            float val = atom_getfloat(av);
            val = val > 20.f ? 20.f : val;
            val = val < 0.f ? 0.f : val;
            x->makeup_gain = val;
            logpost(x,2,"makeup_gain: %.2f", x->makeup_gain);
        } else pd_error(x,"wrong argument to message '%s'", s->s_name);
    } else if (strcmp(s->s_name, "reset_print") == 0) {
            x->reset_print = true;
    } else if (strcmp(s->s_name, "noise_listen") == 0) {
        if (ac > 0 && av[0].a_type == A_FLOAT)
            x->noise_listen = atom_getfloat(av)>0;
        else pd_error(x,"wrong argument to message '%s'", s->s_name);
    } else if (strcmp(s->s_name, "enable") == 0) {
        if (ac > 0 && av[0].a_type == A_FLOAT)
            x->enable = atom_getfloat(av)>0;
        else pd_error(x,"wrong argument to message '%s'", s->s_name);
    } else if (strcmp(s->s_name, "wet_dry") == 0) {
        if (ac > 0 && av[0].a_type == A_FLOAT){
            float val = atom_getfloat(av);
            val = val > 20.f ? 20.f : val;
            val = val < 0.f ? 0.f : val;
            x->wet_dry = val;
        } else pd_error(x,"wrong argument to message '%s'", s->s_name);
    } else if (strcmp(s->s_name, "debug") == 0){
        if (ac > 0 && av[0].a_type == A_FLOAT)
            x->debug = atom_getfloat(av)>0;
        else pd_error(x,"wrong argument to message '%s'", s->s_name);
    } else if (strcmp(s->s_name, "set") == 0){
        if (ac > 1 && av[0].a_type == A_SYMBOL && av[1].a_type == A_SYMBOL){
            t_garray *a;
            t_word *vec;
            int arraysize;

            // set p2 array
            t_symbol* arrayname = atom_getsymbol(av);
            if (!(a = (t_garray *)pd_findbyclass(arrayname, garray_class)))
            {
                if (*arrayname->s_name) pd_error(x, "%s: no such array",
                                         arrayname->s_name);
            } else if (!garray_getfloatwords(a, &arraysize, &vec))
            {
                pd_error(x, "%s: can't read array", arrayname->s_name);
            } else if (arraysize!=x->fft_size_2+1){
                pd_error(x, "%s array size (%d) doesn't match internal buffer size (%d)", arrayname->s_name, arraysize, x->fft_size_2+1);
            } else {
                for (int i=0;i<arraysize;i++){
                    x->noise_thresholds_p2[i] = vec[i].w_float;
                }
            }

            // get magnitude array
            arrayname = atom_getsymbol(av+1);
            if (!(a = (t_garray *)pd_findbyclass(arrayname, garray_class)))
            {
                if (*arrayname->s_name) pd_error(x, "%s: no such array",
                                         arrayname->s_name);
            } else if (!garray_getfloatwords(a, &arraysize, &vec))
            {
                pd_error(x, "%s: can't read array", arrayname->s_name);
            } else if (arraysize!=x->fft_size_2+1){
                pd_error(x, "%s array size (%d) doesn't match internal buffer size (%d)", arrayname->s_name, arraysize, x->fft_size_2+1);
            } else {
                for (int i=0;i<arraysize;i++){
                    x->noise_thresholds_magnitude[i] = vec[i].w_float;
                }
            }
            x->noise_thresholds_availables = true;
        } else {
            pd_error(x,"set message need 2 array name as argument");
        }
    } else if (strcmp(s->s_name, "get") == 0){
        if (ac > 1 && av[0].a_type == A_SYMBOL && av[1].a_type == A_SYMBOL){
            t_garray *a;
            t_word *vec;
            int arraysize;

            // set p2 array
            t_symbol* arrayname = atom_getsymbol(av);
            if (!(a = (t_garray *)pd_findbyclass(arrayname, garray_class)))
            {
                if (*arrayname->s_name) pd_error(x, "%s: no such array",
                                         arrayname->s_name);
            } else if (!garray_getfloatwords(a, &arraysize, &vec))
            {
                pd_error(x, "%s: can't read array", arrayname->s_name);
            } else if (arraysize!=x->fft_size_2+1){
                pd_error(x, "%s array size (%d) doesn't match internal buffer size (%d)", arrayname->s_name, arraysize, x->fft_size_2+1);
            } else {
                for (int i=0;i<arraysize;i++){
                     vec[i].w_float = x->noise_thresholds_p2[i];
                }
            }
            garray_redraw(a);

            // get magnitude array
            arrayname = atom_getsymbol(av+1);
            if (!(a = (t_garray *)pd_findbyclass(arrayname, garray_class)))
            {
                if (*arrayname->s_name) pd_error(x, "%s: no such array",
                                         arrayname->s_name);
            } else if (!garray_getfloatwords(a, &arraysize, &vec))
            {
                pd_error(x, "%s: can't read array", arrayname->s_name);
            } else if (arraysize!=x->fft_size_2+1){
                pd_error(x, "%s array size (%d) doesn't match internal buffer size (%d)", arrayname->s_name, arraysize, x->fft_size_2+1);
            } else {
                for (int i=0;i<arraysize;i++){
                    vec[i].w_float = x->noise_thresholds_magnitude[i];
                }
            }
            garray_redraw(a);
        } else {
            pd_error(x,"'get' message need 2 array name as argument");
        }
    } else {
        pd_error(x, "doesn't understand message \'%s\'", s->s_name);
    }
}

/**
 * define the function-space of the class
 * within a single-object external the name of this function is very special
 */
void denoiser2_tilde_setup(void) {
  denoiser2_tilde_class = class_new(gensym("denoiser2~"),
        (t_newmethod)denoiser2_tilde_new,
        (t_method)denoiser2_tilde_free,
        sizeof(t_denoiser2_tilde),
        CLASS_DEFAULT, 0);

  class_addmethod(denoiser2_tilde_class,
        (t_method)denoiser2_tilde_dsp, gensym("dsp"), 0);
  CLASS_MAINSIGNALIN(denoiser2_tilde_class, t_denoiser2_tilde, f);

  class_addanything(denoiser2_tilde_class, (t_method)denoiser2_tilde_param_mess);
}
