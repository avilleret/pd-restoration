#pragma once
#include "m_pd.h"
#include <stdbool.h>
#include <fftw3.h>

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

    //Gain application buffer
    float* denoised_fft_buffer;
    float* residual_spectrum;
    float* tappering_filter;

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
