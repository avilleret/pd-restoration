#pragma once

#include <stdbool.h>
#include "denoiser2~.h"

void spectral_gain_computing(float* fft_p2,
                             float* fft_p2_prev,
                             float* fft_magnitude,
                             float* fft_magnitude_prev,
                             float time_smoothing,
                             float snr_influence,
                             float* noise_thresholds_p2,
                             float* noise_thresholds_magnitude,
                             int fft_size_2,
                             float reduction_strenght,
                             float* Gk,
                             float frequency_smoothing);

void gain_application(float amount_of_reduction,
                      int fft_size_2,
                      int fft_size,
                      float* output_fft_buffer,
                      float* Gk,
                      float makeup_gain,
                      float wet_dry,
                      float residual_whitening,
                      float noise_listen,
                      bool debug);

void gain_application_simple(t_denoiser2_tilde* nrepel);
