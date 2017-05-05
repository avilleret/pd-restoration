#pragma once

//Non linear Power Sustraction
void nonlinear_power_sustraction(float reduction_strenght,
                                 float snr_influence,
                                 int fft_size_2,
                                 float* spectrum,
                                 float* noise_thresholds,
                                 float* Gk);


