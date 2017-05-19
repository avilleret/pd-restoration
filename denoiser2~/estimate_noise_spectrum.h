#pragma once

//This thresholds will dictate how louizou algorithm recognizes noise
void compute_auto_thresholds(float* auto_thresholds,
                             float fft_size,
                             float fft_size_2,
                             float samp_rate);

void auto_capture_noise(float* p2,
                        int fft_size_2,
                        float* noise_thresholds_p2,
                        float* noise_thresholds_magnitude,
                        float* thresh,
                        float* prev_noise_thresholds,
                        float* s_pow_spec,
                        float* prev_s_pow_spec,
                        float* p_min,
                        float* prev_p_min,
                        float* speech_p_p,
                        float* prev_speech_p_p);

void get_noise_statistics(float* fft_p2,
                          float* fft_magnitude,
                          int fft_size_2,
                          float* noise_thresholds_p2,
                          float* noise_thresholds_magnitude,
                          float* window_count);
