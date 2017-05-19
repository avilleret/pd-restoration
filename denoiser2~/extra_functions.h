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

#pragma once

#include <math.h>
#include <float.h>
#include <stdbool.h>

//Window types
#define HANN_WINDOW 0
#define HAMMING_WINDOW 1
#define BLACKMAN_WINDOW 2
#define HANN_HANN_SCALING 0.375f       //This is for overlapadd scaling
#define HAMMING_HANN_SCALING 0.385f    // 1/average(window[i]^2)
#define BLACKMAN_HANN_SCALING 0.335f

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

//AUXILIARY Functions
#define MAX(a,b) (((a) > (b)) ? (a) : (b))
#define MIN(a,b) (((a) < (b)) ? (a) : (b))

//Force already-denormal float value to zero
inline float sanitize_denormal(float value) {
  if (isnan(value)) {
    return FLT_MIN; //to avoid log errors
    //return 0.f; //to avoid log errors
  } else {
    return value;
  }

}

inline int sign(float x) {
  return (x >= 0.f ? 1.f : -1.f);
}

//-----------dB SCALE-----------

float from_dB(float gdb);

inline float to_dB(float g) {
  return (20.f*log10f(g));
}

//-----------FREQ <> INDEX OR BIN------------

inline float Index2Freq(int i, float samp_rate, int N) {
  return (float) i * (samp_rate / N / 2.f);
}

inline int Freq2Index(float freq, float samp_rate, int N) {
  return (int) (freq / (samp_rate / N / 2.f));
}

//---------SPECTRAL OPERATIONS-------------

//verifies if the spectrum is full of zeros
inline bool is_empty(float* spectrum, int N){
  int k;
  for(k = 0;k <= N; k++){
    if(spectrum[k] > FLT_MIN){
      return false;
    }
  }
  return true;
}

//finds the max value of the spectrum
inline float max_spectral_value(float* spectrum, int N){
  int k;
  float max = 0.f;
  for(k = 0; k <= N; k++){
    max = MAX(spectrum[k],max);
  }
  return max;
}

//finds the min value of the spectrum
inline float min_spectral_value(float* spectrum, int N){
  int k;
  float min = FLT_MAX;
  for(k = 0; k <= N; k++){
    min = MIN(spectrum[k],min);
  }
  return min;
}

//Mean value of a spectrum
inline float spectral_mean(int m, float* a) {
    float sum=0.f;
    for(int i=0; i<=m; i++)
        sum+=a[i];
    return(sum/(float)(m+1));
}

//Median value of a spectrum
inline float spectral_median(int n, float* x) {
    float temp;
    int i, j;
    // the following two loops sort the array x in ascending order
    for(i=0; i<n-1; i++) {
        for(j=i+1; j<n; j++) {
            if(x[j] < x[i]) {
                // swap elements
                temp = x[i];
                x[i] = x[j];
                x[j] = temp;
            }
        }
    }

    if(n%2==0) {
        // if there is an even number of elements, return mean of the two elements in the middle
        return((x[n/2] + x[n/2 - 1]) / 2.f);
    } else {
        // else return the element in the middle
        return x[n/2];
    }
}

inline float spectral_moda(int n, float* x) {
  float temp[n];
  int i,j,pos_max;
  float max;

  for(i = 0;i<n; i++) {
      temp[i]=0.f;
  }

  for(i=0; i<n; i++) {
      for(j=i; j<n; j++) {
          if(x[j] == x[i]) temp[i]++;
      }
  }

  max=temp[0];
  pos_max = 0;
  for(i=0; i<n; i++) {
      if(temp[i] > max) {
          pos_max = i;
          max=temp[i];
      }
  }
  return x[pos_max];
}

//-----------WINDOW---------------

//blackman window values computing
inline float blackman(int k, int N) {
  float p = ((float)(k))/((float)(N));
  return 0.42-0.5*cosf(2.f*M_PI*p) + 0.08*cosf(4.f*M_PI*p);
}

//hanning window values computing
inline float hanning(int k, int N) {
  float p = ((float)(k))/((float)(N));
  return 0.5 - 0.5 * cosf(2.f*M_PI*p);
}

//hamming window values computing
inline float hamming(int k, int N) {
  float p = ((float)(k))/((float)(N));
  return 0.54 - 0.46 * cosf(2.f*M_PI*p);
}

//wrapper to compute windows values
void fft_window(float* window, int N, int window_type);

//wrapper for pre and post processing windows
void fft_pre_and_post_window(float* window_input,
                             float* window_output,
                             int fft_size,
                             int window_combination,
                             float* overlap_scale_factor);

//---------------WHITENING--------------

//unnormalized Hann windows for whitening tappering
void tappering_filter_calc(float* filter, int N);
void whitening_of_spectrum(float* spectrum,float wa,int N);
void apply_tappering_filter(float* spectrum,float* filter,int N);

//---------------------SMOOTHERS--------------------------

//Spectral smoothing with rectangular boxcar or unweighted sliding-average smooth
void spectral_smoothing_MA(float* spectrum, int kernel_width,int N);
//Spectral smoothing with median filter
void spectral_smoothing_MM(float* spectrum, int kernel_width, int N);
void spectral_smoothing_MAH(float* spectrum, int kernel_width,int N);
//With quadratic coefficients
void spectral_smoothing_SG_quad(float* spectrum, int kernel_width,int N);

void spectrum_time_smoothing(int fft_size_2,
                                  float* prev_spectrum,
                                  float* spectrum,
                                  float coeff);
