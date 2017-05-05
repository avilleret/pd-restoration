#include "extra_functions.h"

inline float from_dB(float gdb) {
  return (expf(gdb/20.f*logf(10.f)));
}

//This is from wikipedia ;)
float savgol_quad_5[5] = {-0.085714,0.342857,0.485714,0.342857,-0.085714};
float savgol_quad_7[7] = {-0.095238,0.142857,0.285714,0.333333,0.285714,0.142857,-0.095238};
float savgol_quad_9[9] = {-0.090909,0.060606,0.168831,0.233766,0.255411,0.233766,0.168831,0.060606,-0.090909};
float savgol_quad_11[11] = {-0.083916,0.020979,0.102564,0.160839,0.195804,0.207459,0.195804,0.160839,0.102564,0.020979,-0.083916};
float savgol_quad_13[13] = {-0.076923,0.000000,0.062937,0.111888,0.146853,0.167832,0.174825,0.167832,0.146853,0.111888,0.062937,0.000000,-0.076923};
float savgol_quad_15[15] = {-0.070588,-0.011765,0.038009,0.078733,0.110407,0.133032,0.146606,0.151131,0.146606,0.133032,0.110407,0.078733,0.038009,-0.011765,-0.070588};
float savgol_quad_17[17] = {-0.065015,-0.018576,0.021672,0.055728,0.083591,0.105263,0.120743,0.130031,0.133127,0.130031,0.120743,0.105263,0.083591,0.055728,0.021672,-0.018576,-0.065015};
float savgol_quad_19[19] = {-0.060150,-0.022556,0.010615,0.039363,0.063689,0.083591,0.099071,0.110128,0.116762,0.118974,0.116762,0.110128,0.099071,0.083591,0.063689,0.039363,0.010615,-0.022556,-0.060150};
float savgol_quad_21[21] = {-0.055901,-0.024845,0.002942,0.027460,0.048709,0.066688,0.081399,0.092841,0.101013,0.105917,0.107551,0.105917,0.101013,0.092841,0.081399,0.066688,0.048709,0.027460,0.002942,-0.024845,-0.055901};
float savgol_quad_23[23] = {-0.052174,-0.026087,-0.002484,0.018634,0.037267,0.053416,0.067081,0.078261,0.086957,0.093168,0.096894,0.098137,0.096894,0.093168,0.086957,0.078261,0.067081,0.053416,0.037267,0.018634,-0.002484,-0.026087,-0.052174};
float savgol_quad_25[25] = {-0.048889,-0.026667,-0.006377,0.011981,0.028406,0.042899,0.055459,0.066280,0.074783,0.081546,0.086377,0.089275,0.090242,0.089275,0.086377,0.081546,0.074783,0.066280,0.055459,0.042899,0.028406,0.011981,-0.006377,-0.026667,-0.048889};
float savgol_quart_7[7] = {0.021645,-0.129870,0.324675,0.567100,0.324675,-0.129870,0.021645};
float savgol_quart_9[9] = {0.034965,-0.128205,0.069930,0.314685,0.417249,0.314685,0.069930,-0.128205,0.034965};

//wrapper to compute windows values
void fft_window(float* window, int N, int window_type) {
  int k;
  for (k = 0; k < N; k++){
    switch (window_type){
      case BLACKMAN_WINDOW:
      window[k] = blackman(k, N);
      break;
      case HANN_WINDOW:
      window[k] = hanning(k, N);
      break;
      case HAMMING_WINDOW:
      window[k] = hamming(k, N);
      break;
    }
  }
}

//wrapper for pre and post processing windows
void fft_pre_and_post_window(float* window_input,
                             float* window_output,
                             int fft_size,
                             int window_combination,
                             float* overlap_scale_factor) {
  switch(window_combination){
    case 0: // HANN-HANN
      fft_window(window_input,fft_size,0); //STFT input window
      fft_window(window_output,fft_size,0); //STFT output window
      *(overlap_scale_factor) = HANN_HANN_SCALING;
      break;
    case 1: //HAMMING-HANN
      fft_window(window_input,fft_size,1); //STFT input window
      fft_window(window_output,fft_size,0); //STFT output window
      *(overlap_scale_factor) = HAMMING_HANN_SCALING;
      break;
    case 2: //BLACKMAN-HANN
      fft_window(window_input,fft_size,2); //STFT input window
      fft_window(window_output,fft_size,0); //STFT output window
      *(overlap_scale_factor) = BLACKMAN_HANN_SCALING;
      break;
  }
}

//---------------WHITENING--------------

//unnormalized Hann windows for whitening tappering
void tappering_filter_calc(float* filter, int N) {
  int k;
  for (k = 0; k < N; k++){
    filter[k] = hamming(k, N);//Half hann window tappering in favor of high frequencies
  }
}

void whitening_of_spectrum(float* spectrum,float wa,int N){
  float whiten_factor = powf(max_spectral_value(spectrum,N),wa);
  for (int k = 0; k <= N; k++) {
    if(whiten_factor > FLT_MIN){ //Protects against division by 0
      spectrum[k] /= whiten_factor;
      if(k < N){
        spectrum[N-k] /= whiten_factor;
      }
    }
  }
}

void apply_tappering_filter(float* spectrum,float* filter,int N) {
  for (int k = 0; k <= N; k++) {
    if(spectrum[k] > FLT_MIN) {
      spectrum[k] *= filter[N-k];//Half hann window tappering in favor of high frequencies
      if(k < N) {
        spectrum[N-k] *= filter[N-k];//Half hann window tappering in favor of high frequencies
      }
    }
  }
}


//---------------------SMOOTHERS--------------------------

//Spectral smoothing with rectangular boxcar or unweighted sliding-average smooth
void spectral_smoothing_MA(float* spectrum, int kernel_width,int N){
  int k;
  float smoothing_tmp[N+1];
  float t_spectrum[N+1];

  if (kernel_width == 0) return;

  //Initialize smothingbins_tmp
  for (k = 0; k <= N; ++k) {
    t_spectrum[k] = logf(spectrum[k]);
    smoothing_tmp[k] = 0.f;//Initialize temporal spectrum
  }

  for (k = 0; k <= N; ++k) {
    const int j0 = MAX(0, k - kernel_width);
    const int j1 = MIN(N, k + kernel_width);
    for(int l = j0; l <= j1; ++l) {
      smoothing_tmp[k] += t_spectrum[l];
    }
    smoothing_tmp[k] /= (j1 - j0 + 1);
  }

  for (k = 0; k <= N; ++k){
    spectrum[k] = expf(smoothing_tmp[k]);
  }
}
//Spectral smoothing with median filter
void spectral_smoothing_MM(float* spectrum, int kernel_width, int N){
  int k;
  float smoothing_tmp[N+1];

  if (kernel_width == 0) return;

  for (k = 0; k <= N; ++k) {
    const int j0 = MAX(0, k - kernel_width);
    const int j1 = MIN(N, k + kernel_width);

    float aux[j1-j0+1];
    for(int l = j0; l <= j1; ++l) {
      aux[l] = spectrum[l];
    }
    smoothing_tmp[k] = spectral_median(j1-j0+1,aux);
  }

  for (k = 0; k <= N; ++k){
    spectrum[k] = smoothing_tmp[k];
  }
}


void spectral_smoothing_MAH(float* spectrum, int kernel_width,int N){
  int k;
  float smoothing_tmp[N+1];
  float extended[N+2*kernel_width+1];
  float window[kernel_width*2 +1];
  float win_sum = 0.f;
  fft_window(window,kernel_width*2+1,0);//Hann window

  if (kernel_width == 0) return;

  //Copy data over the extended array to contemplate edge cases
  //Initialize smothingbins_tmp
  for (k = 0; k <= N; ++k) {
    extended[k+kernel_width] = spectrum[k];
    smoothing_tmp[k] = 0.f; //Initialize with zeros
  }

  for (k = 0; k <= kernel_width*2; ++k) {
    win_sum += window[k];
  }


  for (k = 0; k <= N; ++k) {
    for(int l = 0; l <= kernel_width*2; ++l) {
      smoothing_tmp[k] += window[l]*extended[k+l]/win_sum;
    }
  }

  for (k = 0; k <= N; ++k){
    spectrum[k] = smoothing_tmp[k];
  }
}


//With quadratic coefficients
void spectral_smoothing_SG_quad(float* spectrum, int kernel_width,int N){
  int k;
  float smoothing_tmp[N+1];
  float extended[N+1+2*kernel_width];

  if (kernel_width < 2 || kernel_width > 12) return;

  //Copy data over the extended array to contemplate edge cases
  for (k = 0; k <= N; ++k) {
    extended[k+kernel_width] = spectrum[k];
    smoothing_tmp[k] = 0.f; //Initialize with zeros
  }

  for (k = 0; k <= N; ++k) {
    for(int l = 0; l <= kernel_width*2; ++l) {
      switch(kernel_width){
        case 2:
          smoothing_tmp[k] += savgol_quad_5[l]*extended[l+k];
          break;
        case 3:
          smoothing_tmp[k] += savgol_quad_7[l]*extended[l+k];
          break;
        case 4:
          smoothing_tmp[k] += savgol_quad_9[l]*extended[l+k];
          break;
        case 5:
          smoothing_tmp[k] += savgol_quad_11[l]*extended[l+k];
          break;
        case 6:
          smoothing_tmp[k] += savgol_quad_13[l]*extended[l+k];
          break;
        case 7:
          smoothing_tmp[k] += savgol_quad_15[l]*extended[l+k];
          break;
        case 8:
          smoothing_tmp[k] += savgol_quad_17[l]*extended[l+k];
          break;
        case 9:
          smoothing_tmp[k] += savgol_quad_19[l]*extended[l+k];
          break;
        case 10:
          smoothing_tmp[k] += savgol_quad_21[l]*extended[l+k];
          break;
        case 11:
          smoothing_tmp[k] += savgol_quad_23[l]*extended[l+k];
          break;
        case 12:
          smoothing_tmp[k] += savgol_quad_25[l]*extended[l+k];
          break;
      }
    }
  }

  for (k = 0; k <= N; ++k){
    spectrum[k] = smoothing_tmp[k];
  }
}


//With quadric coefficients
void spectral_smoothing_SG_quart(float* spectrum, int kernel_width,int N){
  int k;
  float smoothing_tmp[N+1];
  float extended[N+2*kernel_width+1];

  if (kernel_width < 3 || kernel_width > 4) return;

  //Initialize smothingbins_tmp
  for (k = 0; k <= N; ++k) {
    extended[k+kernel_width] = spectrum[k];
    smoothing_tmp[k] = 0.f; //Initialize with zeros
  }

  for (k = 0; k <= N; ++k) {
    for(int l = 0; l < kernel_width*2; ++l) {
      switch(kernel_width){
        case 3:
          smoothing_tmp[k] += savgol_quart_7[l]*extended[l+k];
          break;
        case 4:
          smoothing_tmp[k] += savgol_quart_9[l]*extended[l+k];
          break;
      }
    }
  }

  for (k = 0; k <= N; ++k){
    spectrum[k] = smoothing_tmp[k];
  }
}

void spectrum_time_smoothing(int fft_size_2,
                                  float* prev_spectrum,
                                  float* spectrum,
                                  float coeff){
  int k;
  for (k = 0; k <= fft_size_2; k++) {
    spectrum[k] = (1.f - coeff) * spectrum[k] + coeff * prev_spectrum[k];
  }
}
