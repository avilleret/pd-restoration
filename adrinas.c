/*------------------------------------------------------------------------------

  ADRINAS: Automatic Detection and Removal of Impulsive Noise in Audio Signals

  This code is part of the following publication and was subject
  to peer review:

    "Automatic Detection and Removal of Impulsive Noise in Audio Signals"
    by Laurent Oudre, Image Processing On Line, 2015.
    DOI:10.5201/ipol.2015.XXXXX
    http://dx.doi.org/10.5201/ipol.2015.XXXXX

  Copyright (c) 2012-2015 Laurent Oudre <laurent.oudre@cmla.ens-cachan.fr>

  This program is free software: you can redistribute it and/or modify
  it under the terms of the GNU Affero General Public License as
  published by the Free Software Foundation, either version 3 of the
  License, or (at your option) any later version.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
  GNU Affero General Public License for more details.

  You should have received a copy of the GNU Affero General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.

------------------------------------------------------------------------------*/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "adrinas.h"

/*----------------------------------------------------------------------------*/
#define PI 3.14159265

/*----------------------------------------------------------------------------*/
#ifndef isfinite
#define isfinite(x) ( (x) * 0 == 0 )
#endif

/*----------------------------------------------------------------------------*/
/* Fatal error, print a message to standard-error output and exit */
static void error(char * msg)
{
  fprintf(stderr,"error: %s\n",msg);
  exit(EXIT_FAILURE);
}

/*----------------------------------------------------------------------------*/
/* Memory allocation, print an error and exit if fail */
static void * xmalloc(size_t size)
{
  void * p;
  if( size == 0 ) error("xmalloc: zero size");
  p = malloc(size);
  if( p == NULL ) error("xmalloc: out of memory");
  return p;
}

/*----------------------------------------------------------------------------*/
/* Allocate an int array of size N initialized to zero values */
static int * zeros_ivect(int N)
{
  int * x = xmalloc( N * sizeof(int) );
  int i = 0;

  #pragma omp parallel for private(i)
  for(i=0; i<N; i++) x[i] = 0;

  return x;
}

/*----------------------------------------------------------------------------*/
/* Allocate a float array of size N initialized to zero values */
static float * zeros_fvect(int N)
{
  float * x = xmalloc( N * sizeof(float) );
  int i = 0;

  #pragma omp parallel for private(i)
  for(i=0; i<N; i++) x[i] = 0.0;

  return x;
}

/*----------------------------------------------------------------------------*/
/* Allocate a double array of size N initialized to zero values */
static double * zeros_dvect(int N)
{
  double * x = xmalloc( N * sizeof(double) );
  int i = 0;

  #pragma omp parallel for private(i)
  for(i=0; i<N; i++) x[i] = 0.0;

  return x;
}

/*----------------------------------------------------------------------------*/
/* Allocate a matrix of double of size MxN, initialized to zero values */
static double ** zeros_dmat(int M, int N)
{
  double * bloc = xmalloc( N * M * sizeof(double) );
  double ** x = xmalloc( M * sizeof(double *) );
  int i = 0;

  #pragma omp parallel for private(i)
  for(i=0; i<N*M; i++) bloc[i]=0.0;

  for(i=0; i<M; i++) x[i] = &bloc[i*N];

  return x;
}

/*----------------------------------------------------------------------------*/
/* Return the sum of the values of an int array x of size N */
static int sum_ivect(int * x, int N)
{
  int value = 0;
  int i;

  for(i=0; i<N; i++) value += x[i];

  return value;
}

/*----------------------------------------------------------------------------*/
/* Return 1 if value is present in an ordered array x of size N, 0 otherwise */
static int ispresent(int value, int * x, int N)
{
  int i,start,end;

  if ( (value < x[0]) || (value > x[N-1]) ) return 0;

  i = start = 0;
  end = N-1;

  while( start <= end )
    {
      i = (end + start) / 2;
      if( x[i] == value ) return 1; /* the value was found */
      if( value < x[i] ) end = i - 1;
      if( value > x[i] ) start = i + 1;
    }

  return 0;
}

/*----------------------------------------------------------------------------*/
/* Returns 0 if there is an undefined value in the array of double x of size N,
   it returns 1 otherwise */
static int isdefined(double * x, int N)
{
  int i;

  for(i=0; i<N; i++)
    if( !isfinite(x[i]) ) return 0;

  return 1;
}

/*----------------------------------------------------------------------------*/
/* Estimation of AR parameters by Levinson-Durbin algorithm.

   x: the signal to be analyzed
   p: order of the AR model
   N: number of samples in the signal x
   a: output AR coefficients of the model (p+1 values) (it must be allocated)
   return value: estimation of the sigma of the error function
 */
static double ar_parameters(double * x, int p, int N, double * a)
{
  double * R = zeros_dvect(p+1);
  double * a_old = zeros_dvect(p);
  double sigma2,k;
  int i,j;

  /* estimation of the autocorrelation function */
  #pragma omp parallel for private(i,j) shared(R)
  for(i=0; i<=p; i++)
    {
      double v = 0.0;
      for(j=i; j<N; j++) v += x[j] * x[j-i];
      R[i] = v / N;
    }

  /* Levinson-Durbin algorithm */
  a[0] = a_old[0] = - R[1] / R[0];
  sigma2 = (1 - a[0]*a[0]) * R[0];
  for(j=1; j<p; j++)
    {
      /* calculation of the reflection coeff */
      k = 0.0;
      for(i=0; i<j; i++) k += a_old[i] * R[j-i];
      k = (R[j+1] + k) / sigma2;

      /* update */
      a[j] = -k;
      sigma2 = (1 - a[j]*a[j]) * sigma2;
      for(i=j-1; i>=0; i--) a[i] = a_old[i] + a[j]*a_old[j-i-1];
      for(i=0; i<=j; i++) a_old[i] = a[i];
    }
  a[0] = 1.0;
  for(i=0; i<p; i++) a[i+1] = a_old[i];

  /* free memory */
  free(a_old);
  free(R);

  return sqrt(sigma2);
}

static double ar_parameters_float(float * x, int p, int N, double * a)
{
  double * R = zeros_dvect(p+1);
  double * a_old = zeros_dvect(p);
  double sigma2,k;
  int i,j;

  /* estimation of the autocorrelation function */
  #pragma omp parallel for private(i,j) shared(R)
  for(i=0; i<=p; i++)
    {
      double v = 0.0;
      for(j=i; j<N; j++) v += x[j] * x[j-i];
      R[i] = v / N;
    }

  /* Levinson-Durbin algorithm */
  a[0] = a_old[0] = - R[1] / R[0];
  sigma2 = (1 - a[0]*a[0]) * R[0];
  for(j=1; j<p; j++)
    {
      /* calculation of the reflection coeff */
      k = 0.0;
      for(i=0; i<j; i++) k += a_old[i] * R[j-i];
      k = (R[j+1] + k) / sigma2;

      /* update */
      a[j] = -k;
      sigma2 = (1 - a[j]*a[j]) * sigma2;
      for(i=j-1; i>=0; i--) a[i] = a_old[i] + a[j]*a_old[j-i-1];
      for(i=0; i<=j; i++) a_old[i] = a[i];
    }
  a[0] = 1.0;
  for(i=0; i<p; i++) a[i+1] = a_old[i];

  /* free memory */
  free(a_old);
  free(R);

  return sqrt(sigma2);
}

/*----------------------------------------------------------------------------*/
/* Solve the matrix equation Ax=b through Cholesky decomposition.

   A is an NxN matrix
   b is a N vector
   x is the solution N vector (it must be allocated)
 */
static void solve_cholesky(double ** A, double * x, double * b, int N)
{
  double ** L = zeros_dmat(N,N);
  double *  d = zeros_dvect(N);
  double *  y = zeros_dvect(N);
  double v;
  int i,j,k;

  /* A = LDL' decomposition
         https://en.wikipedia.org/wiki/Cholesky_decomposition
     L is a lower triangular matrix
     D is a diagonal matrix
   */
  for(j=0; j<N; j++)
    {
      /* D_j = A_jj - sum_k=1^j-1 L_jk^2 D_k */
      d[j] = A[j][j];
      for(i=0; i<j; i++) d[j] -=  L[j][i] * L[j][i] * d[i];

      /* check for a singularity */
      if( d[j] == 0.0 )
        {
          fprintf(stderr,"error: singular matrix\n");
          exit(EXIT_FAILURE);
        }

      /* for i>j, L_ij = (A_ij - sum_k=1^j-1 L_ik L_jk D_k) / D_j */
      for(i=j+1; i<N; i++)
        {
          v = A[i][j];
          for(k=0; k<j; k++) v -=  L[i][k] * L[j][k] * d[k];
          L[j][i] = L[i][j] = v / d[j];
        }
    }

  /* solve Ly=b by forward substitution */
  for(i=0; i<N; i++)
    {
      y[i] = b[i];
      for(j=0; j<=i; j++) y[i] -= L[i][j] * y[j];
    }

  /* solve Lx=y by back substitution */
  for(i=N-1; i>=0; i--)
    {
      x[i] = y[i]/d[i];
      for(j=i; j<N; j++) x[i] -= L[i][j] * x[j];
    }

  /* free memory */
  free(*L);
  free(L);
  free(y);
  free(d);
}

static void solve_cholesky_float(double ** A, float * x, double * b, int N)
{
  double ** L = zeros_dmat(N,N);
  double *  d = zeros_dvect(N);
  double *  y = zeros_dvect(N);
  double v;
  int i,j,k;

  /* A = LDL' decomposition
         https://en.wikipedia.org/wiki/Cholesky_decomposition
     L is a lower triangular matrix
     D is a diagonal matrix
   */
  for(j=0; j<N; j++)
    {
      /* D_j = A_jj - sum_k=1^j-1 L_jk^2 D_k */
      d[j] = A[j][j];
      for(i=0; i<j; i++) d[j] -=  L[j][i] * L[j][i] * d[i];

      /* check for a singularity */
      if( d[j] == 0.0 )
        {
          fprintf(stderr,"error: singular matrix\n");
          exit(EXIT_FAILURE);
        }

      /* for i>j, L_ij = (A_ij - sum_k=1^j-1 L_ik L_jk D_k) / D_j */
      for(i=j+1; i<N; i++)
        {
          v = A[i][j];
          for(k=0; k<j; k++) v -=  L[i][k] * L[j][k] * d[k];
          L[j][i] = L[i][j] = v / d[j];
        }
    }

  /* solve Ly=b by forward substitution */
  for(i=0; i<N; i++)
    {
      y[i] = b[i];
      for(j=0; j<=i; j++) y[i] -= L[i][j] * y[j];
    }

  /* solve Lx=y by back substitution */
  for(i=N-1; i>=0; i--)
    {
      x[i] = y[i]/d[i];
      for(j=i; j<N; j++) x[i] -= L[i][j] * x[j];
    }

  /* free memory */
  free(*L);
  free(L);
  free(y);
  free(d);
}

/*----------------------------------------------------------------------------*/
/* Interpolate the missing samples based on an autoregressive model using
   the method by Janssen, Veldhuis & Vries.

   y: the input signal
   p: order of the AR model
   a: vector of length p, with the parameters of AR model
   t: vector of length m, with index the missing samples
   m: number of missing samples
   x: interpolated values (must be allocated)
 */
static void interpolation( double * y, int p, double * a, int * t, int m,
                           double * x )
{
  double * b = zeros_dvect(p+1);
  double ** B = zeros_dmat(m,m);
  double * d = zeros_dvect(m);
  int i,j;

  /*
   Interpolation by the algorithm described in:

   A. Janssen, R. Veldhuis, L. Vries, "Adaptive interpolation of discrete-time
   signals that can be modeled as autoregressive processes", IEEE Transactions
   on Acoustics, Speech and Signal Processing, 34(2):317-330, 1986.

   The missing samples x are set as the ones that minimize

     Q(x) = sum_k=p+1^N |x[k] + sum_l=1^p a[l]*x[k-l]|^2

   which corresponds to the sum of the absolute AR predicting error.
   The error Q(x) can also be expressed as

     Q(x) = x'Bx + 2x'd + C

   where x' is the transposed of x, C is a term that depends only on
   known samples and is thus constant, and the matrix B and vector d
   are defined as below.

   The interpolated samples are obtained by solving Bx=-d.
   */

  /* compute auxiliary vector b */
  for(i=0; i<=p; i++)
    {
      b[i] = 0.0;
      for(j=i; j<=p; j++)
        b[i] += a[j] * a[j-i];
    }

  /* compute matrix B */
  for(i=0; i<m; i++)
    for(j=i; j<m; j++)
      if( abs(t[i] - t[j]) <= p )
        B[j][i] = B[i][j] = b[ abs(t[i] - t[j]) ];

  /* compute vector d (actually -d) */
  for(i=0; i<m; i++)
    {
      d[i] = 0;
      for(j=-p; j<=p; j++)
        if( ispresent(t[i]-j,t,m) == 0 )
          d[i] -= b[abs(j)] * y[t[i] - j];
    }

  /* solve Bx=-d (the minus sign was already included in d) */
  solve_cholesky(B,x,d,m);

  /* free memory */
  free(*B);
  free(B);
  free(b);
  free(d);
}

static void interpolation_float( float * y, int p, double * a, int * t, int m,
                           float * x )
{
  double * b = zeros_dvect(p+1);
  double ** B = zeros_dmat(m,m);
  double * d = zeros_dvect(m);
  int i,j;

  /*
   Interpolation by the algorithm described in:

   A. Janssen, R. Veldhuis, L. Vries, "Adaptive interpolation of discrete-time
   signals that can be modeled as autoregressive processes", IEEE Transactions
   on Acoustics, Speech and Signal Processing, 34(2):317-330, 1986.

   The missing samples x are set as the ones that minimize

     Q(x) = sum_k=p+1^N |x[k] + sum_l=1^p a[l]*x[k-l]|^2

   which corresponds to the sum of the absolute AR predicting error.
   The error Q(x) can also be expressed as

     Q(x) = x'Bx + 2x'd + C

   where x' is the transposed of x, C is a term that depends only on
   known samples and is thus constant, and the matrix B and vector d
   are defined as below.

   The interpolated samples are obtained by solving Bx=-d.
   */

  /* compute auxiliary vector b */
  for(i=0; i<=p; i++)
    {
      b[i] = 0.0;
      for(j=i; j<=p; j++)
        b[i] += a[j] * a[j-i];
    }

  /* compute matrix B */
  for(i=0; i<m; i++)
    for(j=i; j<m; j++)
      if( abs(t[i] - t[j]) <= p )
        B[j][i] = B[i][j] = b[ abs(t[i] - t[j]) ];

  /* compute vector d (actually -d) */
  for(i=0; i<m; i++)
    {
      d[i] = 0;
      for(j=-p; j<=p; j++)
        if( ispresent(t[i]-j,t,m) == 0 )
          d[i] -= b[abs(j)] * y[t[i] - j];
    }

  /* solve Bx=-d (the minus sign was already included in d) */
  solve_cholesky_float(B,x,d,m);

  /* free memory */
  free(*B);
  free(B);
  free(b);
  free(d);
}

/*----------------------------------------------------------------------------*/
/* ADRINAS: Automatic Detection and Removal of Impulsive Noise in Audio Signals

   signal : input audio signal, an array of doubles
   N      : number of samples in the input signal
   p      : order of the AR model to be used
   K      : detection threshold
   Nw     : window size

   return value: pointer to a int signal of size N, containing burst detections
 */
int * adrinas(double * signal, int N, int p, double K, int b, int Nw)
{
  int nHop = Nw / 4;
  int nFrames = floor( (N+Nw) / nHop ) + 1;
  double * input_zeropad  = zeros_dvect( Nw + (nFrames-1) * nHop );
  double * output_zeropad = zeros_dvect( Nw + (nFrames-1) * nHop );
  int    * burst_zeropad  = zeros_ivect( Nw + (nFrames-1) * nHop );
  double * w = zeros_dvect(Nw);
  int * burst = zeros_ivect(N);
  int i;

  /* Zero-padded versions of the audio signal and of the burst*/
  #pragma omp parallel for private(i) shared(input_zeropad)
  for(i=0; i<N; i++)
    input_zeropad[ i + Nw ] = signal[i];

  /* Hamming window of size Nw */
  #pragma omp parallel for private(i) shared(w)
  for(i=0; i<Nw; i++)
    w[i] = ( 0.54 - 0.46*cos( 2*PI*i/Nw ) ) / (4*0.54);

  /* process frame by frame */
  #pragma omp parallel for private(i) shared(input_zeropad,w,output_zeropad,burst_zeropad)
  for(i=0; i<nFrames; i++)
    {
      double * frame = zeros_dvect(Nw);
      double * x = zeros_dvect(Nw);
      double * d = zeros_dvect(Nw);
      int * t    = zeros_ivect(Nw);
      int * i_t  = zeros_ivect(Nw);
      double * a = zeros_dvect(p+1);
      double sigmae;
      int j,k,m,prev;

      /* extract frame from input */
      for(j=0; j<Nw; j++)
        frame[j] = input_zeropad[ i*nHop + j ];

      /* compute AR parameters and sigma_e */
      sigmae = ar_parameters(frame,p,Nw,a);

      if( isdefined(a,p+1) )
        {
          /* compute detection signal */
          for(j=p; j<Nw; j++)
            for(k=0; k<=p; k++)
              d[j] += a[k] * frame[j-k];

          /* first step of burst detection */
          for(j=0; j<Nw; j++)
            {
              if( fabs(d[j]) > K*sigmae ) i_t[j] = 1;
              else                        i_t[j] = 0;
            }

          /* second burst detection step: merge burst separate by small gap */
          prev = -1;
          for(j=0; j<Nw; j++)
            if( i_t[j] == 1 )
              {
                if( prev>=0 && (j-prev)>1 && (j-prev)<=b )
                  for(k=prev+1; k<j; k++) i_t[k] = 1;
                prev = j;
              }
          for(j=0; j<p; j++) i_t[j] = 0;
          for(j=Nw-p; j<Nw; j++) i_t[j] = 0;

          /* fill burst index table and count detected bursts */
          for(m=0, j=p; j<Nw-p; j++)
            if( i_t[j] == 1 ) t[ m++ ] = j;

          /* if there are missing samples, interpolate them */
          if( m > 0 )
            {
              /* interpolation of missing samples */
              interpolation(frame,p,a,t,m,x);

              /* copy interpolated values to the frame */
              for(j=0; j<m; j++) frame[ t[j] ] = x[j];
            }
        }

     /* copy burst detection positions in frame to full burst output */
     if( sum_ivect(i_t,Nw) > 0 )
        for(j=0; j<Nw; j++)
          if( i_t[j] == 1 )
            burst_zeropad[i*nHop+j] = 1;

      /* add the processed frame to output using a Hamming window */
      for(j=0; j<Nw; j++)
        output_zeropad[ i*nHop + j ] += frame[j]*w[j];

      /* free memory */
      free(x);
      free(t);
      free(d);
      free(frame);
      free(a);
      free(i_t);
    }

  /* get the final output from zero-padded versions */
  #pragma omp parallel for private(i) shared(signal,output_zeropad)
  for(i=0; i<N; i++)
    signal[i] = output_zeropad[i+Nw];
  #pragma omp parallel for private(i) shared(burst,burst_zeropad)
  for(i=0; i<N; i++)
    burst[i] = burst_zeropad[i+Nw];

  /* free memory */
  free(w);
  free(input_zeropad);
  free(output_zeropad);
  free(burst_zeropad);

  return burst;
}

int * adrinas_float(float * signal, int N, int p, double K, int b, int Nw)
{
  int nHop = Nw / 4;
  int nFrames = floor( (N+Nw) / nHop ) + 1;
  float * input_zeropad  = zeros_fvect( Nw + (nFrames-1) * nHop );
  float * output_zeropad = zeros_fvect( Nw + (nFrames-1) * nHop );
  int    * burst_zeropad  = zeros_ivect( Nw + (nFrames-1) * nHop );
  float * w = zeros_fvect(Nw);
  // int * burst = zeros_ivect(N);
  int i;

  /* Zero-padded versions of the audio signal and of the burst*/
  #pragma omp parallel for private(i) shared(input_zeropad)
  for(i=0; i<N; i++)
    input_zeropad[ i + Nw ] = signal[i];

  /* Hamming window of size Nw */
  #pragma omp parallel for private(i) shared(w)
  for(i=0; i<Nw; i++)
    w[i] = ( 0.54 - 0.46*cos( 2*PI*i/Nw ) ) / (4*0.54);

  /* process frame by frame */
  #pragma omp parallel for private(i) shared(input_zeropad,w,output_zeropad,burst_zeropad)
  for(i=0; i<nFrames; i++)
    {
      float * frame = zeros_fvect(Nw);
      float * x = zeros_fvect(Nw);
      float * d = zeros_fvect(Nw);
      int * t    = zeros_ivect(Nw);
      int * i_t  = zeros_ivect(Nw);
      double * a = zeros_dvect(p+1);
      double sigmae;
      int j,k,m,prev;

      /* extract frame from input */
      for(j=0; j<Nw; j++)
        frame[j] = input_zeropad[ i*nHop + j ];

      /* compute AR parameters and sigma_e */
      sigmae = ar_parameters_float(frame,p,Nw,a);

      if( isdefined(a,p+1) )
        {
          /* compute detection signal */
          for(j=p; j<Nw; j++)
            for(k=0; k<=p; k++)
              d[j] += a[k] * frame[j-k];

          /* first step of burst detection */
          for(j=0; j<Nw; j++)
            {
              if( fabs(d[j]) > K*sigmae ) i_t[j] = 1;
              else                        i_t[j] = 0;
            }

          /* second burst detection step: merge burst separate by small gap */
          prev = -1;
          for(j=0; j<Nw; j++)
            if( i_t[j] == 1 )
              {
                if( prev>=0 && (j-prev)>1 && (j-prev)<=b )
                  for(k=prev+1; k<j; k++) i_t[k] = 1;
                prev = j;
              }
          for(j=0; j<p; j++) i_t[j] = 0;
          for(j=Nw-p; j<Nw; j++) i_t[j] = 0;

          /* fill burst index table and count detected bursts */
          for(m=0, j=p; j<Nw-p; j++)
            if( i_t[j] == 1 ) t[ m++ ] = j;

          /* if there are missing samples, interpolate them */
          if( m > 0 )
            {
              /* interpolation of missing samples */
              interpolation_float(frame,p,a,t,m,x);

              /* copy interpolated values to the frame */
              for(j=0; j<m; j++) frame[ t[j] ] = x[j];
              // for(j=0; j<m; j++) frame[ t[j] ] = 0.;
            }
        }

     /* copy burst detection positions in frame to full burst output */
     if( sum_ivect(i_t,Nw) > 0 )
        for(j=0; j<Nw; j++)
          if( i_t[j] == 1 )
            burst_zeropad[i*nHop+j] = 1;

      /* add the processed frame to output using a Hamming window */
      for(j=0; j<Nw; j++)
        output_zeropad[ i*nHop + j ] += frame[j]*w[j];

      /* free memory */
      free(x);
      free(t);
      free(d);
      free(frame);
      free(a);
      free(i_t);
    }

  /* get the final output from zero-padded versions */
  #pragma omp parallel for private(i) shared(signal,output_zeropad)
  for(i=0; i<N; i++)
    signal[i] = output_zeropad[i+Nw];
/*
#pragma omp parallel for private(i) shared(burst,burst_zeropad)
  for(i=0; i<N; i++)
    burst[i] = burst_zeropad[i+Nw];
*/
  /* free memory */
  free(w);
  free(input_zeropad);
  free(output_zeropad);
  free(burst_zeropad);

  return NULL;
}

/*----------------------------------------------------------------------------*/
