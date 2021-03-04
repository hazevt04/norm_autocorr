#pragma once

#include "my_cuda_utils.hpp"
#include "my_cufft_utils.hpp"

template<typename T>
__device__
void delay16( T* delayed_vals, 
   const T* vals, 
   const int num_vals 
);


__device__
void auto_correlation( 
   cufftDoubleComplex* __restrict__ conj_sqrs, 
   const cufftDoubleComplex* __restrict__ samples_d16,
   const cufftDoubleComplex* __restrict__ samples, 
   const int num_samples 
);


__device__
void calc_conj_sqr_sums( 
   cufftDoubleComplex* __restrict__ conj_sqr_sums, 
   const cufftDoubleComplex* __restrict__ conj_sqrs, 
   const int conj_sqr_window_size, 
   const int num_windowed_conj_sqrs
);

__device__
void calc_conj_sqr_sum_mags( double* __restrict__ conj_sqr_sum_mags, 
   const cufftDoubleComplex* __restrict__ conj_sqr_sums, 
   const int num_conj_sqr_sums 
);


__device__
void calc_mag_sqrs( 
   double* __restrict__ mag_sqrs, 
   const cufftDoubleComplex* __restrict__ samples, 
   const int num_samples
);

__device__
void calc_mag_sqr_sums( 
   double* __restrict__ mag_sqr_sums, 
   const double* __restrict__ mag_sqrs,
   const int mag_sqr_window_size, 
   const int num_windowed_mag_sqrs 
);

__device__
void normalize( double* __restrict__ norms, 
   const double* __restrict__ conj_sqr_sum_mags, 
   const double* __restrict__ mag_sqr_sums, 
   const int num_mag_sqr_sums
);


__global__
void norm_autocorr_kernels( 
   double* __restrict__ norms, 
   double* __restrict__ mag_sqr_sums, 
   double* __restrict__ mag_sqrs, 
   double* __restrict__ conj_sqr_sum_mags, 
   cufftDoubleComplex* __restrict__ conj_sqr_sums, 
   cufftDoubleComplex* __restrict__ conj_sqrs, 
   cufftDoubleComplex* __restrict__ samples_d16, 
   const cufftDoubleComplex* __restrict__ samples, 
   const int conj_sqr_window_size, 
   const int mag_sqr_window_size,
   const int num_samples 
);
