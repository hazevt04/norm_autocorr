#pragma once

#include "my_cuda_utils.hpp"
#include "my_cufft_utils.hpp"

__device__ __host__ __inline__
cufftComplex complex_divide_by_scalar( cufftComplex cval, float scalar_divisor ) {
   return make_cuFloatComplex( cval.x/scalar_divisor, cval.y/scalar_divisor );
}


template<typename T>
__device__
void delay16( T* delayed_vals, const T* vals, const int num_vals );


__device__
void auto_correlation( 
   cufftComplex* __restrict__ conj_sqrs, 
   const cufftComplex* __restrict__ samples_d16,
   const cufftComplex* __restrict__ samples, 
   const int num_vals 
);


__device__
void calc_conj_sqr_win_sums( 
   cufftComplex* __restrict__ conj_sqr_win_sums, 
   const cufftComplex* __restrict__ conj_sqrs, 
   const int conj_sqr_window_size, 
   const int num_vals 
);

__device__
void calc_conj_sqr_win_sum_mags( float* __restrict__ conj_sqr_win_sum_mags, const cufftComplex* __restrict__ conj_sqr_win_sums, const int num_vals );


__device__
void calc_mag_sqrs( 
   float* __restrict__ mag_sqrs, 
   const cufftComplex* __restrict__ samples, 
   const int num_vals 
);

__device__
void calc_mag_sqr_win_sums( 
   float* __restrict__ mag_sqr_win_sums, 
   const float* __restrict__ mag_sqrs,
   const int mag_sqr_window_size, 
   const int num_vals 
);

__device__
void normalize( float* __restrict__ norms, const float* __restrict__ conj_sqr_win_sum_mags, 
   const float* __restrict__ mag_sqr_win_sums, const int num_samples );


__global__
void norm_autocorr_kernel( 
   float* __restrict__ norms, 
   float* __restrict__ mag_sqr_win_sums, 
   float* __restrict__ mag_sqrs, 
   float* __restrict__ conj_sqr_win_sum_mags, 
   cufftComplex* __restrict__ conj_sqr_win_sums, 
   cufftComplex* __restrict__ conj_sqrs, 
   cufftComplex* __restrict__ samples_d16, 
   const cufftComplex* __restrict__ samples, 
   const int conj_sqr_window_size, 
   const int mag_sqr_window_size,
   const int num_samples 
);
