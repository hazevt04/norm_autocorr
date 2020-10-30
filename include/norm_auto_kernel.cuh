#pragma once

#include "my_cuda_utils.hpp"
#include "my_cufft_utils.hpp"

template<typename T>
__device__
void delay16( T* delayed_vals, const T* vals, const int num_vals );

__device__
void auto_correlation( cufftComplex* __restrict__ conj_sqrs, const cufftComplex* __restrict__ samples_d16,
   const cufftComplex* __restrict__ samples, const int num_vals );

__device__
void complex_mag_squared( float* __restrict__ mag_sqrs, const cufftComplex* __restrict__ samples, const int num_vals );

__device__
void complex_mags( float* __restrict__ mags, const cufftComplex* __restrict__ samples, const int num_vals );

__device__
void moving_averages( cufftComplex* __restrict__ conj_sqr_means, float* __restrict__ mag_sqr_means, 
      const cufftComplex* __restrict__ conj_sqrs, const float* __restrict__ mag_sqrs,
      const int conj_sqr_window_size, const int mag_sqr_window_size, const int num_vals );

__device__
void normalize( float* __restrict__ norms, const float* __restrict__ conj_sqr_mean_mags, 
   const float* __restrict__ mag_sqr_means, const int num_samples );

__global__
void norm_auto_kernel( 
   float* __restrict__ norms, 
   float* __restrict__ mag_sqr_means, 
   float* __restrict__ mag_sqrs, 
   float* __restrict__ conj_sqr_mean_mags, 
   cufftComplex* __restrict__ conj_sqr_means, 
   cufftComplex* __restrict__ conj_sqrs, 
   cufftComplex* __restrict__ samples_d16, 
   const cufftComplex* __restrict__ samples, 
   const int conj_sqr_window_size, 
   const int mag_sqr_window_size,
   const int num_samples );
