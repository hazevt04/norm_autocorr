#pragma once

#include "my_cufft_utils.hpp"

__global__
void norm_auto_kernel( 
   float* __restrict__ norms, float* __restrict__ mag_sqr_means, float* __restrict__ mag_sqrs, float* __restrict__ conj_sqr_mean_mags, cufftComplex* __restrict__ conj_sqr_means, cufftComplex* __restrict__ conj_sqrs, cufftComplex* __restrict__ samples, const int num_samples );
