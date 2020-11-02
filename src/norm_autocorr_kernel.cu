
#include "my_cufft_utils.hpp"

#include "norm_autocorr_kernel.cuh"

template<typename T>
__device__
void delay16( T* delayed_vals, const T* vals, const int num_vals ) {

   int global_index = blockDim.x * blockIdx.x + threadIdx.x;

   if ( global_index < 16 ) {
      delayed_vals[global_index] = T{0};
   } else if ( global_index < num_vals ) {
      delayed_vals[global_index] = vals[global_index-16];
   }

}


template
__device__
void delay16<cufftComplex>( cufftComplex* delayed_vals, const cufftComplex* vals, const int num_vals );


__device__
void auto_correlation( cufftComplex* __restrict__ conj_sqrs, const cufftComplex* __restrict__ samples_d16,
   const cufftComplex* __restrict__ samples, const int num_vals ) {

   int global_index = blockDim.x * blockIdx.x + threadIdx.x;
   int stride = blockDim.x * gridDim.x;

   for (int index = global_index; index < num_vals; index += stride) {
      conj_sqrs[index] = cuCmulf( samples[index], cuConjf( samples_d16[index] ) );
   }
}


__device__
void complex_mag_squared( float* __restrict__ mag_sqrs, const cufftComplex* __restrict__ samples, const int num_vals ) {

   int global_index = blockDim.x * blockIdx.x + threadIdx.x;
   int stride = blockDim.x * gridDim.x;

   for (int index = global_index; index < num_vals; index += stride) {
      float temp = cuCabsf( samples[index] );
      mag_sqrs[index] = temp * temp;
   }
}


__device__
void complex_mags( float* __restrict__ mags, const cufftComplex* __restrict__ samples, const int num_vals ) {

   int global_index = blockDim.x * blockIdx.x + threadIdx.x;
   int stride = blockDim.x * gridDim.x;

   for (int index = global_index; index < num_vals; index += stride) {
      mags[index] = cuCabsf( samples[index] );
   }
}


__device__ __inline__
cufftComplex complex_divide_by_scalar( cufftComplex cval, float scalar_divisor ) {
   return make_cuFloatComplex( cval.x/scalar_divisor, cval.y/scalar_divisor );
}

__device__
void moving_averages( 
      cufftComplex* __restrict__ conj_sqr_means, 
      float* __restrict__ mag_sqr_means, 
      const cufftComplex* __restrict__ conj_sqrs, 
      const float* __restrict__ mag_sqrs,
      const int conj_sqr_window_size, 
      const int mag_sqr_window_size, 
      const int num_vals 
   ) { 

   int global_index = blockDim.x * blockIdx.x + threadIdx.x;
   int stride = blockDim.x * gridDim.x;

   for (int index = global_index; index < num_vals; index += stride) {
      cufftComplex  t_conj_sqr_sum = make_cuFloatComplex(0.0,0.0);
      float  t_mag_sqr_sum = 0.0;
      for( int w_index = 0; w_index < conj_sqr_window_size; ++w_index ) {
         t_conj_sqr_sum = cuCaddf( t_conj_sqr_sum, conj_sqrs[index + w_index] );
      }
      for( int w_index = 0; w_index < mag_sqr_window_size; ++w_index ) {
         t_mag_sqr_sum = t_mag_sqr_sum + mag_sqrs[index + w_index];
      }
      conj_sqr_means[index] = complex_divide_by_scalar( t_conj_sqr_sum, (float)conj_sqr_window_size );
      mag_sqr_means[index] = t_mag_sqr_sum/(float)mag_sqr_window_size;
   }

}

__device__
void normalize( float* __restrict__ norms, const float* __restrict__ conj_sqr_mean_mags, 
   const float* __restrict__ mag_sqr_means, const int num_samples ) {

   int global_index = blockDim.x * blockIdx.x + threadIdx.x;
   int stride = blockDim.x * gridDim.x;

   for (int index = global_index; index < num_samples; index += stride) {
      norms[index] =  conj_sqr_mean_mags[index]/mag_sqr_means[index];  
   }

}

__global__
void norm_autocorr_kernel( 
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
   const int num_samples ) {


   delay16<cufftComplex>( samples_d16, samples, num_samples );
   auto_correlation( conj_sqrs, samples_d16, samples, num_samples );
   complex_mag_squared( mag_sqrs, samples, num_samples );
   moving_averages( conj_sqr_means, mag_sqr_means, conj_sqrs, mag_sqrs, 
      conj_sqr_window_size, mag_sqr_window_size, num_samples );
   complex_mags( conj_sqr_mean_mags, conj_sqr_means, num_samples );
   normalize( norms, conj_sqr_mean_mags, mag_sqr_means, num_samples );
}
