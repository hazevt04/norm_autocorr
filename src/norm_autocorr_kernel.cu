
#include <stdio.h>

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
void auto_correlations( cufftComplex* __restrict__ conj_sqrs, const cufftComplex* __restrict__ samples_d16,
   const cufftComplex* __restrict__ samples, const int num_vals ) {

   int thread_index = threadIdx.x;

   printf( "%s(): blockIdx.x is %d. thread_index is %d\n", __func__, blockIdx.x, thread_index );
   conj_sqrs[thread_index] = cuCmulf( samples[thread_index], cuConjf( samples_d16[thread_index] ) );
}


__device__
void calc_conj_sqr_means( 
      cufftComplex* __restrict__ conj_sqr_means, 
      const cufftComplex* __restrict__ conj_sqrs, 
      const int conj_sqr_window_size, 
      const int num_vals 
   ) { 

   int global_index = blockDim.x * blockIdx.x + threadIdx.x;
   int stride = blockDim.x * gridDim.x;

   for (int index = global_index; index < num_vals; index += stride) {
      cufftComplex  t_conj_sqr_sum = make_cuFloatComplex(0.0,0.0);

      for( int w_index = 0; w_index < conj_sqr_window_size; ++w_index ) {
         t_conj_sqr_sum = cuCaddf( t_conj_sqr_sum, conj_sqrs[index + w_index] );
      }
      conj_sqr_means[index] = complex_divide_by_scalar( t_conj_sqr_sum, (float)conj_sqr_window_size );
   }

}


__device__
void calc_conj_sqr_means_smem( 
      cufftComplex* __restrict__ conj_sqr_means, 
      const cufftComplex* __restrict__ conj_sqrs, 
      const int conj_sqr_window_size, 
      const int num_vals 
   ) { 

   int global_index = blockDim.x * blockIdx.x + threadIdx.x;
   int thread_index = threadIdx.x;

   if ( thread_index < (blockDim.x - conj_sqr_window_size)  ) {
      //cufftComplex  t_conj_sqr_sum = make_cuFloatComplex(0.0,0.0);
      conj_sqr_means[thread_index] = make_cuFloatComplex(0.0,0.0);

      for( int w_index = 0; w_index < conj_sqr_window_size; ++w_index ) {
         __syncthreads();
         //t_conj_sqr_sum = cuCaddf( t_conj_sqr_sum, conj_sqrs[thread_index + w_index] );
         conj_sqr_means[thread_index] = cuCaddf( conj_sqr_means[thread_index], conj_sqrs[thread_index + w_index] );
      }
      __syncthreads();
      //conj_sqr_means[thread_index] = complex_divide_by_scalar( t_conj_sqr_sum, (float)conj_sqr_window_size );
      conj_sqr_means[thread_index] = complex_divide_by_scalar( conj_sqr_means[thread_index], (float)conj_sqr_window_size );
   }
}


__device__
void calc_conj_sqr_mean_mags( float* __restrict__ conj_sqr_mean_mags, const cufftComplex* __restrict__ conj_sqr_means, const int num_vals ) {

   int global_index = blockDim.x * blockIdx.x + threadIdx.x;
   int stride = blockDim.x * gridDim.x;

   for (int index = global_index; index < num_vals; index += stride) {
      conj_sqr_mean_mags[index] = cuCabsf( conj_sqr_means[index] );
   }
}


__device__
void calc_conj_sqr_mean_magss( float* __restrict__ conj_sqr_mean_mags, const cufftComplex* __restrict__ conj_sqr_means, const int num_vals ) {

   int thread_index = threadIdx.x;

   conj_sqr_mean_mags[thread_index] = cuCabsf( conj_sqr_means[thread_index] );
}


__device__
void calc_mag_sqrs( float* __restrict__ mag_sqrs, const cufftComplex* __restrict__ samples, const int num_vals ) {

   int global_index = blockDim.x * blockIdx.x + threadIdx.x;
   int stride = blockDim.x * gridDim.x;

   for (int index = global_index; index < num_vals; index += stride) {
      float temp = cuCabsf( samples[index] );
      mag_sqrs[index] = temp * temp;
   }
}


__device__
void calc_mag_sqrss( float* __restrict__ mag_sqrs, const cufftComplex* __restrict__ samples, const int num_vals ) {

   int thread_index = threadIdx.x;

   float temp = cuCabsf( samples[thread_index] );
   mag_sqrs[thread_index] = temp * temp;
}


__device__
void calc_mag_sqr_means( 
      float* __restrict__ mag_sqr_means, 
      const float* __restrict__ mag_sqrs,
      const int mag_sqr_window_size, 
      const int num_vals 
   ) { 

   int global_index = blockDim.x * blockIdx.x + threadIdx.x;
   int stride = blockDim.x * gridDim.x;

   for (int index = global_index; index < num_vals; index += stride) {
      float  t_mag_sqr_sum = 0.0;
      for( int w_index = 0; w_index < mag_sqr_window_size; ++w_index ) {
         t_mag_sqr_sum = t_mag_sqr_sum + mag_sqrs[index + w_index];
      }
      mag_sqr_means[index] = t_mag_sqr_sum/(float)mag_sqr_window_size;
   }

}


__device__
void normalize( float* __restrict__ norms, const float* __restrict__ conj_sqr_mean_mags, 
   const float* __restrict__ mag_sqr_means, const int num_samples ) {

   int global_index = blockDim.x * blockIdx.x + threadIdx.x;
   int stride = blockDim.x * gridDim.x;

   for (int index = global_index; index < num_samples; index += stride) {
      if ( mag_sqr_means[index] > 0.f ) {
         norms[index] =  conj_sqr_mean_mags[index]/mag_sqr_means[index];
      } else {
         norms[index] = 0.f;
      }
   }
}


__device__
void normalizes( float* __restrict__ norms, const float* __restrict__ conj_sqr_mean_mags, 
   const float* __restrict__ mag_sqr_means, const int num_samples ) {

   int thread_index = threadIdx.x;

   if ( mag_sqr_means[thread_index] > 0.f ) {
      norms[thread_index] =  conj_sqr_mean_mags[thread_index]/mag_sqr_means[thread_index];
   } else {
      norms[thread_index] = 0.f;
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

   int num_conj_sqr_sums = num_samples - conj_sqr_window_size;
   int num_mag_sqr_sums = num_samples - mag_sqr_window_size;

   __shared__ cufftComplex s_samples[THREADS_PER_BLOCK];
   __shared__ cufftComplex s_samples_d16[THREADS_PER_BLOCK];
   __shared__ cufftComplex s_conj_sqrs[THREADS_PER_BLOCK];
   __shared__ cufftComplex s_conj_sqr_means[THREADS_PER_BLOCK];
   __shared__ float s_conj_sqr_mean_mags[THREADS_PER_BLOCK];
   __shared__ float s_mag_sqrs[THREADS_PER_BLOCK];
   __shared__ float s_mag_sqr_means[THREADS_PER_BLOCK];
   __shared__ float s_norms[THREADS_PER_BLOCK];

   int global_index = blockDim.x * blockIdx.x + threadIdx.x;
   int thread_index = threadIdx.x;

   delay16<cufftComplex>( samples_d16, samples, num_samples );
   __syncthreads();

   
}

