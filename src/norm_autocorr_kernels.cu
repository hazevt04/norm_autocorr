
#include <stdio.h>

#include "my_cufft_utils.hpp"

#include "norm_autocorr_kernels.cuh"

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
void delay16<cufftDoubleComplex>( cufftDoubleComplex* delayed_vals, const cufftDoubleComplex* vals, const int num_vals );


__device__
void auto_correlation( cufftDoubleComplex* __restrict__ conj_sqrs, 
      const cufftDoubleComplex* __restrict__ samples_d16,
      const cufftDoubleComplex* __restrict__ samples, 
      const int num_samples 
   ) {

   int global_index = blockDim.x * blockIdx.x + threadIdx.x;
   int stride = blockDim.x * gridDim.x;

   for (int index = global_index; index < num_samples; index += stride) {
      conj_sqrs[index] = cuCmul( samples[index], cuConj( samples_d16[index] ) );
   }
}


__device__
void calc_conj_sqr_sums( 
      cufftDoubleComplex* __restrict__ conj_sqr_sums, 
      const cufftDoubleComplex* __restrict__ conj_sqrs, 
      const int conj_sqr_window_size, 
      const int num_windowed_conj_sqrs 
   ) { 

   int global_index = blockDim.x * blockIdx.x + threadIdx.x;
   int stride = blockDim.x * gridDim.x;

   for (int index = global_index; index < num_windowed_conj_sqrs; index += stride) {
      cufftDoubleComplex  t_conj_sqr_sum = make_cuDoubleComplex(0.0,0.0);

      for( int w_index = 0; w_index < conj_sqr_window_size; ++w_index ) {
         t_conj_sqr_sum = cuCadd( t_conj_sqr_sum, conj_sqrs[index + w_index] );
      }
      conj_sqr_sums[index] = t_conj_sqr_sum;
   }

}

__device__
void calc_conj_sqr_sum_mags( double* __restrict__ conj_sqr_sum_mags, 
      const cufftDoubleComplex* __restrict__ conj_sqr_sums, 
      const int num_conj_sqr_sums 
   ) {

   int global_index = blockDim.x * blockIdx.x + threadIdx.x;
   int stride = blockDim.x * gridDim.x;

   for (int index = global_index; index < num_conj_sqr_sums; index += stride) {
      conj_sqr_sum_mags[index] = cuCabs( conj_sqr_sums[index] );
   }
}


__device__
void calc_mag_sqrs( double* __restrict__ mag_sqrs, 
      const cufftDoubleComplex* __restrict__ samples, 
      const int num_samples 
   ) {

   int global_index = blockDim.x * blockIdx.x + threadIdx.x;
   int stride = blockDim.x * gridDim.x;

   for (int index = global_index; index < num_samples; index += stride) {
      double temp = cuCabs( samples[index] );
      mag_sqrs[index] = temp * temp;
   }
}


__device__
void calc_mag_sqr_sums( 
      double* __restrict__ mag_sqr_sums, 
      const double* __restrict__ mag_sqrs,
      const int mag_sqr_window_size, 
      const int num_windowed_mag_sqrs 
   ) { 

   int global_index = blockDim.x * blockIdx.x + threadIdx.x;
   int stride = blockDim.x * gridDim.x;

   for (int index = global_index; index < num_windowed_mag_sqrs; index += stride) {
      double  t_mag_sqr_sum = 0.0;
      for( int w_index = 0; w_index < mag_sqr_window_size; ++w_index ) {
         t_mag_sqr_sum = t_mag_sqr_sum + mag_sqrs[index + w_index];
      }
      mag_sqr_sums[index] = t_mag_sqr_sum;
   }

}


__device__
void normalize( double* __restrict__ norms, 
   const double* __restrict__ conj_sqr_sum_mags, 
   const double* __restrict__ mag_sqr_sums, 
   const int num_mag_sqr_sums 
) {

   int global_index = blockDim.x * blockIdx.x + threadIdx.x;
   int stride = blockDim.x * gridDim.x;

   for (int index = global_index; index < num_mag_sqr_sums; index += stride) {
      if ( mag_sqr_sums[index] > 0.f ) {
         norms[index] =  conj_sqr_sum_mags[index]/mag_sqr_sums[index];
      } else {
         norms[index] = 0.f;
      }
   }

}


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
) {

   int num_windowed_conj_sqrs  = num_samples - conj_sqr_window_size;
   int num_windowed_mag_sqrs = num_samples - mag_sqr_window_size;

   delay16<cufftDoubleComplex>( samples_d16, samples, num_samples );

   __syncthreads();

   auto_correlation( 
      conj_sqrs, 
      samples_d16, 
      samples, 
      num_samples 
   );
   __syncthreads();
   
   calc_conj_sqr_sums( 
      conj_sqr_sums, 
      conj_sqrs, 
      conj_sqr_window_size, 
      num_windowed_conj_sqrs 
   );
   __syncthreads();

   calc_conj_sqr_sum_mags( 
      conj_sqr_sum_mags, 
      conj_sqr_sums, 
      num_samples 
   );
   __syncthreads();

   calc_mag_sqrs( 
      mag_sqrs, 
      samples, 
      num_samples 
   );
   __syncthreads();

   calc_mag_sqr_sums( 
      mag_sqr_sums, 
      mag_sqrs,
      mag_sqr_window_size, 
      num_windowed_mag_sqrs 
   );
   __syncthreads();
   
   normalize( 
      norms, 
      conj_sqr_sum_mags, 
      mag_sqr_sums, 
      num_samples 
   );
}
