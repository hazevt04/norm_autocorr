#include <cuda_runtime.h>

#include "my_utils.hpp"
#include "my_cuda_utils.hpp"
#include "my_cufft_utils.hpp"

#include "device_allocator.hpp"
#include "managed_allocator_host.hpp"
#include "managed_allocator_global.hpp"

#include "NormAutocorrGPU.cuh"

#include "norm_autocorr_kernel.cuh"

void NormAutocorrGPU::run() {
   try {
      cudaError_t cerror = cudaSuccess;
      int num_shared_bytes = 0;
      int threads_per_block = 1024;
      int num_blocks = (num_samples + threads_per_block - 1) / threads_per_block;

      debug_cout( debug, __func__, "(): num_samples is ", num_samples, "\n" ); 
      debug_cout( debug, __func__, "(): threads_per_block is ", threads_per_block, "\n" ); 
      debug_cout( debug, __func__, "(): num_blocks is ", num_blocks, "\n" ); 

      gen_data();
      gen_expected_norms();
      
      debug_cout( debug, __func__, "(): num_samples is ", num_samples, "\n" ); 
      
      if ( debug ) {
         print_cufftComplexes( samples.data(), num_samples, "Samples: ", " ", "\n" ); 
         print_cufftComplexes( exp_samples_d16, num_samples, "Expected Samples D16: ", " ", "\n" ); 
         //print_cufftComplexes( exp_conj_sqrs, num_samples, "Expected Conjugate Squares: ", " ", "\n" );
         //print_cufftComplexes( exp_conj_sqr_means, num_samples, "Expected Conjugate Square Means: ", " ", "\n" );
         //print_vals( exp_conj_sqr_mean_mags, num_samples, "Expected Conjugate Square Mean Mags: ", " ", "\n" ); 
         //print_vals( exp_mag_sqrs, num_samples, "Expected Magnitude Squares: ", " ", "\n" ); 
         //print_vals( exp_mag_sqr_means, num_samples, "Expected Magnitude Square Means: ", " ", "\n" );
         //print_vals( exp_norms, num_samples, "Expected Norms: ", " ", "\n" ); 
      }
      cudaStreamAttachMemAsync( *(stream_ptr.get()), samples.data(), 0, cudaMemAttachGlobal );

      norm_autocorr_kernel<<<num_blocks, threads_per_block, num_shared_bytes, *(stream_ptr.get())>>>( 
         norms.data(), 
         mag_sqr_means.data(), 
         mag_sqrs.data(), 
         conj_sqr_mean_mags.data(), 
         conj_sqr_means.data(), 
         conj_sqrs.data(), 
         samples_d16.data(), 
         samples.data(),
         conj_sqrs_window_size,
         mag_sqrs_window_size,
         num_samples 
      );

      //// Prefetch fspecs from the GPU
      cudaStreamAttachMemAsync( *(stream_ptr.get()), norms.data(), 0, cudaMemAttachHost );   
      
      try_cuda_func_throw( cerror, cudaStreamSynchronize( *(stream_ptr.get())  ) );
      
      // num_samples is 0 because the add_kernel modified the data and not a std::vector function
      debug_cout( debug, __func__, "(): num_samples is ", num_samples, "\n" ); 

      //print_results( "Norms: " );
      std::cout << "\n"; 

      float max_diff = 1e-1;
      bool all_close = cufftComplexes_are_close( samples_d16.data(), 
         exp_samples_d16, num_samples, max_diff, true);
      if (!all_close) {
         throw std::runtime_error{ std::string{__func__} + 
            std::string{"(): Mismatch between actual samples_d16 from GPU and expected samples_d16."} };
      }

      //bool all_close = vals_are_close( norms.data(), exp_norms, num_samples, max_diff, debug );
      //if (!all_close) {
      //   throw std::runtime_error{ std::string{__func__} + 
      //      std::string{"(): Mismatch between actual norms from GPU and expected norms."} };
      //}

   } catch( std::exception& ex ) {
      std::cout << __func__ << "(): " << ex.what() << "\n"; 
   }
}


void NormAutocorrGPU::calc_norms() {
   
   for( int index = 0; index < num_samples; ++index ) {
      if ( exp_mag_sqr_means[index] > 0 ) {
         exp_norms[index] = exp_conj_sqr_mean_mags[index]/exp_mag_sqr_means[index];
      } else {
         exp_norms[index] = 0.f;
      }
   } 

}


void NormAutocorrGPU::calc_mags() {
   
   for( int index = 0; index < num_samples; ++index ) {
      exp_conj_sqr_mean_mags[index] = cuCabsf( exp_conj_sqr_means[index] );
   } 

}


void NormAutocorrGPU::calc_complex_mag_squares() {

   for( int index = 0; index < num_samples; ++index ) {
      float temp = cuCabsf( samples[index] );
      exp_mag_sqrs[index] = temp * temp;
   } 
}


void NormAutocorrGPU::calc_auto_corrs() {
   
   dout << __func__ << "() start\n";
   for( int index = 0; index < num_samples; ++index ) {
      exp_conj_sqrs[index] = cuCmulf( samples[index], cuConjf( exp_samples_d16[index] ) );
   } 
   dout << __func__ << "() end\n";
}

void NormAutocorrGPU::calc_exp_conj_sqr_means() {

   // exp_conj_sqr_means must already be all zeros
   for( int index = 0; index < conj_sqrs_window_size; ++index ) {
      exp_conj_sqr_means[0] = cuCaddf( exp_conj_sqr_means[0], exp_conj_sqrs[index] );
   }
      
   int num_sums = num_samples - conj_sqrs_window_size;
   for( int index = 1; index < num_sums; ++index ) {
      exp_conj_sqr_means[index] = cuCsubf( cuCaddf( exp_conj_sqr_means[index-1], exp_conj_sqrs[index + conj_sqrs_window_size-1] ), 
         exp_conj_sqrs[index-1] );
   } 

   for( int index = 0; index < num_samples; ++index ) {
      exp_conj_sqr_means[index] = complex_divide_by_scalar( exp_conj_sqr_means[index], (float)conj_sqrs_window_size );
   } 
}


void NormAutocorrGPU::calc_exp_mag_sqr_means() {

   // exp_mag_sqr_means must already be all zeros
   for( int index = 0; index < mag_sqrs_window_size; ++index ) {
      exp_mag_sqr_means[0] = exp_mag_sqr_means[0] + exp_mag_sqrs[index];
   }
      
   int num_sums = num_samples - mag_sqrs_window_size;
   for( int index = 1; index < num_sums; ++index ) {
      exp_mag_sqr_means[index] = exp_mag_sqr_means[index-1] + exp_mag_sqrs[index + mag_sqrs_window_size-1] - exp_mag_sqrs[index-1];
   } 

   for( int index = 0; index  < num_samples; ++index ) {
      exp_mag_sqr_means[index] = exp_mag_sqr_means[index]/(float)mag_sqrs_window_size;
   } 
}


void NormAutocorrGPU::gen_expected_norms() {
    
   dout << "num_samples is " << num_samples << "\n";

   delay_vals16();
   calc_auto_corrs();
   calc_exp_conj_sqr_means();
   calc_mags();
   
   calc_complex_mag_squares();
   calc_exp_mag_sqr_means();
   
   calc_norms();
}
