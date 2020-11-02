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
      
      debug_cout( debug, __func__, "(): samples.size() is ", samples.size(), "\n" ); 
      
      print_cufftComplexes( samples.data(), num_samples, "Samples: ", " ", "\n" ); 

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

      // Prefetch fspecs from the GPU
      cudaStreamAttachMemAsync( *(stream_ptr.get()), norms.data(), 0, cudaMemAttachHost );   
      
      try_cuda_func_throw( cerror, cudaStreamSynchronize( *(stream_ptr.get())  ) );
      
      // norms.size() is 0 because the add_kernel modified the data and not a std::vector function
      debug_cout( debug, __func__, "(): norms.size() is ", norms.size(), "\n" ); 

      print_results( "Norms: " );
      std::cout << "\n"; 

   } catch( std::exception& ex ) {
      std::cout << __func__ << "(): " << ex.what() << "\n"; 
   }
}


void NormAutocorrGPU::calc_norms( std::vector<float>& norms, const std::vector<float>& vals, const std::vector<float>& divisors ) {
   
   for( size_t index = 0; index != vals.size(); ++index ) {
      norms[index] = vals[index]/divisors[index];
   } 

}


void NormAutocorrGPU::calc_mags( std::vector<float>& mags, const std::vector<cufftComplex>& vals ) {
   
   for( size_t index = 0; index != vals.size(); ++index ) {
      mags[index] = cuCabsf( vals[index] );
   } 

}


void NormAutocorrGPU::calc_complex_mag_squares( std::vector<float>& mag_sqrs, const managed_vector_host<cufftComplex>& vals ) {

   for( size_t index = 0; index != vals.size(); ++index ) {
      float temp = cuCabsf( vals[index] );
      mag_sqrs[index] = temp * temp;
   } 
}


void NormAutocorrGPU::calc_auto_corrs( std::vector<cufftComplex>& auto_corrs, const managed_vector_host<cufftComplex>& lvals, const std::vector<cufftComplex>& rvals ) {
   
   for( size_t index = 0; index != lvals.size(); ++index ) {
      auto_corrs[index] = cuCmulf( lvals[index], cuConjf( rvals[index] ) );
   } 
}

void NormAutocorrGPU::calc_comp_moving_avgs( std::vector<cufftComplex>& avgs, const std::vector<cufftComplex>& vals, const int window_size ) {

   // avgs must already be all zeros
   for( size_t index = 0; index != window_size; ++index ) {
      avgs[0] = cuCaddf( avgs[0], vals[index] );
   }
      
   int num_sums = vals.size() - window_size;
   for( size_t index = 1; index != num_sums; ++index ) {
      avgs[index] = cuCsubf( cuCaddf( avgs[index-1], vals[index + window_size-1] ), vals[index-1] );
   } 

   for( size_t index = 0; index != avgs.size(); ++index ) {
      avgs[index] = complex_divide_by_scalar( avgs[index], (float)window_size );
   } 
}


void NormAutocorrGPU::calc_moving_avgs( std::vector<float>& avgs, const std::vector<float>& vals, const int window_size ) {

   // avgs must already be all zeros
   for( size_t index = 0; index != window_size; ++index ) {
      avgs[0] = avgs[0] + vals[index];
   }
      
   int num_sums = vals.size() - window_size;
   for( size_t index = 1; index != num_sums; ++index ) {
      avgs[index] = avgs[index-1] + vals[index + window_size-1] - vals[index-1];
   } 

   for( size_t index = 0; index != avgs.size(); ++index ) {
      avgs[index] = avgs[index]/(float)window_size;
   } 
}


void NormAutocorrGPU::gen_expected_norms() {
    
   exp_samples_d16.reserve( num_samples );
   exp_conj_sqrs.reserve( num_samples );
   exp_conj_sqr_means.reserve( num_samples );
   exp_conj_sqr_mean_mags.reserve( num_samples );
   exp_mag_sqrs.reserve( num_samples );
   exp_mag_sqr_means.reserve( num_samples );
   exp_norms.reserve( num_samples );

   delay_vals16( exp_samples_d16, samples, debug );
   calc_auto_corrs( exp_conj_sqrs, samples, exp_samples_d16 );
   calc_comp_moving_avgs( exp_conj_sqr_means, exp_conj_sqrs, conj_sqrs_window_size );
   calc_mags( exp_conj_sqr_mean_mags, exp_conj_sqr_means );
   
   calc_complex_mag_squares( exp_mag_sqrs, samples );
   calc_moving_avgs( exp_mag_sqr_means, exp_mag_sqrs, mag_sqrs_window_size );
   
   calc_norms( exp_norms, exp_conj_sqr_mean_mags, exp_mag_sqr_means );
}
