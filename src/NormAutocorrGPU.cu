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
      int num_blocks = (adjusted_num_samples + threads_per_block - 1) / threads_per_block;

      dout << __func__ << "(): num_samples is " << num_samples << "\n"; 
      dout << __func__ << "(): threads_per_block is " << threads_per_block << "\n"; 
      dout << __func__ << "(): adjusted_num_samples is " << adjusted_num_samples << "\n"; 
      dout << __func__ << "(): num_blocks is " << num_blocks << "\n"; 

      gen_expected_norms();

      if ( debug ) {
         print_cufftComplexes( exp_samples_d16, num_samples, "Expected Samples D16: ", " ", "\n" ); 
         print_cufftComplexes( exp_conj_sqrs, num_samples, "Expected Conjugate Squares: ", " ", "\n" );
         print_cufftComplexes( exp_conj_sqr_means, num_samples, "Expected Conjugate Square Means: ", " ", "\n" );
         print_vals( exp_conj_sqr_mean_mags, num_samples, "Expected Conjugate Square Mean Mags: ", " ", "\n" ); 
         print_vals( exp_mag_sqrs, num_samples, "Expected Magnitude Squares: ", " ", "\n" ); 
         print_vals( exp_mag_sqr_means, num_samples, "Expected Magnitude Square Means: ", " ", "\n" );
         print_vals( exp_norms, num_samples, "Expected Norms: ", " ", "\n" ); 
      }
     
      float gpu_milliseconds = 0.f;
      Time_Point start = Steady_Clock::now();
      
      //try_cuda_func_throw( cerror, cudaStreamAttachMemAsync( *(stream_ptr.get()), samples.data(), 0, cudaMemAttachGlobal ) );
      //try_cuda_func_throw( cerror, cudaMemPrefetchAsync( samples.data(), adjusted_num_sample_bytes, device_id, *(stream_ptr.get()) ) );

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

      //try_cuda_func_throw( cerror, cudaStreamAttachMemAsync( *(stream_ptr.get()), norms.data(), 0, cudaMemAttachHost ) );
      //try_cuda_func_throw( cerror, cudaMemPrefetchAsync( norms.data(), adjusted_num_sample_bytes, cudaCpuDeviceId, *(stream_ptr.get()) ) );

      try_cuda_func_throw( cerror, cudaDeviceSynchronize() );
      
      Duration_ms duration_ms = Steady_Clock::now() - start;
      gpu_milliseconds = duration_ms.count();

      float max_diff = 1;
      bool all_close = false;
      if ( debug ) {
         print_results( "Norms: " );
         std::cout << "\n"; 
      }
      dout << __func__ << "(): norms Check:\n"; 
      all_close = vals_are_close( norms.data(), exp_norms, num_samples, max_diff, "norms: ", debug );
      if (!all_close) {
         throw std::runtime_error{ std::string{__func__} + 
            std::string{"(): Mismatch between actual norms from GPU and expected norms."} };
      }
      dout << "\n"; 
      
      std::cout << "All " << num_samples << " Norm Values matched expected values. Test Passed.\n\n"; 
      std::cout << "It took the GPU " << gpu_milliseconds 
         << " milliseconds to process " << num_samples 
         << " samples\n";

      std::cout << "That's a rate of " << ( (num_samples*1000.f)/gpu_milliseconds ) << " samples processed per second\n"; 


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
   debug_printf( debug, "%s(): exp_conj_sqr_means[0] = { %f, %f }\n", __func__, exp_conj_sqr_means[0].x, exp_conj_sqr_means[0].y ); 
   for( int index = 0; index < conj_sqrs_window_size; ++index ) {
      exp_conj_sqr_means[0] = cuCaddf( exp_conj_sqr_means[0], exp_conj_sqrs[index] );
   }
   debug_printf( debug, "%s(): after initial sum: exp_conj_sqr_means[0] = { %f, %f }\n", __func__, exp_conj_sqr_means[0].x, exp_conj_sqr_means[0].y ); 
      
   int num_sums = num_samples - conj_sqrs_window_size;
   debug_printf( debug, "%s(): num_sums is %d\n", __func__, num_sums ); 
   for( int index = 1; index < num_sums; ++index ) {
      cufftComplex temp = cuCsubf( exp_conj_sqr_means[index-1], exp_conj_sqrs[index-1] );
      exp_conj_sqr_means[index] = cuCaddf( temp, exp_conj_sqrs[index + conj_sqrs_window_size-1] );
   } 

   /*for( int index = 0; index < num_samples; ++index ) {*/
   /*   exp_conj_sqr_means[index] = complex_divide_by_scalar( exp_conj_sqr_means[index], (float)conj_sqrs_window_size );*/
   /*} */
}


void NormAutocorrGPU::calc_exp_mag_sqr_means() {

   debug_printf( debug, "%s(): exp_mag_sqr_means[0] = %f\n", __func__, exp_mag_sqr_means[0] ); 
   // exp_mag_sqr_means must already be all zeros
   for( int index = 0; index < mag_sqrs_window_size; ++index ) {
      exp_mag_sqr_means[0] = exp_mag_sqr_means[0] + exp_mag_sqrs[index];
   }
   debug_printf( debug, "%s(): After initial sum: exp_mag_sqr_means[0] = %f\n", __func__, exp_mag_sqr_means[0] ); 
      
   int num_sums = num_samples - mag_sqrs_window_size;
   for( int index = 1; index < num_sums; ++index ) {
      exp_mag_sqr_means[index] = exp_mag_sqr_means[index-1] - exp_mag_sqrs[index-1] + exp_mag_sqrs[index + mag_sqrs_window_size-1];
   } 

   /*for( int index = 0; index  < num_samples; ++index ) {*/
   /*   exp_mag_sqr_means[index] = exp_mag_sqr_means[index]/(float)mag_sqrs_window_size;*/
   /*} */
}

void NormAutocorrGPU::cpu_run() {
   try { 
      dout << __func__ << "(): num_samples is " 
         << num_samples << "\n";

      float cpu_milliseconds = 0.f;
      Time_Point start = Steady_Clock::now();

      delay_vals16();
      calc_auto_corrs();
      calc_exp_conj_sqr_means();
      calc_mags();
      
      calc_complex_mag_squares();
      calc_exp_mag_sqr_means();
      
      calc_norms();

      Duration_ms duration_ms = Steady_Clock::now() - start;
      cpu_milliseconds = duration_ms.count();

      std::cout << "It took the CPU " << cpu_milliseconds << " milliseconds to process " << num_samples << " samples\n";
      std::cout << "That's a rate of " << ((num_samples*1000.f)/cpu_milliseconds) << " samples processed per second\n\n"; 

   } catch( std::exception& ex ) {
      throw std::runtime_error( std::string{__func__} +
         std::string{"(): "} + ex.what() ); 
   }

}

void NormAutocorrGPU::gen_expected_norms() {
   try {
      cpu_run();

      if( test_select_string =="Filebased" ) {
         float norms_from_file[num_samples];
         read_binary_file<float>( norms_from_file, 
            norm_filepath.c_str(), 
            num_samples, debug );

         float max_diff = 1.f;
         bool all_close = false;
         dout << __func__ << "(): Exp Norms Check Against File:\n"; 
         all_close = vals_are_close( exp_norms, norms_from_file, 
            num_samples, max_diff, "exp norms: ", debug );
         if (!all_close) {
            throw std::runtime_error{ std::string{__func__} + 
               std::string{"(): Mismatch between expected norms and norms from file."} };
         }
         dout << "\n";
      }
   } catch( std::exception& ex ) {
      throw std::runtime_error( std::string{__func__} + 
         std::string{"(): "} + ex.what() ); 
   }

}
