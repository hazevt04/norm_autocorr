#include <cuda_runtime.h>

#include "my_utils.hpp"
#include "my_cuda_utils.hpp"
#include "my_cufft_utils.hpp"

#include "norm_autocorr_kernel.cuh"


#include "device_allocator.hpp"
#include "managed_allocator_host.hpp"
#include "managed_allocator_global.hpp"

#include "NormAutocorrGPU.cuh"

void NormAutocorrGPU::run() {
   try {
      cudaError_t cerror = cudaSuccess;
      int num_shared_bytes = 0;
      int threads_per_block = 64;
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
         conj_window_size,
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


