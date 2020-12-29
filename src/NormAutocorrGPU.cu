
#include "NormAutocorrGPU.cuh"

#include "norm_autocorr_kernel.cuh"

#include "my_cufft_utils.hpp"
#include "my_cuda_utils.hpp"
#include "my_utils.hpp"

NormAutocorrGPU::NormAutocorrGPU( 
   const int new_num_samples, 
   const int new_threads_per_block,
   const int new_seed,
   const mode_select_t new_mode_select,
   const std::string new_filename,
   const bool new_debug ):
      num_samples( new_num_samples ),
      threads_per_block( new_threads_per_block ),
      seed( new_seed ),
      mode_select( new_mode_select ),
      filename( new_filename ),
      debug( new_debug ) {

   try {
      cudaError_t cerror = cudaSuccess;
      dout << __func__ << "(): num_samples is " << num_samples << "\n";

      num_blocks = (num_samples + (threads_per_block-1))/threads_per_block;

      adjusted_num_samples = threads_per_block * num_blocks;
      dout << __func__ << "(): adjusted number of samples for allocation is " 
         << adjusted_num_samples << "\n";

      adjusted_num_sample_bytes = adjusted_num_samples * sizeof( cufftComplex );
      adjusted_num_norm_bytes = adjusted_num_samples * sizeof( float );

      try_cuda_func_throw( cerror, cudaGetDevice( &device_id ) );
      
      cudaDeviceProp props;
      try_cuda_func_throw( cerror, cudaGetDeviceProperties(&props, device_id) );

      can_prefetch = props.concurrentManagedAccess;
      can_map_memory = props.canMapHostMemory;
      gpu_is_integrated = props.integrated;

      dout << __func__ << "(): can_prefetch is " << (can_prefetch ? "true" : "false") << "\n";
      dout << __func__ << "(): can_map_memory is " << (can_map_memory ? "true" : "false") << "\n";
      dout << __func__ << "(): gpu_is_integrated is " << (gpu_is_integrated ? "true" : "false") << "\n";

      stream_ptr = my_make_unique<cudaStream_t>();
      try_cudaStreamCreate( stream_ptr.get() );
      dout << __func__ << "(): after cudaStreamCreate()\n"; 
      
      samples.reserve( adjusted_num_samples );
      samples_d16.reserve( adjusted_num_samples );
      conj_sqrs.reserve( adjusted_num_samples );
      conj_sqr_means.reserve( adjusted_num_samples );
      conj_sqr_mean_mags.reserve( adjusted_num_samples );
      mag_sqrs.reserve( adjusted_num_samples );
      mag_sqr_means.reserve( adjusted_num_samples );
      norms.reserve( adjusted_num_samples );

      exp_samples_d16 = new cufftComplex[num_samples];
      exp_conj_sqrs = new cufftComplex[num_samples];
      exp_conj_sqr_means = new cufftComplex[num_samples];
      exp_conj_sqr_mean_mags = new float[num_samples];
      exp_mag_sqrs = new float[num_samples];
      exp_mag_sqr_means = new float[num_samples];
      exp_norms = new float[num_samples];

      for( int index = 0; index < num_samples; ++index ) {

         exp_samples_d16[index] = make_cuFloatComplex(0.f,0.f);
         exp_conj_sqrs[index] = make_cuFloatComplex(0.f,0.f);
         exp_conj_sqr_means[index] = make_cuFloatComplex(0.f,0.f);
         exp_mag_sqr_means[index] = 0.f;
      } 

      samples.resize(adjusted_num_samples);
      
      char* user_env = getenv( "USER" );
      if ( user_env == nullptr ) {
         throw std::runtime_error( "Empty USER env. USER environment variable needed for paths to files" ); 
      }
      
      std::string filepath_prefix = "/home/" + std::string{user_env} + "/Sandbox/CUDA/norm_autocorr/";

      dout << __func__ << "(): filename is " << filename << "\n";
      dout << __func__ << "(): norm_filename is " << norm_filename << "\n";

      filepath = filepath_prefix + filename;
      norm_filepath = filepath_prefix + norm_filename;

      dout << __func__ << "(): filepath is " << filepath << "\n";
      dout << __func__ << "(): norm_filepath is " << norm_filepath << "\n"; 
      


      try_cuda_func_throw( cerror, cudaMemset( samples_d16.data(), adjusted_num_sample_bytes, 0 ) );
      try_cuda_func_throw( cerror, cudaMemset( conj_sqrs.data(), adjusted_num_sample_bytes, 0 ) );
      try_cuda_func_throw( cerror, cudaMemset( conj_sqr_means.data(), adjusted_num_sample_bytes, 0 ) );
      try_cuda_func_throw( cerror, cudaMemset( conj_sqr_mean_mags.data(), adjusted_num_norm_bytes, 0 ) );
      try_cuda_func_throw( cerror, cudaMemset( mag_sqrs.data(), adjusted_num_norm_bytes, 0 ) );
      try_cuda_func_throw( cerror, cudaMemset( mag_sqr_means.data(), adjusted_num_norm_bytes, 0 ) );

      //std::fill( samples_d16.begin(), samples_d16.end(), make_cuFloatComplex(0.f,0.f) );
      //std::fill( conj_sqrs.begin(), conj_sqrs.end(), make_cuFloatComplex(0.f,0.f) );
      //std::fill( conj_sqr_means.begin(), conj_sqr_means.end(), make_cuFloatComplex(0.f,0.f) );
      //std::fill( conj_sqr_mean_mags.begin(), conj_sqr_mean_mags.end(), 0 );
      //std::fill( mag_sqrs.begin(), mag_sqrs.end(), 0 );
      //std::fill( mag_sqr_means.begin(), mag_sqr_means.end(), 0 );

      std::fill( norms.begin(), norms.end(), 0 );

   } catch( std::exception& ex ) {
      throw std::runtime_error{
         std::string{__func__} + std::string{"(): "} + ex.what()
      }; 
   }
}


void NormAutocorrGPU::initialize_samples( ) {
   try {
      if( mode_select == mode_select_t::Sinusoidal ) {
         dout << __func__ << "(): Sinusoidal Sample Test Selected\n";
         for( size_t index = 0; index < num_samples; ++index ) {
            float t_val_real = AMPLITUDE*sin(2*PI*FREQ*index);
            float t_val_imag = AMPLITUDE*sin(2*PI*FREQ*index);
            samples[index] = make_cuFloatComplex( t_val_real, t_val_imag );
         } 
      } else if ( mode_select == mode_select_t::Random ) {
         dout << __func__ << "(): Random Sample Test Selected\n";
         gen_cufftComplexes( samples.data(), num_samples, -50.0, 50.0 );
      } else if ( mode_select == mode_select_t::Filebased ) {
         dout << __func__ << "(): File-Based Sample Test Selected. File is " << filepath << "\n";
         read_binary_file<cufftComplex>( 
            samples,
            filepath.c_str(),
            num_samples, 
            debug );
      }           
      if (debug) {
         print_cufftComplexes( samples.data(), num_samples, "Samples: ",  " ",  "\n" ); 
      }
   } catch( std::exception& ex ) {
      throw std::runtime_error{
         std::string{__func__} + std::string{"(): "} + ex.what()
      }; 
   } // end of try
} // end of initialize_samples( const NormAutocorrGPU::TestSelect_e test_select = Sinusoidal, 


void NormAutocorrGPU::print_results( const std::string& prefix = "Norms: " ) {
   print_vals<float>( norms.data(), num_samples, prefix.data(),  " ",  "\n" );
}


void NormAutocorrGPU::check_results( const std::string& prefix = "" ) {
   try {
      float max_diff = 1;
      bool all_close = false;
      if ( debug ) {
         print_results( prefix + std::string{"Norms: "} );
         std::cout << "\n"; 
      }
      dout << __func__ << "():" << prefix << "norms Check:\n"; 
      all_close = vals_are_close( norms.data(), exp_norms, num_samples, max_diff, "norms: ", debug );
      if (!all_close) {
         throw std::runtime_error{ std::string{"Mismatch between actual norms from GPU and expected norms."} };
      }
      dout << "\n"; 
      
      std::cout << prefix << "All " << num_samples << " Norm Values matched expected values. Test Passed.\n\n"; 

   } catch( std::exception& ex ) {
      throw std::runtime_error{
         std::string{__func__} + std::string{"(): "} + ex.what()
      }; 
   }
}


void NormAutocorrGPU::run() {
   try {
      cudaError_t cerror = cudaSuccess;
      int num_shared_bytes = 0;
      int num_blocks = (adjusted_num_samples + threads_per_block - 1) / threads_per_block;

      dout << __func__ << "(): num_samples is " << num_samples << "\n"; 
      dout << __func__ << "(): threads_per_block is " << threads_per_block << "\n"; 
      dout << __func__ << "(): adjusted_num_samples is " << adjusted_num_samples << "\n"; 
      dout << __func__ << "(): num_blocks is " << num_blocks << "\n"; 
      
      initialize_samples();
      
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

      check_results();

      std::cout << "It took the GPU " << gpu_milliseconds 
         << " milliseconds to process " << num_samples 
         << " samples\n";

      float samples_per_second = (num_samples*1000.f)/gpu_milliseconds;
      std::cout << "That's a rate of " << samples_per_second/1e6 << " Msamples processed per second\n"; 


   } catch( std::exception& ex ) {
      throw std::runtime_error{
         std::string{__func__} + std::string{"(): "} + ex.what()
      }; 
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


void NormAutocorrGPU::delay_vals16() {
   
   dout << __func__ << "() start\n";
   dout << __func__ << "() samples.size() is " << samples.size() << "\n";
   dout << __func__ << "() samples[0] is " << samples[0] << "\n";
   dout << __func__ << "() samples[1] is " << samples[1] << "\n";

   for( int index = 0; index < num_samples; ++index ) {
      if ( index < 16 ) {
         exp_samples_d16[index] = make_cuFloatComplex(0.f, 0.f);
      } else {
         exp_samples_d16[index] = samples[index-16]; 
      }
   } 

   dout << __func__ << "() exp_samples_d16[15] is " << exp_samples_d16[15] << "\n";
   dout << __func__ << "() exp_samples_d16[16] is " << exp_samples_d16[16] << "\n";
   dout << __func__ << "() exp_samples_d16[17] is " << exp_samples_d16[17] << "\n";
   dout << __func__ << "() done\n";
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
      float samples_per_second = (num_samples*1000.f)/cpu_milliseconds;

      std::cout << "It took the CPU " << cpu_milliseconds << " milliseconds to process " << num_samples << " samples\n";
      std::cout << "That's a rate of " << samples_per_second/1e6 << " Msamples processed per second\n\n"; 

   } catch( std::exception& ex ) {
      throw std::runtime_error( std::string{__func__} +
         std::string{"(): "} + ex.what() ); 
   }

}

void NormAutocorrGPU::gen_expected_norms() {
   try {
      cpu_run();

      if( mode_select == mode_select_t::Filebased ) {
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

NormAutocorrGPU::~NormAutocorrGPU() {
   dout << __func__ << "() started\n";
   samples.clear();    
   samples_d16.clear();
   conj_sqrs.clear();
   conj_sqr_means.clear();
   conj_sqr_mean_mags.clear();
   mag_sqrs.clear();
   mag_sqr_means.clear();
   norms.clear();

   delete [] exp_samples_d16;
   if ( exp_conj_sqrs ) delete [] exp_conj_sqrs;
   if ( exp_conj_sqr_means ) delete [] exp_conj_sqr_means;
   if ( exp_conj_sqr_mean_mags ) delete [] exp_conj_sqr_mean_mags;
   if ( exp_mag_sqrs ) delete [] exp_mag_sqrs;
   if ( exp_mag_sqr_means ) delete [] exp_mag_sqr_means;
   if ( exp_norms ) delete [] exp_norms;

   if ( stream_ptr ) cudaStreamDestroy( *(stream_ptr.get()) );

   dout << __func__ << "() done\n";
}
