
#include "NormAutocorrGPU.cuh"

#include "norm_autocorr_kernels.cuh"

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

      dout << __func__ << "(): Mode Select is " << get_mode_select_string( mode_select ) << "\n";

      if ( mode_select == mode_select_t::Increasing ) {
         if ( num_samples > MAX_NUM_SAMPLES_INCREASING ) {
            std::cout << "WARNING: num_samples, " << num_samples << " too large. The sum will not fit in a 32-bit integer.\n";
            std::cout << "Changing num_samples to the max: " << MAX_NUM_SAMPLES_INCREASING << "\n";
            num_samples = MAX_NUM_SAMPLES_INCREASING;
         }
      }

      try_cuda_func_throw( cerror, cudaGetDevice( &device_id ) );

      stream_ptr = my_make_unique<cudaStream_t>();
      try_cudaStreamCreate( stream_ptr.get() );
      dout << __func__ << "(): after cudaStreamCreate()\n"; 

      dout << __func__ << "(): num_samples is " << num_samples << "\n";

      num_blocks = (num_samples + (threads_per_block-1))/threads_per_block;
      dout << __func__ << "(): num_blocks is " << num_blocks << "\n";

      adjusted_num_samples = threads_per_block * num_blocks;
      adjusted_num_sample_bytes = adjusted_num_samples * sizeof( cufftComplex );
      adjusted_num_norm_bytes = adjusted_num_samples * sizeof( float );
      num_norm_bytes = adjusted_num_samples * sizeof( float );
      num_shared_bytes = 0;

      dout << __func__ << "(): adjusted number of samples for allocation is " 
         << adjusted_num_samples << "\n";
      dout << __func__ << "(): adjusted number of sample bytes for cudaMemcpyAsync is "
         << adjusted_num_sample_bytes << "\n";
      dout << __func__ << "(): adjusted number of norm bytes for cudaMemcpyAsync is "
         << adjusted_num_norm_bytes << "\n\n";

      samples.reserve( adjusted_num_samples );
      
      d_samples.reserve( adjusted_num_samples );
      samples_d16.reserve( adjusted_num_samples );
      conj_sqrs.reserve( adjusted_num_samples );
      conj_sqr_sums.reserve( adjusted_num_samples );
      conj_sqr_sum_mags.reserve( adjusted_num_samples );
      mag_sqrs.reserve( adjusted_num_samples );
      mag_sqr_sums.reserve( adjusted_num_samples );
      norms.reserve( adjusted_num_samples );
      d_norms.reserve( adjusted_num_samples );
      
      norms.resize(adjusted_num_samples);
      std::fill( norms.begin(), norms.end(), 0 );

      //try_cuda_func_throw( cerror, cudaMemset( d_samples.data(), adjusted_num_sample_bytes, 0 ) );
      //try_cuda_func_throw( cerror, cudaMemset( samples_d16.data(), adjusted_num_sample_bytes, 0 ) );
      //try_cuda_func_throw( cerror, cudaMemset( conj_sqrs.data(), adjusted_num_sample_bytes, 0 ) );
      //try_cuda_func_throw( cerror, cudaMemset( conj_sqr_sums.data(), adjusted_num_sample_bytes, 0 ) );
      //try_cuda_func_throw( cerror, cudaMemset( conj_sqr_sum_mags.data(), adjusted_num_sample_bytes, 0 ) );
      //try_cuda_func_throw( cerror, cudaMemset( mag_sqrs.data(), adjusted_num_norm_bytes, 0 ) );
      //try_cuda_func_throw( cerror, cudaMemset( mag_sqr_sums.data(), adjusted_num_norm_bytes, 0 ) );
      //try_cuda_func_throw( cerror, cudaMemset( d_norms.data(), adjusted_num_norm_bytes, 0 ) );

      exp_samples_d16 = new cufftComplex[num_samples];
      exp_conj_sqrs = new cufftComplex[num_samples];
      exp_conj_sqr_sums = new cufftComplex[num_samples];
      exp_conj_sqr_sum_mags = new float[num_samples];
      exp_mag_sqrs = new float[num_samples];
      exp_mag_sqr_sums = new float[num_samples];
      exp_norms = new float[num_samples];

      for( int index = 0; index < num_samples; ++index ) {

         exp_samples_d16[index] = make_cuFloatComplex(0.f,0.f);
         exp_conj_sqrs[index] =  make_cuFloatComplex(0.f,0.f);
         exp_conj_sqr_sums[index] = make_cuFloatComplex(0.f,0.f);
         exp_mag_sqrs[index] = 0.f;
         exp_mag_sqr_sums[index] = 0.f;
         exp_norms[index] = 0.f;
      } 

      char* user_env = getenv( "USER" );
      if ( user_env == nullptr ) {
         throw std::runtime_error( std::string{__func__} + 
            "(): Empty USER env. USER environment variable needed for paths to files" ); 
      }
      
      std::string filepath_prefix = "/home/" + std::string{user_env} + "/Sandbox/CUDA/norm_autocorr/";

      dout << "Filename is " << filename << "\n";
      dout << "Norm Filename is " << norm_filename << "\n";

      filepath = filepath_prefix + filename;
      norm_filepath = filepath_prefix + norm_filename;

      dout << "Filepath is " << filepath << "\n";
      dout << "Norm Filepath is " << norm_filepath << "\n";
      
   } catch( std::exception& ex ) {
      throw std::runtime_error{
         std::string{__func__} + std::string{"(): "} + ex.what()
      }; 
   } // end of catch( std::exception& ex ) {
} // end of NormAutocorrGPU::NormAutocorrGPU( 


void NormAutocorrGPU::initialize_samples( ) {
   try {
      std::fill( samples.begin(), samples.end(), make_cuFloatComplex(0.f,0.f) );

      if( mode_select == mode_select_t::Sinusoidal ) {
         dout << __func__ << "(): Sinusoidal Sample Mode Selected\n";
         for( size_t index = 0; index < num_samples; ++index ) {
            float t_val_real = AMPLITUDE*sin(2*PI*FREQ*index);
            float t_val_imag = AMPLITUDE*cos(2*PI*FREQ*index);
            samples[index] = make_cuFloatComplex( t_val_real, t_val_imag );
         }
      } else if ( mode_select == mode_select_t::Increasing ) {
         dout << __func__ << "(): Increasing Sample Mode Selected.\n";
         for( size_t index = 0; index < num_samples; ++index ) {
            samples[index] = make_cuFloatComplex( (float)(index+1), (float)(index+1) );
         }
      } else if ( mode_select == mode_select_t::Random ) {
         dout << __func__ << "(): Random Sample Test Selected. Seed is " << seed << "\n";
         gen_cufftComplexes( samples.data(), num_samples, -AMPLITUDE, AMPLITUDE, seed );

      } else if ( mode_select == mode_select_t::Filebased ) {
         dout << __func__ << "(): File-Based Sample Test Selected. File is " << filename << "\n";
         read_binary_file<cufftComplex>( 
            samples,
            filepath.c_str(),
            num_samples, 
            debug 
         );
      } // end of else-ifs for mode_select

   } catch( std::exception& ex ) {
      throw std::runtime_error{
         std::string{__func__} + std::string{"(): "} + ex.what()
      }; 
   } // end of try
} // end of initialize_samples( const NormAutocorrGPU::TestSelect_e test_select = Sinusoidal, 


void NormAutocorrGPU::print_results( const std::string& prefix = "Norms: " ) {
   print_vals<float>( norms.data(), num_samples, "Norms: ",  " ",  "\n" );
}


void NormAutocorrGPU::check_results( const std::string& prefix = "Original" ) {
   try {
      float max_diff = 1;
      bool all_close = false;
      if ( debug ) {
         print_results( "Norms: " );
         std::cout << "\n"; 
      }
      dout << __func__ << "(): norms Check:\n"; 
      all_close = vals_are_close( norms.data(), exp_norms, num_samples, max_diff, "norms: ", debug );
      if (!all_close) {
         throw std::runtime_error{  
            std::string{"Mismatch between actual norms from GPU and expected norms."} };
      }
      dout << "\n"; 
   } catch( std::exception& ex ) {
      throw std::runtime_error( std::string{__func__} + std::string{"(): "} + ex.what() ); 
   }
}


void NormAutocorrGPU::run_warmup() {
   try {
      cudaError_t cerror = cudaSuccess;
      try_cuda_func( cerror, cudaMemcpyAsync( d_samples.data(), samples.data(), adjusted_num_sample_bytes,
               cudaMemcpyHostToDevice, *(stream_ptr.get()) ) );

      norm_autocorr_kernels<<<num_blocks, threads_per_block, num_shared_bytes, *(stream_ptr.get())>>>( 
         d_norms.data(), 
         mag_sqr_sums.data(), 
         mag_sqrs.data(), 
         conj_sqr_sum_mags.data(), 
         conj_sqr_sums.data(), 
         conj_sqrs.data(), 
         samples_d16.data(), 
         d_samples.data(),
         conj_sqrs_window_size,
         mag_sqrs_window_size,
         num_samples 
      );

      try_cuda_func( cerror, cudaMemcpyAsync( norms.data(), d_norms.data(), adjusted_num_norm_bytes,
               cudaMemcpyDeviceToHost, *(stream_ptr.get()) ) );
      
      try_cuda_func_throw( cerror, cudaDeviceSynchronize() );
      
   } catch( std::exception& ex ) {
      std::cout << __func__ << "(): " << ex.what() << "\n"; 
   }

}


void NormAutocorrGPU::run_original() {
   try {
      cudaError_t cerror = cudaSuccess;
      float gpu_milliseconds = 0.f;
      std::string prefix = "Original: ";

      std::fill( norms.begin(), norms.end(), 0.f );
      Time_Point start = Steady_Clock::now();
      
      try_cuda_func( cerror, cudaMemcpyAsync( d_samples.data(), samples.data(), adjusted_num_sample_bytes,
               cudaMemcpyHostToDevice, *(stream_ptr.get()) ) );

      norm_autocorr_kernels<<<num_blocks, threads_per_block, num_shared_bytes, *(stream_ptr.get())>>>( 
         d_norms.data(), 
         mag_sqr_sums.data(), 
         mag_sqrs.data(), 
         conj_sqr_sum_mags.data(), 
         conj_sqr_sums.data(), 
         conj_sqrs.data(), 
         samples_d16.data(), 
         d_samples.data(),
         conj_sqrs_window_size,
         mag_sqrs_window_size,
         num_samples 
      );

      try_cuda_func( cerror, cudaMemcpyAsync( norms.data(), d_norms.data(), adjusted_num_norm_bytes,
               cudaMemcpyDeviceToHost, *(stream_ptr.get()) ) );
      
      try_cuda_func_throw( cerror, cudaDeviceSynchronize() );
      
      Duration_ms duration_ms = Steady_Clock::now() - start;
      gpu_milliseconds = duration_ms.count();

      float samples_per_second = (num_samples*1000.f)/gpu_milliseconds;
      std::cout << prefix << "All " << num_samples << " Norm Values matched expected values. Test Passed.\n"; 
      std::cout << prefix << "It took the GPU " << gpu_milliseconds 
         << " milliseconds to process " << num_samples 
         << " samples\n";

      std::cout << prefix << "That's a rate of " << samples_per_second/1e6 << " Msamples processed per second\n"; 

   } catch( std::exception& ex ) {
      std::cout << __func__ << "(): " << ex.what() << "\n"; 
   }

}


void NormAutocorrGPU::run() {
   try {
      dout << __func__ << "(): num_samples is " << num_samples << "\n"; 
      dout << __func__ << "(): threads_per_block is " << threads_per_block << "\n"; 
      dout << __func__ << "(): num_blocks is " << num_blocks << "\n\n"; 
      
      dout << __func__ << "(): adjusted_num_samples is " << adjusted_num_samples << "\n"; 
      dout << __func__ << "(): adjusted_num_sample_bytes is " << adjusted_num_sample_bytes << "\n"; 
      dout << __func__ << "(): adjusted_num_norm_bytes is " << adjusted_num_norm_bytes << "\n"; 

      initialize_samples();
      gen_expected_norms();
 
      run_warmup();

      run_original();
      check_results( "Original: " );

   } catch( std::exception& ex ) {
      std::cout << __func__ << "(): " << ex.what() << "\n"; 
   }
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


void NormAutocorrGPU::calc_norms() {
   
   for( int index = 0; index < num_samples; ++index ) {
      if ( exp_mag_sqr_sums[index] > 0 ) {
         exp_norms[index] = exp_conj_sqr_sum_mags[index]/exp_mag_sqr_sums[index];
      } else {
         exp_norms[index] = 0.f;
      }
   } 

}


void NormAutocorrGPU::calc_mags() {
   
   for( int index = 0; index < num_samples; ++index ) {
      exp_conj_sqr_sum_mags[index] = cuCabsf( exp_conj_sqr_sums[index] );
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


void NormAutocorrGPU::calc_exp_conj_sqr_sums() {

   // exp_conj_sqr_sums must already be all zeros
   dout << __func__ << "(): exp_conj_sqr_sums[0] = { " 
      << exp_conj_sqr_sums[0].x << ", " << exp_conj_sqr_sums[0].y << " }\n"; 

   for( int index = 0; index < conj_sqrs_window_size; ++index ) {
      exp_conj_sqr_sums[0] = cuCaddf( exp_conj_sqr_sums[0], exp_conj_sqrs[index] );
   }
   dout << __func__ << "(): after initial summation, exp_conj_sqr_sums[0] = { " 
      << exp_conj_sqr_sums[0].x << ", " << exp_conj_sqr_sums[0].y << " }\n"; 
      
   int num_sums = num_samples - conj_sqrs_window_size;
   dout << __func__ << "(): num_sums is " << num_sums << "\n"; 
   for( int index = 1; index < num_sums; ++index ) {
      cufftComplex temp = cuCsubf( exp_conj_sqr_sums[index-1], exp_conj_sqrs[index-1] );
      exp_conj_sqr_sums[index] = cuCaddf( temp, exp_conj_sqrs[index + conj_sqrs_window_size-1] );
   } 

}


void NormAutocorrGPU::calc_exp_mag_sqr_sums() {

   dout << __func__ << "(): exp_mag_sqr_sums[0] = " << exp_mag_sqr_sums[0] << "\n"; 
   // exp_mag_sqr_sums must already be all zeros
   for( int index = 0; index < mag_sqrs_window_size; ++index ) {
      exp_mag_sqr_sums[0] = exp_mag_sqr_sums[0] + exp_mag_sqrs[index];
   }
   dout << __func__ << "(): After initial sum, exp_mag_sqr_sums[0] = " << exp_mag_sqr_sums[0] << "\n"; 
    
   int num_sums = num_samples - mag_sqrs_window_size;
   for( int index = 1; index < num_sums; ++index ) {
      exp_mag_sqr_sums[index] = exp_mag_sqr_sums[index-1] - exp_mag_sqrs[index-1] + exp_mag_sqrs[index + mag_sqrs_window_size-1];
   } 

}


void NormAutocorrGPU::cpu_run() {
   try { 
      float cpu_milliseconds = 0.f;
      
      dout << __func__ << "(): num_samples is " << num_samples << "\n";
      
      Time_Point start = Steady_Clock::now();

      delay_vals16();
      calc_auto_corrs();
      calc_exp_conj_sqr_sums();
      calc_mags();
      
      calc_complex_mag_squares();
      calc_exp_mag_sqr_sums();
      
      calc_norms();

      Duration_ms duration_ms = Steady_Clock::now() - start;
      cpu_milliseconds = duration_ms.count();

      float samples_per_second = (num_samples*1000.f)/cpu_milliseconds;
      std::cout << "It took the CPU " << cpu_milliseconds << " milliseconds to process " << num_samples << " samples\n";
      std::cout << "That's a rate of " << samples_per_second/1e6 << " Msamples processed per second\n\n"; 

   } catch( std::exception& ex ) {
      throw std::runtime_error( std::string{__func__} +  std::string{"(): "} + ex.what() ); 
   }
}


void NormAutocorrGPU::gen_expected_norms() {
   try {
      cpu_run();
      
      if ( mode_select == mode_select_t::Filebased ) {
         float norms_from_file[num_samples];
         read_binary_file<float>( norms_from_file, norm_filepath.c_str(), num_samples, debug );

         float max_diff = 1.f;
         bool all_close = false;
         dout << __func__ << "(): Exp Norms Check Against File:\n"; 
         all_close = vals_are_close( exp_norms, norms_from_file, num_samples, max_diff, "exp norms: ", debug );
         if (!all_close) {
            throw std::runtime_error{ std::string{__func__} + 
               std::string{"(): Mismatch between expected norms and norms from file."} };
         }
         dout << "\n";
      }

      if ( debug ) {
         print_cufftComplexes( exp_samples_d16, num_samples, "Expected Samples D16: ", " ", "\n" ); 
         print_cufftComplexes( exp_conj_sqrs, num_samples, "Expected Conjugate Squares: ", " ", "\n" );
         print_cufftComplexes( exp_conj_sqr_sums, num_samples, "Expected Conjugate Square Means: ", " ", "\n" );
         print_vals( exp_conj_sqr_sum_mags, num_samples, "Expected Conjugate Square Mean Mags: ", " ", "\n" ); 
         print_vals( exp_mag_sqrs, num_samples, "Expected Magnitude Squares: ", " ", "\n" ); 
         print_vals( exp_mag_sqr_sums, num_samples, "Expected Magnitude Square Means: ", " ", "\n" );
         print_vals( exp_norms, num_samples, "Expected Norms: ", " ", "\n" ); 
      }

   } catch( std::exception& ex ) {
      throw std::runtime_error( std::string{__func__} +  std::string{"(): "} + ex.what() ); 
   }

}


NormAutocorrGPU::~NormAutocorrGPU() {
   dout << "dtor called\n";
   d_samples.clear();    
   samples.clear();    
   samples_d16.clear();
   conj_sqrs.clear();
   conj_sqr_sums.clear();
   conj_sqr_sum_mags.clear();
   mag_sqrs.clear();
   mag_sqr_sums.clear();
   norms.clear();
   d_norms.clear();

   delete [] exp_samples_d16;
   if ( exp_conj_sqrs ) delete [] exp_conj_sqrs;
   if ( exp_conj_sqr_sums ) delete [] exp_conj_sqr_sums;
   if ( exp_conj_sqr_sum_mags ) delete [] exp_conj_sqr_sum_mags;
   if ( exp_mag_sqrs ) delete [] exp_mag_sqrs;
   if ( exp_mag_sqr_sums ) delete [] exp_mag_sqr_sums;
   if ( exp_norms ) delete [] exp_norms;

   if ( stream_ptr ) cudaStreamDestroy( *(stream_ptr.get()) );
   
   dout << "dtor done\n";
}

