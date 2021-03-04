
#include "NormAutocorrGPU.cuh"

#include "norm_autocorr_kernels.cuh"


NormAutocorrGPU::NormAutocorrGPU( 
   const int new_num_samples, 
   const int new_threads_per_block,
   const int new_seed,
   const mode_select_t new_mode_select,
   const bool new_debug ):
      num_samples( new_num_samples ),
      threads_per_block( new_threads_per_block ),
      seed( new_seed ),
      mode_select( new_mode_select ),
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
      adjusted_num_sample_bytes = adjusted_num_samples * sizeof( complex_double );
      adjusted_num_norm_bytes = adjusted_num_samples * sizeof( double );
      num_norm_bytes = adjusted_num_samples * sizeof( double );
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

      exp_samples_d16.reserve( num_samples );
      exp_conj_sqrs.reserve( num_samples );
      exp_conj_sqr_sums.reserve( num_samples );
      exp_conj_sqr_sum_mags.reserve( num_samples );
      exp_mag_sqrs.reserve( num_samples );
      exp_mag_sqr_sums.reserve( num_samples );
      exp_norms.reserve( num_samples );

      for( int index = 0; index < num_samples; ++index ) {
         exp_samples_d16[index] = make_cuDoubleComplex(0.f,0.f);
         exp_conj_sqrs[index] = make_cuDoubleComplex(0.f,0.f);
         exp_conj_sqr_sums[index] = make_cuDoubleComplex(0.f,0.f);
         exp_mag_sqrs[index] = 0.f;
         exp_mag_sqr_sums[index] = 0.f;
         exp_norms[index] = 0.f;
      } 

   } catch( std::exception& ex ) {
      throw std::runtime_error{ "NormAutocorrGPU::" +  std::string{__func__} + "(): " + ex.what() }; 
   } // end of catch( std::exception& ex ) {
} // end of NormAutocorrGPU::NormAutocorrGPU( 


void NormAutocorrGPU::initialize_samples( ) {
   try {
      std::fill( samples.begin(), samples.end(), make_cuDoubleComplex(0.f,0.f) );

      if( mode_select == mode_select_t::Sinusoidal ) {
         dout << __func__ << "(): Sinusoidal Sample Mode Selected\n";
         for( size_t index = 0; index < num_samples; ++index ) {
            double t_val_real = AMPLITUDE*sin(2*PI*FREQ*index);
            double t_val_imag = AMPLITUDE*cos(2*PI*FREQ*index);
            samples[index] = make_cuDoubleComplex( t_val_real, t_val_imag );
         }
      } else if ( mode_select == mode_select_t::Increasing ) {
         dout << __func__ << "(): Increasing Sample Mode Selected.\n";
         for( size_t index = 0; index < num_samples; ++index ) {
            samples[index] = make_cuDoubleComplex( (double)(index+1), (double)(index+1) );
         }
      } else if ( mode_select == mode_select_t::Random ) {
         dout << __func__ << "(): Random Sample Test Selected. Seed is " << seed << "\n";
         gen_cufftDoubleComplexes( 
            samples.data(), 
            num_samples, 
            -AMPLITUDE,
            AMPLITUDE,
            seed 
         );

      } 

   } catch( std::exception& ex ) {
      throw std::runtime_error{ "NormAutocorrGPU::" +  std::string{__func__} + "(): " + ex.what() }; 
   } // end of try
} // end of initialize_samples( const NormAutocorrGPU::TestSelect_e test_select = Sinusoidal, 


void NormAutocorrGPU::print_results( const std::string& prefix = "Norms: " ) {
   print_vals<double>( norms.data(), num_samples, "Norms: ",  " ",  "\n" );
}


void NormAutocorrGPU::check_results( const std::string& prefix = "Original" ) {
   try {
      double max_diff = 1;
      int mismatch_index = -1;
      bool all_close = false;
      if ( debug ) {
         print_results( "Norms: " );
         std::cout << "\n"; 
      }
      dout << __func__ << "(): norms Check:\n"; 
      all_close = vals_are_close<double>( mismatch_index, norms.data(), exp_norms.data(), num_samples, max_diff );
      if (!all_close) {
         throw std::runtime_error{
            "Mismatch between actual norms from GPU and expected norms at index " +
            std::to_string(mismatch_index) 
         };
      }
      dout << "\n"; 
   } catch( std::exception& ex ) {
      throw std::runtime_error{ "NormAutocorrGPU::" +  std::string{__func__} + "(): " + ex.what() }; 
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
      std::cout << "NormAutocorrGPU::" << __func__ << "(): " << ex.what() << "\n"; 
   }

}


void NormAutocorrGPU::run_original() {
   try {
      cudaError_t cerror = cudaSuccess;
      double gpu_milliseconds = 0.f;
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

      double samples_per_second = (num_samples*1000.f)/gpu_milliseconds;
      std::cout << prefix << "All " << num_samples << " Norm Values matched expected values. Test Passed.\n"; 
      std::cout << prefix << "It took the GPU " << gpu_milliseconds 
         << " milliseconds to process " << num_samples 
         << " samples\n";

      std::cout << prefix << "That's a rate of " << samples_per_second/1e6 << " Msamples processed per second\n"; 

   } catch( std::exception& ex ) {
      std::cout << "NormAutocorrGPU::" << __func__ << "(): " << ex.what() << "\n"; 
   }

}


void NormAutocorrGPU::run() {
   try {
      dout << "NormAutocorrGPU::" << __func__ << "(): num_samples is " << num_samples << "\n"; 
      dout << "NormAutocorrGPU::" << __func__ << "(): threads_per_block is " << threads_per_block << "\n"; 
      dout << "NormAutocorrGPU::" << __func__ << "(): num_blocks is " << num_blocks << "\n\n"; 
      
      dout << "NormAutocorrGPU::" << __func__ << "(): adjusted_num_samples is " << adjusted_num_samples << "\n"; 
      dout << "NormAutocorrGPU::" << __func__ << "(): adjusted_num_sample_bytes is " << adjusted_num_sample_bytes << "\n"; 
      dout << "NormAutocorrGPU::" << __func__ << "(): adjusted_num_norm_bytes is " << adjusted_num_norm_bytes << "\n"; 

      initialize_samples();
      gen_expected_norms();
 
      run_warmup();

      run_original();
      check_results( "Original: " );

   } catch( std::exception& ex ) {
      std::cout << "NormAutocorrGPU::" << __func__ << "(): " << ex.what() << "\n"; 
   }
}


void NormAutocorrGPU::delay_vals16() {
   
   dout << "NormAutocorrGPU::" << __func__ << "() start\n";
   dout << "NormAutocorrGPU::" << __func__ << "() samples.size() is " << samples.size() << "\n";
   dout << "NormAutocorrGPU::" << __func__ << "() samples[0] is " << samples[0] << "\n";
   dout << "NormAutocorrGPU::" << __func__ << "() samples[1] is " << samples[1] << "\n";

   for( int index = 0; index < num_samples; ++index ) {
      if ( index < 16 ) {
         exp_samples_d16[index] = make_cuDoubleComplex(0.f, 0.f);
      } else {
         exp_samples_d16[index] = samples[index-16]; 
      }
   } 

   dout << "NormAutocorrGPU::" << __func__ << "() exp_samples_d16[15] is " << exp_samples_d16[15] << "\n";
   dout << "NormAutocorrGPU::" << __func__ << "() exp_samples_d16[16] is " << exp_samples_d16[16] << "\n";
   dout << "NormAutocorrGPU::" << __func__ << "() exp_samples_d16[17] is " << exp_samples_d16[17] << "\n";
   dout << "NormAutocorrGPU::" << __func__ << "() done\n";
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
      exp_conj_sqr_sum_mags[index] = cuCabs( exp_conj_sqr_sums[index] );
   } 

}


void NormAutocorrGPU::calc_complex_mag_squares() {

   for( int index = 0; index < num_samples; ++index ) {
      double temp = cuCabs( samples[index] );
      exp_mag_sqrs[index] = temp * temp;
   } 
}


void NormAutocorrGPU::calc_auto_corrs() {
   
   dout << "NormAutocorrGPU::" << __func__ << "() start\n";
   for( int index = 0; index < num_samples; ++index ) {
      exp_conj_sqrs[index] = cuCmul( samples[index], cuConj( exp_samples_d16[index] ) );
   } 
   dout << "NormAutocorrGPU::" << __func__ << "() end\n";
}


void NormAutocorrGPU::calc_exp_conj_sqr_sums() {

   // exp_conj_sqr_sums must already be all zeros
   dout << "NormAutocorrGPU::" << __func__ << "(): exp_conj_sqr_sums[0] = { " 
      << exp_conj_sqr_sums[0].x << ", " << exp_conj_sqr_sums[0].y << " }\n"; 

   for( int index = 0; index < conj_sqrs_window_size; ++index ) {
      exp_conj_sqr_sums[0] = cuCadd( exp_conj_sqr_sums[0], exp_conj_sqrs[index] );
   }
   dout << "NormAutocorrGPU::" << __func__ << "(): after initial summation, exp_conj_sqr_sums[0] = { " 
      << exp_conj_sqr_sums[0].x << ", " << exp_conj_sqr_sums[0].y << " }\n"; 
      
   int num_sums = num_samples - conj_sqrs_window_size;
   dout << "NormAutocorrGPU::" << __func__ << "(): num_sums is " << num_sums << "\n"; 
   for( int index = 1; index < num_sums; ++index ) {
      cufftDoubleComplex temp = cuCsub( exp_conj_sqr_sums[index-1], exp_conj_sqrs[index-1] );
      exp_conj_sqr_sums[index] = cuCadd( temp, exp_conj_sqrs[index + conj_sqrs_window_size-1] );
   } 

}


void NormAutocorrGPU::calc_exp_mag_sqr_sums() {

   dout << "NormAutocorrGPU::" << __func__ << "(): exp_mag_sqr_sums[0] = " << exp_mag_sqr_sums[0] << "\n"; 
   // exp_mag_sqr_sums must already be all zeros
   for( int index = 0; index < mag_sqrs_window_size; ++index ) {
      exp_mag_sqr_sums[0] = exp_mag_sqr_sums[0] + exp_mag_sqrs[index];
   }
   dout << "NormAutocorrGPU::" << __func__ << "(): After initial sum, exp_mag_sqr_sums[0] = " << exp_mag_sqr_sums[0] << "\n"; 
    
   int num_sums = num_samples - mag_sqrs_window_size;
   for( int index = 1; index < num_sums; ++index ) {
      exp_mag_sqr_sums[index] = exp_mag_sqr_sums[index-1] - exp_mag_sqrs[index-1] + exp_mag_sqrs[index + mag_sqrs_window_size-1];
   } 

}


void NormAutocorrGPU::cpu_run() {
   try { 
      double cpu_milliseconds = 0.f;
      
      dout << "NormAutocorrGPU::" << __func__ << "(): num_samples is " << num_samples << "\n";
      
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

      double samples_per_second = (num_samples*1000.f)/cpu_milliseconds;
      std::cout << "It took the CPU " << cpu_milliseconds << " milliseconds to process " << num_samples << " samples\n";
      std::cout << "That's a rate of " << samples_per_second/1e6 << " Msamples processed per second\n\n"; 

   } catch( std::exception& ex ) {
      throw std::runtime_error{ "NormAutocorrGPU::" + std::string{__func__} +  "(): " + ex.what() }; 
   }
}


void NormAutocorrGPU::gen_expected_norms() {
   try {
      cpu_run();
      
      if ( debug ) {
         print_cufftDoubleComplexes( exp_samples_d16.data(), num_samples, "Expected Samples D16: ", " ", "\n" ); 
         print_cufftDoubleComplexes( exp_conj_sqrs.data(), num_samples, "Expected Conjugate Squares: ", " ", "\n" );
         print_cufftDoubleComplexes( exp_conj_sqr_sums.data(), num_samples, "Expected Conjugate Square Means: ", " ", "\n" );
         print_vals<double>( exp_conj_sqr_sum_mags.data(), num_samples, "Expected Conjugate Square Mean Mags: ", " ", "\n" ); 
         print_vals<double>( exp_mag_sqrs.data(), num_samples, "Expected Magnitude Squares: ", " ", "\n" ); 
         print_vals<double>( exp_mag_sqr_sums.data(), num_samples, "Expected Magnitude Square Means: ", " ", "\n" );
         print_vals<double>( exp_norms.data(), num_samples, "Expected Norms: ", " ", "\n" ); 
      }

   } catch( std::exception& ex ) {
      throw std::runtime_error{ "NormAutocorrGPU::" + std::string{__func__} +  "(): " + ex.what() }; 
   }

}


NormAutocorrGPU::~NormAutocorrGPU() {
   dout << "NormAutocorrGPU::" << __func__ << "() (Destructor) called\n";
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

   exp_samples_d16.clear();
   exp_conj_sqrs.clear();
   exp_conj_sqr_sums.clear();
   exp_conj_sqr_sum_mags.clear();
   exp_mag_sqrs.clear();
   exp_mag_sqr_sums.clear();
   exp_norms.clear();

   if ( stream_ptr ) cudaStreamDestroy( *(stream_ptr.get()) );
   
   dout << "NormAutocorrGPU::" << __func__ << "() (Destructor) done\n";
}

