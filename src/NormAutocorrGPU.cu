#include <cuda_runtime.h>

#include "my_utils.hpp"
#include "my_cuda_utils.hpp"
#include "my_cufft_utils.hpp"

#include "NormAutocorrGPU.hpp"

#include "norm_autocorr_kernel.cuh"

void NormAutocorrGPU::initialize_samples( const int seed = 0, const bool debug = false ) {
   try {
      samples.resize(adjusted_num_samples);
      std::fill( samples.begin(), samples.end(), make_cuFloatComplex(0.f,0.f) );

      if( test_select_string =="Sinusoidal" ) {
         dout << __func__ << "(): Sinusoidal Sample Test Selected\n";
         for( size_t index = 0; index < num_samples; ++index ) {
            float t_val_real = AMPLITUDE*sin(2*PI*FREQ*index);
            float t_val_imag = AMPLITUDE*sin(2*PI*FREQ*index);
            samples[index] = make_cuFloatComplex( t_val_real, t_val_imag );
         } 
      } else if ( test_select_string == "Random" ) {
         dout << __func__ << "(): Random Sample Test Selected\n";
         gen_cufftComplexes( samples.data(), num_samples, -50.0, 50.0 );
      } else if ( test_select_string == "Filebased" ) {
         dout << __func__ << "(): File-Based Sample Test Selected. File is " << filename << "\n";
         read_binary_file<cufftComplex>( 
            samples,
            filepath.c_str(),
            num_samples, 
            debug );
      } else {
         throw std::runtime_error( std::string{__func__} + 
            std::string{"(): Error: Invalid test select: "} + 
               test_select_string );
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


NormAutocorrGPU::NormAutocorrGPU( 
   const my_args_t& args
):
      num_samples( args.num_samples ),
      conj_sqrs_window_size( args.conj_sqrs_window_size ),
      mag_sqrs_window_size( args.mag_sqrs_window_size ),
      max_num_iters( args.max_num_iters ),
      test_select_string( args.test_select_string ),
      filename( args.filename ),
      exp_norms_filename( args.exp_norms_filename ),
      debug( args.debug ) {

   try {
      cudaError_t cerror = cudaSuccess;         
      try_cuda_func_throw( cerror, cudaGetDevice( &device_id ) );

      stream_ptr = my_make_unique<cudaStream_t>();
      try_cudaStreamCreate( stream_ptr.get() );
      dout << __func__ << "(): after cudaStreamCreate()\n"; 

      dout << __func__ << "(): num_samples is " << num_samples << "\n";

      num_blocks = (num_samples + (threads_per_block-1))/threads_per_block;
      dout << __func__ << "(): threads_per_block is " << threads_per_block << "\n";
      dout << __func__ << "(): num_blocks is " << num_blocks << "\n";

      adjusted_num_samples = threads_per_block * num_blocks;
      adjusted_num_sample_bytes = adjusted_num_samples * sizeof( cufftComplex );
      adjusted_num_norm_bytes = adjusted_num_samples * sizeof( float );
      num_norm_bytes = adjusted_num_samples * sizeof( float );

      dout << __func__ << "(): adjusted number of samples for allocation is " 
         << adjusted_num_samples << "\n";
      dout << __func__ << "(): adjusted number of sample bytes for cudaMemcpyAsync is "
         << adjusted_num_sample_bytes << "\n";
      dout << __func__ << "(): adjusted number of norm bytes for cudaMemcpyAsync is "
         << adjusted_num_norm_bytes << "\n\n";

      samples.reserve( adjusted_num_samples );
      
      samples_d16.reserve( adjusted_num_samples );
      conj_sqrs.reserve( adjusted_num_samples );
      conj_sqr_sums.reserve( adjusted_num_samples );
      conj_sqr_sum_mags.reserve( adjusted_num_samples );
      mag_sqrs.reserve( adjusted_num_samples );
      mag_sqr_sums.reserve( adjusted_num_samples );
      norms.reserve( adjusted_num_samples );

      samples.resize(adjusted_num_samples); 
      samples_d16.resize( adjusted_num_samples );
      conj_sqrs.resize( adjusted_num_samples );
      conj_sqr_sums.resize( adjusted_num_samples );
      conj_sqr_sum_mags.resize( adjusted_num_samples );
      mag_sqrs.resize( adjusted_num_samples );
      mag_sqr_sums.resize( adjusted_num_samples );
      norms.resize(adjusted_num_samples);
      
      std::fill( samples_d16.begin(), samples_d16.end(), make_cuFloatComplex(0.f,0.f) );
      std::fill( conj_sqrs.begin(), conj_sqrs.end(), make_cuFloatComplex(0.f,0.f) );
      std::fill( conj_sqr_sums.begin(), conj_sqr_sums.end(), make_cuFloatComplex(0.f,0.f) );
      std::fill( conj_sqr_sum_mags.begin(), conj_sqr_sum_mags.end(), 0.f );
      std::fill( mag_sqrs.begin(), mag_sqrs.end(), 0.f );
      std::fill( mag_sqr_sums.begin(), mag_sqr_sums.end(), 0.f );
      std::fill( norms.begin(), norms.end(), 0.f );
      
      exp_samples_d16.resize(num_samples);
      exp_conj_sqrs.resize(num_samples);
      exp_conj_sqr_sums.resize(num_samples);
      exp_conj_sqr_sum_mags.resize(num_samples);
      exp_mag_sqrs.resize(num_samples);
      exp_mag_sqr_sums.resize(num_samples);
      exp_norms.resize(num_samples);

      exp_samples_d16.reserve(num_samples);
      exp_conj_sqrs.reserve(num_samples);
      exp_conj_sqr_sums.reserve(num_samples);
      exp_conj_sqr_sum_mags.reserve(num_samples);
      exp_mag_sqrs.reserve(num_samples);
      exp_mag_sqr_sums.reserve(num_samples);
      exp_norms.reserve(num_samples);
      
      for( int index = 0; index < num_samples; ++index ) {
         exp_samples_d16[index] = make_cuFloatComplex(0.f,0.f);
         exp_conj_sqrs[index] =  make_cuFloatComplex(0.f,0.f);
         exp_conj_sqr_sums[index] = make_cuFloatComplex(0.f,0.f);
         exp_conj_sqr_sum_mags[index] = 0.f;
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

      filepath = filepath_prefix + filename;
      exp_norms_filepath = filepath_prefix + exp_norms_filename;

      exp_samples_d16_filepath = filepath_prefix + exp_samples_d16_filename;
      exp_conj_sqrs_filepath = filepath_prefix + exp_conj_sqrs_filename;
      exp_conj_sqr_sums_filepath = filepath_prefix + exp_conj_sqr_sums_filename;
      exp_conj_sqr_sum_mags_filepath = filepath_prefix + exp_conj_sqr_sum_mags_filename;
      exp_mag_sqrs_filepath = filepath_prefix + exp_mag_sqrs_filename;
      exp_mag_sqr_sums_filepath = filepath_prefix + exp_mag_sqr_sums_filename;

      dout << "Filepath is " << filepath << "\n";
      dout << "Expected Norms Filepath is " << exp_norms_filepath << "\n";
      dout << "exp_samples_d16 Filepath is '" << exp_samples_d16_filepath << "'\n";
      dout << "exp_conj_sqrs Filepath is '" << exp_conj_sqrs_filepath << "'\n";
      dout << "exp_conj_sqr_sums Filepath is '" << exp_conj_sqr_sums_filepath << "'\n";
      dout << "exp_conj_sqr_sum_mags Filepath is '" << exp_conj_sqr_sum_mags_filepath << "'\n";
      dout << "exp_mag_sqrs Filepath is '" << exp_mag_sqrs_filepath << "'\n";
      dout << "exp_mag_sqr_sums Filepath is '" << exp_mag_sqr_sums_filepath << "'\n";

      initialize_samples();

   } catch( std::exception& ex ) {
      throw std::runtime_error{
         std::string{__func__} + std::string{"(): "} + ex.what()
      }; 
   }
} // end of constructor


void NormAutocorrGPU::run() {
   try {
      cudaError_t cerror = cudaSuccess;

      dout << __func__ << "(): num_samples is " << num_samples << "\n"; 
      dout << __func__ << "(): threads_per_block is " << threads_per_block << "\n"; 
      dout << __func__ << "(): num_blocks is " << num_blocks << "\n\n"; 
      
      dout << __func__ << "(): adjusted_num_samples is " << adjusted_num_samples << "\n"; 
      dout << __func__ << "(): adjusted_num_sample_bytes is " << adjusted_num_sample_bytes << "\n"; 
      dout << __func__ << "(): adjusted_num_norm_bytes is " << adjusted_num_norm_bytes << "\n"; 

      gen_expected_norms();

      int num_to_print = 80;
      if ( debug ) {
         print_cufftComplexes( samples.data(), num_to_print, "Samples: ", " ", "\n" ); 
         print_cufftComplexes( exp_samples_d16.data(), num_to_print, "Expected Samples D16: ", " ", "\n" ); 
         print_cufftComplexes( exp_conj_sqrs.data(), num_to_print, "Expected Conjugate Squares: ", " ", "\n" );
         print_cufftComplexes( exp_conj_sqr_sums.data(), num_to_print, "Expected Conjugate Square Sums: ", " ", "\n" );
         print_vals( exp_conj_sqr_sum_mags.data(), num_to_print, "Expected Conjugate Square Sum Mags: ", " ", "\n" ); 
         print_vals( exp_mag_sqrs.data(), num_to_print, "Expected Magnitude Squares: ", " ", "\n" ); 
         print_vals( exp_mag_sqr_sums.data(), num_to_print, "Expected Magnitude Square Sums: ", " ", "\n" );
         print_vals( exp_norms.data(), num_to_print, "Expected Norms: ", " ", "\n" ); 
      }
      
      float gpu_milliseconds = 0.f;
      Time_Point start = Steady_Clock::now();
      
      dout << __func__ << "(): Running the CUDA kernels...\n";
      delay16<<<num_blocks, threads_per_block, 0, *(stream_ptr.get())>>>( 
         samples_d16.data(), samples.data(), adjusted_num_samples );
 
      auto_correlation<<<num_blocks, threads_per_block, 0, *(stream_ptr.get())>>>( 
         conj_sqrs.data(), samples_d16.data(), samples.data(), adjusted_num_samples );

      calc_conj_sqr_sums<<<num_blocks, threads_per_block, 0, *(stream_ptr.get())>>>( 
         conj_sqr_sums.data(), 
         conj_sqrs.data(), 
         conj_sqrs_window_size, 
         adjusted_num_samples 
      ); 

      calc_conj_sqr_sum_mags<<<num_blocks, threads_per_block, 0, *(stream_ptr.get())>>>( 
         conj_sqr_sum_mags.data(), 
         conj_sqr_sums.data(), 
         adjusted_num_samples
      );
   
      calc_mag_sqrs<<<num_blocks, threads_per_block, 0, *(stream_ptr.get())>>>( 
         mag_sqrs.data(), 
         samples.data(), 
         adjusted_num_samples 
      );
      
      calc_mag_sqr_sums<<<num_blocks, threads_per_block, 0, *(stream_ptr.get())>>>( 
         mag_sqr_sums.data(), 
         mag_sqrs.data(),
         mag_sqrs_window_size, 
         adjusted_num_samples 
      );      

      normalize<<<num_blocks, threads_per_block, 0, *(stream_ptr.get())>>>( 
         norms.data(), 
         conj_sqr_sum_mags.data(), 
         mag_sqr_sums.data(), 
         adjusted_num_samples 
      );

      try_cuda_func_throw( cerror, cudaDeviceSynchronize() );
      if ( debug ) {
         print_cufftComplexes( samples_d16.data(), num_to_print, "Actual Samples D16: ", " ", "\n" ); 
         print_cufftComplexes( conj_sqrs.data(), num_to_print, "Actual Conjugate Squares: ", " ", "\n" );
         print_cufftComplexes( conj_sqr_sums.data(), num_to_print, "Actual Conjugate Square Sums: ", " ", "\n" );
         print_vals( conj_sqr_sum_mags.data(), num_to_print, "Actual Conjugate Square Sum Mags: ", " ", "\n" ); 
         print_vals( mag_sqrs.data(), num_to_print, "Actual Magnitude Squares: ", " ", "\n" ); 
         print_vals( mag_sqr_sums.data(), num_to_print, "Actual Magnitude Square Sums: ", " ", "\n" );
      }

      Duration_ms duration_ms = Steady_Clock::now() - start;
      gpu_milliseconds = duration_ms.count();
      dout << __func__ << "(): CUDA kernels took " << gpu_milliseconds << " ms\n";

      float max_diff = 0.01;
      bool all_close = false;
      if ( debug ) {
         print_results( "Norms: " );
         std::cout << "\n"; 
      }
      dout << __func__ << "(): norms Check:\n"; 
      all_close = vals_are_close( norms.data(), exp_norms.data(), num_samples, max_diff, "norms: ", debug );
      if (!all_close) {
         throw std::runtime_error{ std::string{__func__} + 
            std::string{"(): Mismatch between actual norms from GPU and expected norms."} };
      }
      dout << "\n"; 
      
      std::cout << "All " << num_samples << " Norm Values matched expected values. Test Passed.\n\n"; 

      std::cout << "That's a rate of " << ( (num_samples*1000.f)/gpu_milliseconds ) << " samples processed per second for the GPU\n"; 


   } catch( std::exception& ex ) {
      std::cout << __func__ << "(): " << ex.what() << "\n"; 
   }
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

   for( int index = 0; index < num_samples; ++index ) {
      exp_conj_sqr_sums[index] = complex_divide_by_scalar( exp_conj_sqr_sums[index], (float)conj_sqrs_window_size );
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

   for( int index = 0; index  < num_samples; ++index ) {
      exp_mag_sqr_sums[index] = exp_mag_sqr_sums[index]/(float)mag_sqrs_window_size;
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

      std::cout << "It took the CPU " << cpu_milliseconds << " milliseconds to process " << num_samples << " samples\n";
      std::cout << "That's a rate of " << ((num_samples*1000.f)/cpu_milliseconds) << " samples processed per second\n\n"; 

   } catch( std::exception& ex ) {
      throw std::runtime_error( std::string{__func__} +  std::string{"(): "} + ex.what() ); 
   }
}


void NormAutocorrGPU::gen_expected_norms() {
   try { 
      
      if ( test_select_string == "Filebased" ) {
         read_binary_file<cufftComplex>( exp_samples_d16.data(), exp_samples_d16_filepath.c_str(), num_samples, debug );
         read_binary_file<cufftComplex>( exp_conj_sqrs.data(), exp_conj_sqrs_filepath.c_str(), num_samples, debug );
         read_binary_file<cufftComplex>( exp_conj_sqr_sums.data(), exp_conj_sqr_sums_filepath.c_str(), num_samples, debug );
         read_binary_file<float>( exp_conj_sqr_sum_mags.data(), exp_conj_sqr_sum_mags_filepath.c_str(), num_samples, debug );
         read_binary_file<float>( exp_mag_sqrs.data(), exp_mag_sqrs_filepath.c_str(), num_samples, debug );
         read_binary_file<float>( exp_mag_sqr_sums.data(), exp_mag_sqr_sums_filepath.c_str(), num_samples, debug );
         read_binary_file<float>( exp_norms.data(), exp_norms_filepath.c_str(), num_samples, debug );

      } else {
         
         cpu_run();
      }

   } catch( std::exception& ex ) {
      throw std::runtime_error( std::string{__func__} +  std::string{"(): "} + ex.what() ); 
   }

}

// private function
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
} // end of delay_vals16 


NormAutocorrGPU::~NormAutocorrGPU() {
   dout << "dtor called\n";
   samples.clear();    
   samples_d16.clear();
   conj_sqrs.clear();
   conj_sqr_sums.clear();
   conj_sqr_sum_mags.clear();
   mag_sqrs.clear();
   mag_sqr_sums.clear();
   norms.clear();

   exp_samples_d16.clear();
   exp_conj_sqrs.clear();
   exp_conj_sqr_sums.clear();
   exp_conj_sqr_sum_mags.clear();
   exp_mag_sqrs.clear();
   exp_mag_sqr_sums.clear();
   exp_norms.clear();

   if ( stream_ptr ) cudaStreamDestroy( *(stream_ptr.get()) );
   
   dout << "dtor done\n";
} // end of destructor

