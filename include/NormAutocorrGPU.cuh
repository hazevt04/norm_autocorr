#include <numeric>
#include <memory>
#include <exception>
#include <algorithm>
#include <numeric>

#include "my_cuda_utils.hpp"
#include "man_vec_file_io_funcs.hpp"

#include "norm_autocorr_kernel.cuh"

#include "device_allocator.hpp"
#include "managed_allocator_global.hpp"
#include "managed_allocator_host.hpp"

#include "VariadicToOutputStream.hpp"

constexpr float PI = 3.1415926535897238463f;
constexpr float FREQ = 1000.f;
constexpr float AMPLITUDE = 50.f;
constexpr int threads_per_block = 1024;

class NormAutocorrGPU {
private:
   managed_vector_host<cufftComplex> samples;
   managed_vector_global<cufftComplex> samples_d16;
   managed_vector_global<cufftComplex> conj_sqrs;
   managed_vector_global<cufftComplex> conj_sqr_means;
   managed_vector_global<float> conj_sqr_mean_mags;
   managed_vector_global<float> mag_sqrs;
   managed_vector_global<float> mag_sqr_means;
   managed_vector_global<float> norms;

   cufftComplex* exp_samples_d16;
   cufftComplex* exp_conj_sqrs;
   cufftComplex* exp_conj_sqr_means;
   float* exp_conj_sqr_mean_mags;
   float* exp_mag_sqrs;
   float* exp_mag_sqr_means;
   float* exp_norms;

   int num_samples = 4000;
   int adjusted_num_samples = 4096;
   int conj_sqrs_window_size = 48;
   int mag_sqrs_window_size = 64;
   int max_num_iters = 4000;
   bool debug;

   std::unique_ptr<cudaStream_t> stream_ptr;

   inline void delay_vals16() {
      
      dout << __func__ << "() start\n";
      dout << __func__ << "() samples.size() is " << samples.size() << "\n";
      dout << __func__ << "() samples[0] is " << samples[0] << "\n";
      dout << __func__ << "() samples[1] is " << samples[1] << "\n";

      for( int index = 0; index < num_samples; ++index ) {
         if ( index < 16 ) {
            dout << __func__ << "() index: " << index << "\n";
            exp_samples_d16[index] = make_cuFloatComplex(0.f, 0.f);
         } else {
            dout << __func__ << "() index greater than or equal to 16: " << index << "\n";
            exp_samples_d16[index] = samples[index-16]; 
         }
      } 

      dout << __func__ << "() exp_samples_d16[15] is " << exp_samples_d16[15] << "\n";
      dout << __func__ << "() exp_samples_d16[16] is " << exp_samples_d16[16] << "\n";
      dout << __func__ << "() exp_samples_d16[17] is " << exp_samples_d16[17] << "\n";
      dout << __func__ << "() done\n";
   }   

   void calc_norms();
   void calc_mags();
   void calc_complex_mag_squares(); 
   void calc_auto_corrs();
   void calc_exp_conj_sqr_means();
   void calc_exp_mag_sqr_means();

public:
   NormAutocorrGPU(){}
   
   NormAutocorrGPU( 
      int new_num_samples, 
      int new_conj_sqrs_window_size,
      int new_mag_sqrs_window_size,
      int new_max_num_iters,
      const bool new_debug ):
         num_samples( new_num_samples ),
         conj_sqrs_window_size( new_conj_sqrs_window_size ),
         mag_sqrs_window_size( new_mag_sqrs_window_size ),
         max_num_iters( new_max_num_iters ),
         debug( new_debug ) {
   
      try {
         debug_cout( debug, __func__, "(): num_samples is ", num_samples, "\n" );

         int resize_factor = (num_samples + (threads_per_block-1))/threads_per_block;

         adjusted_num_samples = threads_per_block * resize_factor;
         debug_printf( debug, "%s(): adjusted number of samples for allocation is %d\n", 
            __func__, adjusted_num_samples ); 

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
            exp_conj_sqrs[index] = 
            exp_conj_sqr_means[index] = make_cuFloatComplex(0.f,0.f);
            exp_mag_sqr_means[index] = 0.f;
         } 

         samples.resize(adjusted_num_samples);
         
         //initialize_samples();
         read_binary_file<cufftComplex>( samples, "/home/glenn/Sandbox/CUDA/norm_autocorr/input_samples.5.9GHz.10MHzBW.560u.LS.dat", num_samples, debug );

         gen_expected_norms();

         std::fill( samples_d16.begin(), samples_d16.end(), make_cuFloatComplex(0.f,0.f) );
         std::fill( conj_sqrs.begin(), conj_sqrs.end(), make_cuFloatComplex(0.f,0.f) );
         std::fill( conj_sqr_means.begin(), conj_sqr_means.end(), make_cuFloatComplex(0.f,0.f) );
         std::fill( conj_sqr_mean_mags.begin(), conj_sqr_mean_mags.end(), 0 );
         std::fill( mag_sqrs.begin(), mag_sqrs.end(), 0 );
         std::fill( mag_sqr_means.begin(), mag_sqr_means.end(), 0 );
         std::fill( norms.begin(), norms.end(), 0 );

         stream_ptr = my_make_unique<cudaStream_t>();
         try_cudaStreamCreate( stream_ptr.get() );
         debug_cout( debug, __func__,  "(): after cudaStreamCreate()\n" ); 

      } catch( std::exception& ex ) {
         throw std::runtime_error{
            std::string{__func__} + std::string{"(): "} + ex.what()
         }; 
      }
   }

   void initialize_samples( int seed = 0 ) {
      gen_cufftComplexes( samples.data(), num_samples, -50.0, 50.0 );
      /*for( size_t index = 0; index < num_samples; ++index ) {*/
      /*   float t_val_real = AMPLITUDE*sin(2*PI*FREQ*index);*/
      /*   float t_val_imag = AMPLITUDE*sin(2*PI*FREQ*index);*/
      /*   samples[index] = make_cuFloatComplex( t_val_real, t_val_imag );*/
      /*} */

      if (debug) {
         print_cufftComplexes( samples.data(), num_samples, "Samples: ",  " ",  "\n" ); 
      }
   }

   void run();
   void gen_expected_norms();

   void print_results( const std::string& prefix = "Norms: " ) {
      print_vals<float>( norms.data(), num_samples, "Norms: ",  " ",  "\n" );
   }

   ~NormAutocorrGPU() {
      dout << "dtor called\n";
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
      
      dout << "dtor done\n";
   }

};


