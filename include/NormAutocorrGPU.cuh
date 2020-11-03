#include <numeric>
#include <memory>
#include <exception>

#include "my_cuda_utils.hpp"

#include <algorithm>
#include <numeric>

#include "norm_autocorr_kernel.cuh"

#include "device_allocator.hpp"
#include "managed_allocator_global.hpp"
#include "managed_allocator_host.hpp"

#include "VariadicToOutputStream.hpp"

class NormAutocorrGPU {
private:
   managed_vector_host<cufftComplex> samples;
   device_vector<cufftComplex> samples_d16;
   device_vector<cufftComplex> conj_sqrs;
   device_vector<cufftComplex> conj_sqr_means;
   device_vector<float> conj_sqr_mean_mags;
   device_vector<float> mag_sqrs;
   device_vector<float> mag_sqr_means;
   managed_vector_global<float> norms;

   cufftComplex* exp_samples_d16;
   cufftComplex* exp_conj_sqrs;
   cufftComplex* exp_conj_sqr_means;
   float* exp_conj_sqr_mean_mags;
   float* exp_mag_sqrs;
   float* exp_mag_sqr_means;
   float* exp_norms;

   int num_samples;
   int conj_sqrs_window_size;
   int mag_sqrs_window_size;
   int max_num_iters;
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
   NormAutocorrGPU():
      num_samples(4000),
      conj_sqrs_window_size(48),
      mag_sqrs_window_size(64),
      max_num_iters(4000),
      debug(false) {}
   
   
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
         samples.reserve( num_samples );
         samples_d16.reserve( num_samples );
         conj_sqrs.reserve( num_samples );
         conj_sqr_means.reserve( num_samples );
         conj_sqr_mean_mags.reserve( num_samples );
         mag_sqrs.reserve( num_samples );
         mag_sqr_means.reserve( num_samples );
         norms.reserve( num_samples );

         exp_samples_d16 = new cufftComplex[num_samples];
         exp_conj_sqrs = new cufftComplex[num_samples];
         exp_conj_sqr_means = new cufftComplex[num_samples];
         exp_conj_sqr_mean_mags = new float[num_samples];
         exp_mag_sqrs = new float[num_samples];
         exp_mag_sqr_means = new float[num_samples];
         exp_norms = new float[num_samples];

         stream_ptr = my_make_unique<cudaStream_t>();
         try_cudaStreamCreate( stream_ptr.get() );
         debug_cout( debug, __func__,  "(): after cudaStreamCreate()\n" ); 

      } catch( std::exception& ex ) {
         throw std::runtime_error{
            std::string{__func__} + std::string{"(): "} + ex.what()
         }; 
      }
   }

   void gen_data( int seed = 0 ) {
      samples.resize(num_samples);
      gen_cufftComplexes( samples.data(), num_samples, -50.0, 50.0 );

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
      if ( stream_ptr ) cudaStreamDestroy( *(stream_ptr.get()) );
      dout << "dtor done\n";
   }

};


