#include <numeric>
#include <memory>
#include <exception>

#include "my_cuda_utils.hpp"

#include "norm_auto_kernel.cuh"

#include "managed_allocator_global.hpp"
#include "managed_allocator.hpp"

#include "VariadicToOutputStream.hpp"

class NormAutoGPU {
private:
   managed_vector_host<cufftComplex> samples;
   device_vector<cufftComplex> samples_d16;
   device_vector<cufftComplex> conj_sqrs;
   device_vector<cufftComplex> conj_sqr_means;
   device_vector<float> conj_sqr_mean_mags;
   device_vector<float> mag_sqrs;
   device_vector<float> mag_sqr_means;
   managed_vector_global<float> norms;

   int num_samples;
   int conj_window_size;
   int mag_sqrs_window_size;
   int max_num_iters;
   bool debug;

   std::unique_ptr<cudaStream_t> stream_ptr;

public:
   NormAutoGPU():
      num_samples(0),
      conj_window_size(0),
      mag_sqrs_window_size(0),
      max_num_iters(0),
      debug(false) {}
   
   
   NormAutoGPU( 
      int new_num_samples, 
      int new_conj_window_size,
      int new_mag_sqrs_window_size,
      int new_max_num_iters,
      const bool new_debug ):
         num_samples( new_num_samples ),
         conj_window_size( new_conj_window_size ),
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
         conj_sqr_mean_mags.reserve( num_samples );
         mag_sqrs.reserve( num_samples );
         mag_sqr_means.reserve( num_samples );
         norms.reserve( num_samples );
         debug_cout( debug, __func__, "(): after reserving vectors for sums, lvals and rvals\n" );

         stream_ptr = my_make_unique<cudaStream_t>();
         try_cudaStreamCreate( stream_ptr.get() );
         debug_cout( debug, __func__,  "(): after cudaStreamCreate()\n" ); 

      } catch( std::exception& ex ) {
         throw std::runtime_error{
            std::string{__func__} + std::string{"(): "} + ex.what(); 
         }; 
      }
   }

   void gen_data( int seed = 0 ) {
      samples.resize(num_samples);
      gen_cufftComplexes( samples.data(), num_samples, -50.0, 50.0 );

      if (debug) {
         print_vec<cufftComplex>( samples, num_samples, "Generated Samples:\n", " " ); 
      }
   }

   void run();
   
   void print_results( const std::string& prefix = "Norms: " ) {
      print_vec<float>( norms, num_samples, prefix.data(), " " );
   }

   ~NormAutoGPU() {
      debug_cout( debug, "dtor called\n" );
      samples.clear();    
      norms.clear();
      if ( stream_ptr ) cudaStreamDestroy( *(stream_ptr.get()) );
      debug_cout( debug, "dtor done\n" );
   }

};


