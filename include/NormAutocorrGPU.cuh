#include <numeric>
#include <memory>
#include <exception>
#include <algorithm>
#include <numeric>

#include "my_args.hpp"

#include "my_cuda_utils.hpp"
#include "pinned_mapped_vec_file_io_funcs.hpp"

#include "norm_autocorr_kernel.cuh"

#include "device_allocator.hpp"
#include "pinned_mapped_allocator.hpp"

constexpr float PI = 3.1415926535897238463f;
constexpr float FREQ = 1000.f;
constexpr float AMPLITUDE = 50.f;
constexpr int threads_per_block = 1024;

const std::string default_filename = "/home/glenn/Sandbox/CUDA/norm_autocorr/input_samples.5.9GHz.10MHzBW.560u.LS.dat"; 

class NormAutocorrGPU {
public:
   NormAutocorrGPU(){}
   
   NormAutocorrGPU( 
      int new_num_samples, 
      int new_conj_sqrs_window_size,
      int new_mag_sqrs_window_size,
      int new_max_num_iters,
      std::string new_test_select_string,
      std::string new_filename,
      const bool new_debug ):
         num_samples( new_num_samples ),
         conj_sqrs_window_size( new_conj_sqrs_window_size ),
         mag_sqrs_window_size( new_mag_sqrs_window_size ),
         max_num_iters( new_max_num_iters ),
         test_select_string( new_test_select_string ),
         filename( new_filename ),
         debug( new_debug ) {
   
      try {
         cudaError_t cerror = cudaSuccess;

         try_cuda_func_throw( cerror, cudaGetDevice( &device_id ) );
         dout << __func__ << "(): Device ID is " << device_id << "\n";

         cudaDeviceProp prop;
         
         try_cuda_func_throw( cerror, cudaGetDeviceProperties(&prop, device_id) );
         
         can_prefetch = prop.concurrentManagedAccess;
         if ( !can_prefetch ) {
            dout << __func__ << "(): cudaMemPrefetchAsync() not supported\n";
         }

         dout << __func__ << "(): num_samples is " << num_samples << "\n";

         num_blocks = (num_samples + (threads_per_block-1))/threads_per_block;

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
         conj_sqr_means.reserve( adjusted_num_samples );
         conj_sqr_mean_mags.reserve( adjusted_num_samples );
         mag_sqrs.reserve( adjusted_num_samples );
         mag_sqr_means.reserve( adjusted_num_samples );
         norms.reserve( adjusted_num_samples );

         //d_samples.reserve( adjusted_num_samples );
         //d_norms.reserve( adjusted_num_samples );
         try_cuda_func_throw( cerror, cudaHostGetDevicePointer( &d_samples, samples.data(), 0 ) );
         try_cuda_func_throw( cerror, cudaHostGetDevicePointer( &d_norms, norms.data(), 0 ) );
         
         exp_samples_d16 = new cufftComplex[num_samples];
         exp_conj_sqrs = new cufftComplex[num_samples];
         exp_conj_sqr_means = new cufftComplex[num_samples];
         exp_conj_sqr_mean_mags = new float[num_samples];
         exp_mag_sqrs = new float[num_samples];
         exp_mag_sqr_means = new float[num_samples];
         exp_norms = new float[num_samples];

         for( int index = 0; index < num_samples; ++index ) {

            exp_samples_d16[index] = make_cuFloatComplex(0.f,0.f);
            exp_conj_sqrs[index] =  make_cuFloatComplex(0.f,0.f);
            exp_conj_sqr_means[index] = make_cuFloatComplex(0.f,0.f);
            exp_mag_sqr_means[index] = 0.f;
         } 

         samples.resize(adjusted_num_samples);
         norms.resize(adjusted_num_samples);
         
         initialize_samples();

         //std::fill( samples_d16.begin(), samples_d16.end(), make_cuFloatComplex(0.f,0.f) );
         //std::fill( conj_sqrs.begin(), conj_sqrs.end(), make_cuFloatComplex(0.f,0.f) );
         //std::fill( conj_sqr_means.begin(), conj_sqr_means.end(), make_cuFloatComplex(0.f,0.f) );
         //std::fill( conj_sqr_mean_mags.begin(), conj_sqr_mean_mags.end(), 0 );
         //std::fill( mag_sqrs.begin(), mag_sqrs.end(), 0 );
         //std::fill( mag_sqr_means.begin(), mag_sqr_means.end(), 0 );
         //std::fill( norms.begin(), norms.end(), 0 );

         stream_ptr = my_make_unique<cudaStream_t>();
         try_cudaStreamCreate( stream_ptr.get() );
         dout << __func__ << "(): after cudaStreamCreate()\n"; 

      } catch( std::exception& ex ) {
         throw std::runtime_error{
            std::string{__func__} + std::string{"(): "} + ex.what()
         }; 
      }
   }

   NormAutocorrGPU( 
      int new_num_samples, 
      int new_conj_sqrs_window_size,
      int new_mag_sqrs_window_size,
      int new_max_num_iters,
      my_args_t my_args ):
         NormAutocorrGPU(
            new_num_samples,
            new_conj_sqrs_window_size,
            new_mag_sqrs_window_size,
            new_max_num_iters,
            my_args.test_select_string,
            my_args.filename,
            my_args.debug ) {}
   
   void initialize_samples( const int seed = 0, const bool debug = false ) {
      try {
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
               filename.c_str(),
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

   void run();
   void cpu_run();
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
private:
   inline void delay_vals16() {
      
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

   void calc_norms();
   void calc_mags();
   void calc_complex_mag_squares(); 
   void calc_auto_corrs();
   void calc_exp_conj_sqr_means();
   void calc_exp_mag_sqr_means();

   pinned_mapped_vector<cufftComplex> samples;
   device_vector<cufftComplex> samples_d16;
   device_vector<cufftComplex> conj_sqrs;
   device_vector<cufftComplex> conj_sqr_means;
   device_vector<float> conj_sqr_mean_mags;
   device_vector<float> mag_sqrs;
   device_vector<float> mag_sqr_means;
   pinned_mapped_vector<float> norms;

   cufftComplex* d_samples;
   float* d_norms;
   //device_vector<cufftComplex> d_samples;
   //device_vector<float> d_norms;

   cufftComplex* exp_samples_d16;
   cufftComplex* exp_conj_sqrs;
   cufftComplex* exp_conj_sqr_means;
   float* exp_conj_sqr_mean_mags;
   float* exp_mag_sqrs;
   float* exp_mag_sqr_means;
   float* exp_norms;

   std::string test_select_string;
   std::string filename = default_filename;

   size_t num_sample_bytes = 32000;
   size_t adjusted_num_sample_bytes = 32768;
   size_t num_norm_bytes = 16000;
   size_t adjusted_num_norm_bytes = 16384;

   int device_id = 0;
   int num_blocks = 4;
   int threads_per_block = 1024;
   int num_samples = 4000;
   int adjusted_num_samples = 4096;
   int conj_sqrs_window_size = 48;
   int mag_sqrs_window_size = 64;
   int max_num_iters = 4000;
   bool debug = false;

   bool can_prefetch = false;
   std::unique_ptr<cudaStream_t> stream_ptr;
};


