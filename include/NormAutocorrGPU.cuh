#pragma once

#include "device_allocator.hpp"
#include "managed_allocator_host.hpp"
#include "managed_allocator_global.hpp"
#include "man_vec_file_io_funcs.hpp"

#include "my_args.hpp"

#include "my_cufft_utils.hpp"
#include "my_cuda_utils.hpp"
#include "my_utils.hpp"

constexpr float PI = 3.1415926535897238463f;
constexpr float FREQ = 1000.f;
constexpr float AMPLITUDE = 50.f;

const int conj_sqrs_window_size = 48;
const int mag_sqrs_window_size = 64;

class NormAutocorrGPU {
public:
   NormAutocorrGPU(){}
   
   NormAutocorrGPU( 
      const int new_num_samples, 
      const int new_threads_per_block,
      const int new_seed,
      const mode_select_t new_mode_select,
      const std::string new_filename,
      const bool new_debug );

   NormAutocorrGPU( 
      my_args_t my_args ):
         NormAutocorrGPU(
            my_args.num_samples,
            my_args.threads_per_block,
            my_args.seed,
            my_args.mode_select,
            my_args.filename,
            my_args.debug ) {}
   

   void initialize_samples( );
   
   void check_results( const std::string& prefix );

   void cpu_run();
   void gen_expected_norms();

   void run();

   void print_results( const std::string& prefix );

   ~NormAutocorrGPU();
private:
   void delay_vals16();
 
   void calc_norms();
   void calc_mags();
   void calc_complex_mag_squares(); 
   void calc_auto_corrs();
   void calc_exp_conj_sqr_means();
   void calc_exp_mag_sqr_means();

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

   mode_select_t mode_select = default_mode_select;
   
   std::string filename = default_filename;
   std::string norm_filename = default_norm_filename;
   
   std::string filepath = "";
   std::string norm_filepath = "";

   int seed = default_seed;
   
   int num_samples = default_num_samples;
   int threads_per_block = default_threads_per_block;
   int num_blocks = default_num_blocks;
   int device_id = -1;
   int adjusted_num_samples = default_adjusted_num_samples;
   bool debug = false;

   bool can_prefetch = false;
   bool can_map_memory = false;
   bool gpu_is_integrated = false;
   
   size_t num_sample_bytes = default_num_sample_bytes; 
   size_t adjusted_num_sample_bytes = default_adjusted_num_sample_bytes; 
   size_t num_norm_bytes = default_num_norm_bytes;
   size_t adjusted_num_norm_bytes = default_num_norm_bytes; 

   std::unique_ptr<cudaStream_t> stream_ptr;
};


