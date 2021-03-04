#pragma once

#include "device_allocator.hpp"
#include "pinned_allocator.hpp"
#include "pinned_vec_file_io_funcs.hpp"

#include "my_args.hpp"

#include "my_cufft_utils.hpp"
#include "my_cuda_utils.hpp"

#include "my_comparators.hpp"
//#include "my_generators.hpp"
#include "my_printers.hpp"
#include "my_utils.hpp"

constexpr double PI = 3.1415926535897238463f;
constexpr double FREQ = 1000.f;
constexpr double AMPLITUDE = 50.f;

const int MAX_NUM_SAMPLES_INCREASING = 65535;

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
      const bool new_debug 
   ); 

   NormAutocorrGPU( 
      my_args_t my_args ):
         NormAutocorrGPU(
            my_args.num_samples,
            my_args.threads_per_block,
            my_args.seed,
            my_args.mode_select,
            my_args.debug ) {}
   
   void initialize_samples();
   void run_warmup();
   void run_original();
   void run();
   void check_results( const std::string& prefix );

   void cpu_run();
   void gen_expected_norms();

   void print_results( const std::string& prefix );

   ~NormAutocorrGPU();

private:
   mode_select_t mode_select;
   
   size_t num_sample_bytes = 32000;
   size_t adjusted_num_sample_bytes = 32768;
   size_t num_norm_bytes = 16000;
   size_t adjusted_num_norm_bytes = 16384;
   size_t num_shared_bytes = 0;

   int device_id = 0;
   int num_blocks = 4;
   int num_samples = 4000;
   int adjusted_num_samples = 4096;
   int threads_per_block = 1024;
   int seed = 0;
   bool debug = false;

   std::unique_ptr<cudaStream_t> stream_ptr;

   pinned_vector<cufftDoubleComplex> samples;
   device_vector<cufftDoubleComplex> d_samples;
   device_vector<cufftDoubleComplex> samples_d16;
   device_vector<cufftDoubleComplex> conj_sqrs;
   device_vector<cufftDoubleComplex> conj_sqr_sums;
   device_vector<double> conj_sqr_sum_mags;
   device_vector<double> mag_sqrs;
   device_vector<double> mag_sqr_sums;
   device_vector<double> d_norms;
   pinned_vector<double> norms;

   std::vector<cufftDoubleComplex> exp_samples_d16;
   std::vector<cufftDoubleComplex> exp_conj_sqrs;
   std::vector<cufftDoubleComplex> exp_conj_sqr_sums;
   std::vector<double> exp_conj_sqr_sum_mags;
   std::vector<double> exp_mag_sqrs;
   std::vector<double> exp_mag_sqr_sums;
   std::vector<double> exp_norms;
   
   void delay_vals16();
   void calc_norms();
   void calc_mags();
   void calc_complex_mag_squares(); 
   void calc_auto_corrs();
   void calc_exp_conj_sqr_sums();
   void calc_exp_mag_sqr_sums();
};


