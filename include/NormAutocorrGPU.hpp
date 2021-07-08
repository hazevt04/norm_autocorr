
#include "norm_autocorr_kernel.cuh"

#include "man_vec_file_io_funcs.hpp"

#include "managed_allocator_global.hpp"
#include "managed_allocator_host.hpp"

#include "my_cuda_utils.hpp"
#include "my_args.hpp"

#include <numeric>
#include <memory>
#include <exception>
#include <algorithm>
#include <numeric>

constexpr float PI = 3.1415926535897238463f;
constexpr float FREQ = 1000.f;
constexpr float AMPLITUDE = 50.f;
constexpr int threads_per_block = 1024;

const std::string default_filename = "samples.dat"; 
const std::string default_exp_norms_filename = "exp_norms.dat";

class NormAutocorrGPU {
public:
   NormAutocorrGPU(){}
   
   NormAutocorrGPU( 
      const my_args_t& args
   );

   void initialize_samples( const int seed, const bool debug );

   void run();
   void cpu_run();
   void gen_expected_norms();

   inline void print_results( const std::string& prefix = "Norms: " ) {
      print_vals<float>( norms.data(), num_samples, "Norms: ",  " ",  "\n" );
   }
   
   ~NormAutocorrGPU();

private:
   void delay_vals16();

   void calc_norms();
   void calc_mags();
   void calc_complex_mag_squares(); 
   void calc_auto_corrs();
   void calc_exp_conj_sqr_sums();
   void calc_exp_mag_sqr_sums();

   managed_vector_host<cufftComplex> samples;
   managed_vector_global<cufftComplex> samples_d16;
   managed_vector_global<cufftComplex> conj_sqrs;
   managed_vector_global<cufftComplex> conj_sqr_sums;
   managed_vector_global<float> conj_sqr_sum_mags;
   managed_vector_global<float> mag_sqrs;
   managed_vector_global<float> mag_sqr_sums;
   managed_vector_global<float> norms;

   std::vector<cufftComplex> exp_samples_d16;
   std::vector<cufftComplex> exp_conj_sqrs;
   std::vector<cufftComplex> exp_conj_sqr_sums;
   std::vector<float> exp_conj_sqr_sum_mags;
   std::vector<float> exp_mag_sqrs;
   std::vector<float> exp_mag_sqr_sums;
   std::vector<float> exp_norms;

   std::string test_select_string;
   
   std::string filename = default_filename;
   std::string exp_norms_filename = default_exp_norms_filename;

   std::string exp_samples_d16_filename = "exp_samples_d16.dat";
   std::string exp_conj_sqrs_filename = "exp_conj_sqrs.dat";
   std::string exp_conj_sqr_sums_filename = "exp_conj_sqr_sums.dat";
   std::string exp_conj_sqr_sum_mags_filename = "exp_conj_sqr_sum_mags.dat";
   std::string exp_mag_sqrs_filename = "exp_mag_sqrs.dat";
   std::string exp_mag_sqr_sums_filename = "exp_mag_sqr_sums.dat";

   std::string filepath = "";
   std::string exp_norms_filepath = "";
   std::string exp_samples_d16_filepath = "";
   std::string exp_conj_sqrs_filepath = "";
   std::string exp_conj_sqr_sums_filepath = "";
   std::string exp_conj_sqr_sum_mags_filepath = "";
   std::string exp_mag_sqrs_filepath = "";
   std::string exp_mag_sqr_sums_filepath = "";

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

   std::unique_ptr<cudaStream_t> stream_ptr;
};


