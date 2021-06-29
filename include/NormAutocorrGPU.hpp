
#include "norm_autocorr_kernel.cuh"

#include "pinned_mapped_vec_file_io_funcs.hpp"

#include "device_allocator.hpp"
#include "pinned_mapped_allocator.hpp"

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

const std::string default_filename = "input_samples.5.180GHz.20MHzBW.560u.LS.dat"; 
const std::string default_exp_norms_filename = "exp_norms.5.180GHz.20MHzBW.560u.LS.dat";

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
   void calc_exp_conj_sqr_means();
   void calc_exp_mag_sqr_means();

   pinned_mapped_vector<cufftComplex> samples;
   //device_vector<cufftComplex> d_samples;
   device_vector<cufftComplex> samples_d16;
   device_vector<cufftComplex> conj_sqrs;
   device_vector<cufftComplex> conj_sqr_means;
   device_vector<float> conj_sqr_mean_mags;
   device_vector<float> mag_sqrs;
   device_vector<float> mag_sqr_means;
   //device_vector<float> d_norms;
   pinned_mapped_vector<float> norms;

   float* d_norms;
   cufftComplex* d_samples;

   std::vector<cufftComplex> exp_samples_d16;
   std::vector<cufftComplex> exp_conj_sqrs;
   std::vector<cufftComplex> exp_conj_sqr_means;
   std::vector<float> exp_conj_sqr_mean_mags;
   std::vector<float> exp_mag_sqrs;
   std::vector<float> exp_mag_sqr_means;
   std::vector<float> exp_norms;

   std::string test_select_string;
   
   std::string filename = default_filename;
   std::string exp_norms_filename = default_exp_norms_filename;
   
   std::string filepath = "";
   std::string exp_norms_filepath = "";

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


