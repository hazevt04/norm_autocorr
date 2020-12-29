#pragma once

#include <string>

#include <cufft.h>

typedef enum mode_select_e { Sinusoidal, Random, Filebased } mode_select_t;

const std::string default_filename = "input_samples.5.9GHz.10MHzBW.560u.LS.dat"; 
const std::string default_norm_filename = "norm_autocorr.5.9GHz.10MHzBW.560u.LS.dat"; 

constexpr int default_seed = 0;
constexpr int default_num_samples = 4000;
constexpr size_t default_num_sample_bytes = default_num_samples * sizeof(cufftComplex);
constexpr size_t default_num_norm_bytes = default_num_samples * sizeof(float);

constexpr int default_adjusted_num_samples = 4096;
constexpr size_t default_adjusted_num_sample_bytes = default_adjusted_num_samples * sizeof(cufftComplex);
constexpr size_t default_adjusted_num_norm_bytes = default_adjusted_num_samples * sizeof(float);

constexpr int default_threads_per_block = 1024;
constexpr int default_num_blocks = (default_adjusted_num_samples + (default_threads_per_block-1))/default_threads_per_block;

const mode_select_t default_mode_select = mode_select_t::Sinusoidal;

mode_select_t decode_mode_select_string( std::string mode_select_string ); 

std::string get_mode_select_string( mode_select_t mode_select );

typedef struct my_args_s {
   mode_select_t mode_select = default_mode_select;
   std::string filename = default_filename;
   int threads_per_block = default_threads_per_block;
   int num_samples = default_num_samples;
   int seed = default_seed;
   bool debug = false;
   bool help_shown = false;

} my_args_t;
