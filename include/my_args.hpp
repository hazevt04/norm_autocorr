#pragma once

#include "my_utils.hpp"

typedef struct my_args_s {
   std::string test_select_string = "Sinusoidal";
   std::string filename = "";
   std::string exp_norms_filename = "";
   int delay = 16;
   int mag_sqrs_window_size = 64;
   // conj_window_size  = mag_sqrs_window_size - delay
   int conj_sqrs_window_size = 48;
   int num_samples = 128000;
   int max_num_iters = 4000;
   bool debug = false;
   bool help_showed =false;
} my_args_t;

