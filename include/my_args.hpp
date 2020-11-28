#pragma once

#include "my_utils.hpp"

typedef struct my_args_s {
   std::string test_select_string = "Sinusoidal";
   std::string filename = "";
   bool debug = false;
   bool help_showed =false;
} my_args_t;

