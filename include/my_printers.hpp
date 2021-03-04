#pragma once

#include "my_utils.hpp"


template <class T>
void print_vals(const std::vector<T>& vals,
   const char* prefix = "",
   const char* line_prefix = "",
   const char* line_suffix = "\n",
   const char* suffix = "\n") {

   int num_vals = vals.size();
   std::cout << prefix;
   for (int index = 0; index < num_vals; ++index) {
      std::cout << line_prefix << vals[index] << line_suffix; 
   }
   std::cout << suffix; 
}


template <class T>
void print_vals(const T* vals,
   const int num_vals,
   const char* prefix = "",
   const char* line_prefix = "",
   const char* line_suffix = "\n",
   const char* suffix = "\n") {

   std::cout << prefix; 
   for (int index = 0; index < num_vals; ++index) {
      std::cout << line_prefix << vals[index] << line_suffix;
   }
   std::cout << suffix;
}

