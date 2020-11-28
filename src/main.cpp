// C++ File for main

#include "NormAutocorrGPU.cuh"

#include "parse_args.hpp"

int main(int argc, char **argv) {
   try {
      int num_vals = 4000;
      int conj_window_size = 48;
      int mag_sqrs_window_size = 64;
      int max_num_iters = 4000;
      
      my_args_t my_args;
      parse_args( my_args, argc, argv ); 

      if ( my_args.help_showed ) {
         return EXIT_SUCCESS;
      }

      NormAutocorrGPU norm_autocorr_gpu{ num_vals, conj_window_size, 
         mag_sqrs_window_size, max_num_iters, my_args };

      norm_autocorr_gpu.run();
      return EXIT_SUCCESS;

   } catch( std::exception& ex ) {
      std::cout << __func__ << "(): ERROR: " << ex.what() << "\n"; 
      return EXIT_FAILURE;

   }
}
// end of C++ file for main
