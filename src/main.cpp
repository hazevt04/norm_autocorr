// C++ File for main

#include "NormAutocorrGPU.cuh"

int main(int argc, char **argv) {
   try {
      int num_vals = 4000;
      int conj_window_size = 48;
      int mag_sqrs_window_size = 64;
      int max_num_iters = 4000;
      bool debug = true;
      
      std::cout << "Number of Vals = " << num_vals << "\n"; 
      NormAutocorrGPU norm_autocorr_gpu{ num_vals, conj_window_size, mag_sqrs_window_size, max_num_iters, debug };
      norm_autocorr_gpu.run();
      return EXIT_SUCCESS;

   } catch( std::exception& ex ) {
      std::cout << __func__ << "(): ERROR: " << ex.what() << "\n"; 
      return EXIT_FAILURE;

   }
}
// end of C++ file for main
