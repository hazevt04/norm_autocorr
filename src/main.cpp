// C++ File for main

#include "NormAutoGPU.cuh"

int main(int argc, char **argv) {
   try {
      int num_vals = 4000;
      std::cout << "Number of Vals = " << num_vals << "\n"; 
      bool debug = false;
      NormAutoGPU norm_auto_gpu{ num_vals, debug };
      norm_auto_gpu.run();
      return EXIT_SUCCESS;

   } catch( std::exception& ex ) {
      std::cout << __func__ << "(): ERROR: " << ex.what() << "\n"; 
      return EXIT_FAILURE;

   }
}
// end of C++ file for main
