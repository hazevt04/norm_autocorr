// C++ File for main

#include "NormAutocorrGPU.hpp"

#include "parse_args.hpp"

int main(int argc, char **argv) {
   try {
      my_args_t my_args;
      parse_args( my_args, argc, argv ); 

      if ( my_args.help_showed ) {
         return EXIT_SUCCESS;
      }

      NormAutocorrGPU norm_autocorr_gpu{ my_args };

      norm_autocorr_gpu.run();
      return EXIT_SUCCESS;

   } catch( std::exception& ex ) {
      std::cout << __func__ << "(): ERROR: " << ex.what() << "\n"; 
      return EXIT_FAILURE;

   }
}
// end of C++ file for main
