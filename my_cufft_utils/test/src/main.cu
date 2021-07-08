// C++ File for main
#include "my_cuda_utils/managed_allocator_host.hpp"

#include "my_cufft_utils/my_cufft_utils.hpp"
#include "my_cuda_utils/my_cuda_utils.hpp"

#include "my_utils/my_utils.hpp"

#include <algorithm>
#include <cuda_runtime.h>
#include <cufft.h>

void test_my_cuda_utils( managed_vector_host<cufftComplex>& vals, const bool& debug ) {
   dout << "This is a test of my_utils. If you see this then, my_cuda_utils library works?\n";
   
   print_cufftComplexes( vals.data(), int(vals.size()), "Vals are:\n", "\n", "\n" ); 

   std::cout << __func__ << "(): Done\n"; 
}

int main( int argc, char* argv[] ) {

   int num_vals = 4;
   bool debug = true;
   managed_vector_host<cufftComplex> vals;
   vals.resize(num_vals);
   vals.reserve(num_vals);

   gen_cufftComplexes( vals.data(), num_vals, 0.f, 10.f );
   test_my_cuda_utils( vals, debug );
}

// end of C++ file for main
