// C++ File for main
#include "my_cuda_utils/managed_allocator_host.hpp"

#include "my_cuda_utils/my_cuda_utils.hpp"

#include "my_utils/my_utils.hpp"

#include <algorithm>
#include <cuda_runtime.h>

void test_my_cuda_utils( managed_vector_host<float>& vals, const bool& debug ) {
   dout << "This is a test of my_utils. If you see this then, my_cuda_utils library works?\n";
   
   std::cout << "Vals are:\n"; 
   for ( const auto& val: vals ) {
      std::cout << val << "\n"; 
   }
   std::cout << "\n"; 

   std::cout << __func__ << "(): Done\n"; 
}

int main( int argc, char* argv[] ) {

   bool debug = true;
   managed_vector_host<float> vals;
   vals.resize(4);
   vals.reserve(4);

   std::iota( vals.begin(), vals.end(), 0 ); 
   test_my_cuda_utils( vals, debug );
}

// end of C++ file for main
