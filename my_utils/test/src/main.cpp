// C++ File for main

#include "my_utils/my_utils.hpp"

void test_my_utils( const bool& debug ) {
   dout << "This is a test of my_utils. If you see this then, my_utils library works?\n";
   std::cout << __func__ << "(): Done\n"; 
}

int main( int argc, char* argv[] ) {

   bool debug = true;
   test_my_utils( debug );
}

// end of C++ file for main
