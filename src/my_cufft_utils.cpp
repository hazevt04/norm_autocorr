
#include "my_cufft_utils.hpp"

void gen_cufftComplexes( cufftComplex* complexes, const int& num_complexes, const float& lower, const float& upper, const int& seed ) {
   //std::random_device random_dev;
   //std::mt19937 mersenne_gen(random_dev());
   std::mt19937 mersenne_gen((float)seed);
   std::uniform_real_distribution<float> dist(lower, upper);
   for( int index = 0; index < num_complexes; ++index ) {
      complexes[index].x = dist( mersenne_gen );
      complexes[index].y = dist( mersenne_gen );
   } 
}


void gen_cufftDoubleComplexes( cufftDoubleComplex* complexes, const int& num_complexes, const double& lower, const double& upper, const int& seed ) {
   //std::random_device random_dev;
   //std::mt19937 mersenne_gen(random_dev());
   std::mt19937 mersenne_gen((double)seed);
   std::uniform_real_distribution<double> dist(lower, upper);
   for( int index = 0; index < num_complexes; ++index ) {
      complexes[index].x = dist( mersenne_gen );
      complexes[index].y = dist( mersenne_gen );
   } 
}


bool cufftComplexes_are_close( const cufftComplex* lvals, const cufftComplex* rvals, 
    const int& num_vals, const float& max_diff, const std::string& prefix, const bool& debug ) {

    for( size_t index = 0; index < num_vals; ++index ) {
      float abs_diff_real = abs( lvals[index].x - rvals[index].x );
      float abs_diff_imag = abs( lvals[index].y - rvals[index].y );

      dout << "Index: " << index << ": max_diff = " << max_diff 
         << " actual diffs: { " <<  abs_diff_real << ", " << abs_diff_imag << " }\n";
      if ( ( abs_diff_real > max_diff ) || ( abs_diff_imag > max_diff ) ) {
         dout << "Actual: {" << lvals[index].x << ", " << lvals[index].y << "}\n";
         dout << "Expected: {" << rvals[index].x << ", " << rvals[index].y << "}\n";
         return false;
      }
   }
   return true;  
}


bool cufftDoubleComplexes_are_close( const cufftDoubleComplex* lvals, const cufftDoubleComplex* rvals, 
    const int& num_vals, const double& max_diff, const std::string& prefix, const bool& debug ) {

    for( size_t index = 0; index < num_vals; ++index ) {
      double abs_diff_real = abs( lvals[index].x - rvals[index].x );
      double abs_diff_imag = abs( lvals[index].y - rvals[index].y );

      dout << "Index: " << index << ": max_diff = " << max_diff 
         << " actual diffs: { " <<  abs_diff_real << ", " << abs_diff_imag << " }\n";
      if ( ( abs_diff_real > max_diff ) || ( abs_diff_imag > max_diff ) ) {
         dout << "Actual: {" << lvals[index].x << ", " << lvals[index].y << "}\n";
         dout << "Expected: {" << rvals[index].x << ", " << rvals[index].y << "}\n";
         return false;
      }
   }
   return true;  
}

void print_cufftComplexes(const cufftComplex* vals,
   const int& num_vals,
   const char* prefix,
   const char* delim,
   const char* suffix ) {

   std::cout << prefix; 
   for (int index = 0; index < num_vals; ++index) {
      std::cout << "Index " << index << ": " << vals[index] << delim;
   }
   std::cout << suffix;
}


void print_cufftDoubleComplexes(const cufftDoubleComplex* vals,
   const int& num_vals,
   const char* prefix,
   const char* delim,
   const char* suffix ) {

   std::cout << prefix;
   for (int index = 0; index < num_vals; ++index) {
      std::cout << "Index " << index << ": " << vals[index] << delim;
   }
   std::cout << suffix;
}


// end of C++ file for my_cufft_utils
