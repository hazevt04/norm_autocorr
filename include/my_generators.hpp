#pragma once

#include "my_utils.hpp"


template<class RealType>
RealType gen_real( const RealType& lower, const RealType& upper ) {
   std::random_device random_dev;
   std::mt19937 mersenne_gen(random_dev());
   std::uniform_real_distribution<> dist(lower, upper);
   return dist( mersenne_gen );
}


template <class RealType>
std::complex<RealType> gen_complex( const std::complex<RealType>& lower, const std::complex<RealType>& upper ) {
   std::random_device random_dev;
   std::mt19937 mersenne_gen(random_dev());
   std::uniform_real_distribution<RealType> real_dist(lower.real(), upper.real());
   std::uniform_real_distribution<RealType> imag_dist(lower.imag(), upper.imag());
   return std::complex<RealType>{ real_dist( mersenne_gen ), imag_dist( mersenne_gen ) };
}

////////////////////////////
// Generating Multiple Values
////////////////////////////

template <class RealType>
void gen_complex_vals( complex_vector<RealType>& vals, 
        const std::complex<RealType>& lower, const std::complex<RealType>& upper ) {
        
   std::random_device random_dev;
   std::mt19937 mersenne_gen(random_dev());
   std::uniform_real_distribution<RealType> real_dist(lower.real(), upper.real());
   std::uniform_real_distribution<RealType> imag_dist(lower.imag(), upper.imag());
   
   int num_vals = vals.size();
   for( int index = 0; index < num_vals; ++index ) {
      vals.emplace_back( real_dist( mersenne_gen ), imag_dist( mersenne_gen ) );
   } 
}


template <class RealType>
void gen_complex_vals( complex_vector<RealType>& vals, 
        const std::complex<RealType>& lower, const std::complex<RealType>& upper, const int& seed ) {
        
   std::mt19937 mersenne_gen(seed);
   std::uniform_real_distribution<RealType> real_dist(lower.real(), upper.real());
   std::uniform_real_distribution<RealType> imag_dist(lower.imag(), upper.imag());
   
   int num_vals = vals.size();
   for( int index = 0; index < num_vals; ++index ) {
      vals.emplace_back( real_dist( mersenne_gen ), imag_dist( mersenne_gen ) );
   } 
}


//==================================================================

template <class RealType>
void gen_complex_vals( std::complex<RealType>* vals, const int& num_vals,
        const std::complex<RealType>& lower, const std::complex<RealType>& upper ) {
        
   std::random_device random_dev;
   std::mt19937 mersenne_gen(random_dev());
   std::uniform_real_distribution<RealType> real_dist(lower.real(), upper.real());
   std::uniform_real_distribution<RealType> imag_dist(lower.imag(), upper.imag());
   
   for( int index = 0; index < num_vals; ++index ) {
      vals[index] = std::complex<RealType>{ real_dist( mersenne_gen ), imag_dist( mersenne_gen ) };
   } 
}


template <class RealType>
void gen_complex_vals( std::complex<RealType>* vals, const int& num_vals,
        const std::complex<RealType>& lower, const std::complex<RealType>& upper, const int& seed ) {
        
   std::mt19937 mersenne_gen(seed);
   std::uniform_real_distribution<RealType> real_dist(lower.real(), upper.real());
   std::uniform_real_distribution<RealType> imag_dist(lower.imag(), upper.imag());
   
   for( int index = 0; index < num_vals; ++index ) {
      vals[index] = std::complex<RealType>{ real_dist( mersenne_gen ), imag_dist( mersenne_gen ) };
   } 
}

//=================================================================


void gen_complex_vals( complex_float* vals, const int& num_vals, 
        const complex_float& lower, const complex_float& upper ) {
        
   std::random_device random_dev;
   std::mt19937 mersenne_gen(random_dev());
   std::uniform_real_distribution<float> real_dist(lower.real(), upper.real());
   std::uniform_real_distribution<float> imag_dist(lower.imag(), upper.imag());
   
   for( int index = 0; index < num_vals; ++index ) {
      vals[index] = complex_float{ real_dist( mersenne_gen ), imag_dist( mersenne_gen ) };
   } 
}

void gen_complex_vals( complex_float* vals, const int& num_vals,
        const complex_float& lower, const complex_float& upper, const int& seed ) {
        
   std::mt19937 mersenne_gen(seed);
   std::uniform_real_distribution<float> real_dist(lower.real(), upper.real());
   std::uniform_real_distribution<float> imag_dist(lower.imag(), upper.imag());
   
   for( int index = 0; index < num_vals; ++index ) {
      vals[index] = complex_float{ real_dist( mersenne_gen ), imag_dist( mersenne_gen ) };
   } 
}

void gen_complex_vals( complex_double* vals, const int& num_vals,
        const complex_double& lower, const complex_double& upper ) {
        
   std::random_device random_dev;
   std::mt19937 mersenne_gen(random_dev());
   std::uniform_real_distribution<double> real_dist(lower.real(), upper.real());
   std::uniform_real_distribution<double> imag_dist(lower.imag(), upper.imag());
   
   for( int index = 0; index < num_vals; ++index ) {
      vals[index] = complex_double{ real_dist( mersenne_gen ), imag_dist( mersenne_gen ) };
   } 
}

void gen_complex_vals( complex_double* vals, const int& num_vals, 
        const complex_double& lower, const complex_double& upper, const int& seed ) {
        
   std::mt19937 mersenne_gen(seed);
   std::uniform_real_distribution<double> real_dist(lower.real(), upper.real());
   std::uniform_real_distribution<double> imag_dist(lower.imag(), upper.imag());
   
   for( int index = 0; index < num_vals; ++index ) {
      vals[index] = complex_double{ real_dist( mersenne_gen ), imag_dist( mersenne_gen ) };
   } 
}

//==================================================================

template <class RealType, size_t size>
void gen_complex_vals( complex_array<RealType, size>& vals, 
        const std::complex<RealType>& lower, const std::complex<RealType>& upper ) {
        
   std::random_device random_dev;
   std::mt19937 mersenne_gen(random_dev());
   std::uniform_real_distribution<RealType> real_dist(lower.real(), upper.real());
   std::uniform_real_distribution<RealType> imag_dist(lower.imag(), upper.imag());
   
   int num_vals = vals.size();
   for( int index = 0; index < num_vals; ++index ) {
      vals[index] = { real_dist( mersenne_gen ), imag_dist( mersenne_gen ) };
      //vals[index].real = real_dist( mersenne_gen );
      //vals[index].imag = imag_dist( mersenne_gen );
   } 
}


template <class RealType, size_t size>
void gen_complex_vals( complex_array<RealType, size>& vals, 
        const std::complex<RealType>& lower, const std::complex<RealType>& upper, const int& seed ) {
        
   std::mt19937 mersenne_gen(seed);
   std::uniform_real_distribution<RealType> real_dist(lower.real(), upper.real());
   std::uniform_real_distribution<RealType> imag_dist(lower.imag(), upper.imag());
   
   int num_vals = vals.size();
   for( int index = 0; index < num_vals; ++index ) {
      vals[index] = { real_dist( mersenne_gen ), imag_dist( mersenne_gen ) };
      //vals[index].real = real_dist( mersenne_gen );
      //vals[index].imag = imag_dist( mersenne_gen );
   } 
}

///////////////////////////////////////////////////
///////////////////////////////////////////////////

template <class RealType>
void gen_reals( RealType* vals, const int& num_vals, const RealType& lower, const RealType& upper ) {
   std::random_device random_dev;
   std::mt19937 mersenne_gen(random_dev());
   std::uniform_real_distribution<RealType> dist(lower, upper);
   for( int index = 0; index < num_vals; ++index ) {
      vals[index] = dist( mersenne_gen );
   } 
}


template <class RealType>
void gen_reals( RealType* vals, const int& num_vals, const RealType& lower, const RealType& upper, const int& seed ) {
   std::mt19937 mersenne_gen(seed);
   std::uniform_real_distribution<RealType> dist(lower, upper);
   for( int index = 0; index < num_vals; ++index ) {
      vals[index] = dist( mersenne_gen );
   } 
}


template <class RealType>
void gen_reals( std::vector<RealType>& vals, const RealType& lower, const RealType& upper ) {
   std::random_device random_dev;
   std::mt19937 mersenne_gen(random_dev());
   std::uniform_real_distribution<RealType> dist(lower, upper);
   for ( auto& val: vals ) {
      val = dist( mersenne_gen );
   } 
}


template <class RealType>
void gen_reals( std::vector<RealType>& vals, const RealType& lower, const RealType& upper, const int& seed ) {
   std::mt19937 mersenne_gen(seed);
   std::uniform_real_distribution<RealType> dist(lower, upper);
   for ( auto& val: vals ) {
      val = dist( mersenne_gen );
   } 
}


template <class RealType, size_t size>
void gen_reals( std::array<RealType, size>& vals, const RealType& lower, const RealType& upper ) {
   std::random_device random_dev;
   std::mt19937 mersenne_gen(random_dev());
   std::uniform_real_distribution<RealType> dist(lower, upper);
   for ( auto& val: vals ) {
      val = dist( mersenne_gen );
   } 
}


template <class RealType, size_t size>
void gen_reals( std::array<RealType, size>& vals, const RealType& lower, const RealType& upper, const int& seed ) {
   std::mt19937 mersenne_gen(seed);
   std::uniform_real_distribution<RealType> dist(lower, upper);
   for ( auto& val: vals ) {
      val = dist( mersenne_gen );
   } 
}


template <class IntType>
void gen_ints( std::vector<IntType>& vals, const IntType& lower, const IntType& upper ) {
   std::default_random_engine generator;
   std::uniform_int_distribution<IntType> dist( lower, upper );
   for ( auto& val: vals ) {
      val = dist( generator );
   }
}

template <class IntType>
void gen_ints( std::vector<IntType>& vals, const IntType& lower, const IntType& upper, const int& seed ) {
   std::default_random_engine generator(seed);
   std::uniform_int_distribution<IntType> dist( lower, upper );
   for ( auto& val: vals ) {
      val = dist( generator );
   }
}

template <class IntType, size_t size>
void gen_ints( std::array<IntType, size>& vals, const IntType& lower, const IntType& upper ) {
   std::default_random_engine generator;
   std::uniform_int_distribution<IntType> dist( lower, upper );
   for ( auto& val: vals ) {
      val = dist( generator );
   }
}

template <class IntType, size_t size>
void gen_ints( std::array<IntType, size>& vals, const IntType& lower, const IntType& upper, const int& seed ) {
   std::default_random_engine generator(seed);
   std::uniform_int_distribution<IntType> dist( lower, upper );
   for ( auto& val: vals ) {
      val = dist( generator );
   }
}
