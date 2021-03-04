#pragma once

#include "my_utils.hpp"

// Only here for consistency sake, probably not needed
// Just use:
// std::equals( lvals.begin(), lvals.end(), rvals.begin() )
template <typename T>
bool compare_vals(int& mismatch_index, const std::vector<T>& lvals, const std::vector<T>& rvals) {
   for (int index = 0; index < (int)lvals.size(); ++index) {
      if (lvals[index] != rvals[index]) {
         std::cout << "Mismatch:\n";
         std::cout << "Lval[" << index << "] = " << std::setprecision(9) << lvals[index] << "\n";
         std::cout << "Rval[" << index << "] = " << std::setprecision(9) << rvals[index] << "\n";
         mismatch_index = index;
         return false;
      }
   }

   return true; 
}

template <typename T, size_t size>
bool compare_vals(int& mismatch_index, const std::array<T, size>& lvals, const std::array<T, size>& rvals) {
   for (int index = 0; index < (int)lvals.size(); ++index) {
      if (lvals[index] != rvals[index]) {
         std::cout << "Mismatch:\n";
         std::cout << "Lval[" << index << "] = " << std::setprecision(9) << lvals[index] << "\n";
         std::cout << "Rval[" << index << "] = " << std::setprecision(9) << rvals[index] << "\n";
         mismatch_index = index;
         return false;
      }
   }

   return true; 
}


template <typename T>
bool compare_vals(int& mismatch_index, const T* lvals, const T* rvals, int num_vals) {
   for (int index = 0; index < num_vals; ++index) {
      if (lvals[index] != rvals[index]) {
         std::cout << "Mismatch:\n";
         std::cout << "Lval[" << index << "] = " << std::setprecision(9) << lvals[index] << "\n";
         std::cout << "Rval[" << index << "] = " << std::setprecision(9) << rvals[index] << "\n";
         mismatch_index = index;
         return false;
      }
   }
   return true;
}


template<typename T>
bool vals_are_close( int& mismatch_index, const std::vector<T>& lvals, const std::vector<T>& rvals, const T& max_diff ) {

   for( size_t index = 0; index != lvals.size(); ++index ) {
      T abs_diff = abs( lvals[index] - rvals[index] );

      if ( ( abs_diff > max_diff ) ) {
         std::cout << "Mismatch:\n";
         std::cout << "Lval[" << index << "] = " << std::setprecision(9) << lvals[index] << "\n";
         std::cout << "Rval[" << index << "] = " << std::setprecision(9) << rvals[index] << "\n";
         std::cout << "Difference = " << abs_diff << "\n";
         std::cout << "Max Difference = " << max_diff << "\n";
         mismatch_index = index;
         return false;
      }
   }
   return true;
}


template<typename T, size_t size>
bool vals_are_close( int& mismatch_index, const std::array<T,size>& lvals, const std::array<T,size>& rvals, const T& max_diff ) {

   for( size_t index = 0; index != lvals.size(); ++index ) {
      T abs_diff = abs( lvals[index] - rvals[index] );

      if ( ( abs_diff > max_diff ) ) {
         std::cout << "Mismatch:\n";
         std::cout << "Lval[" << index << "] = " << std::setprecision(9) << lvals[index] << "\n";
         std::cout << "Rval[" << index << "] = " << std::setprecision(9) << rvals[index] << "\n";
         std::cout << "Difference = " << abs_diff << "\n";
         std::cout << "Max Difference = " << max_diff << "\n";
         mismatch_index = index;
         return false;
      }
   }
   return true;
}


template<typename T>
bool vals_are_close( int& mismatch_index, const T* lvals, const T* rvals, const int num_vals, const T& max_diff ) {

   for( int index = 0; index < num_vals; ++index ) {
      T abs_diff = abs( lvals[index] - rvals[index] );

      if ( ( abs_diff > max_diff ) ) {
         std::cout << "Mismatch:\n";
         std::cout << "Lval[" << index << "] = " << std::setprecision(9) << lvals[index] << "\n";
         std::cout << "Rval[" << index << "] = " << std::setprecision(9) << rvals[index] << "\n";
         std::cout << "Difference = " << abs_diff << "\n";
         std::cout << "Max Difference = " << max_diff << "\n";
         mismatch_index = index;
         return false;
      }
   }
   return true;
}


template<typename T>
bool vals_are_close( int& mismatch_index, const complex_vector<T>& lvals, const complex_vector<T>& rvals, const T& max_diff ) {

   for( size_t index = 0; index != lvals.size(); ++index ) {
      T abs_diff_real = abs( lvals[index].real() - rvals[index].real() );
      T abs_diff_imag = abs( lvals[index].imag() - rvals[index].imag() );

      if ( ( abs_diff_real > max_diff ) || ( abs_diff_imag > max_diff ) ) {
         std::cout << "Mismatch:\n";
         std::cout << "Lval[" << index << "] = {" << lvals[index] << "}\n";
         std::cout << "Rval[" << index << "] = {" << rvals[index] << "}\n";
         std::cout << "Difference = {" << abs_diff_real << ", " << abs_diff_imag << "}\n";
         std::cout << "Max Difference = " << max_diff << "\n";
         mismatch_index = index;
         return false;
      }
   }
   return true;
}


