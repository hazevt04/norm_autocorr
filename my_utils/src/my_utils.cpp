// C++ File for utils

#include "my_utils.hpp"

// Just in case there is no intrinsic
// From Hacker's Delight
int my_popcount(unsigned int x) {
   x -= ((x >> 1) & 0x55555555);
   x = (x & 0x33333333) + ((x >> 2) & 0x33333333);
   x = (x + (x >> 4)) & 0x0F0F0F0F;
   x += (x >> 8);
   x += (x >> 16);    
   return x & 0x0000003F;
}

// In case of no __builtin_clz() and no __builtin_popcount()
int my_count_leading_zeros(unsigned int x) {
   x = x | ( x >> 1 );
   x = x | ( x >> 2 );
   x = x | ( x >> 4 );
   x = x | ( x >> 8 );
   x = x | ( x >> 16 );
   return my_popcount(~x);
}
 
// In case of no __builtin_clz() and no __builtin_popcount()
int my_ilog2(unsigned int x) {
   x = x | ( x >> 1 );
   x = x | ( x >> 2 );
   x = x | ( x >> 4 );
   x = x | ( x >> 8 );
   x = x | ( x >> 16 );
   return (my_popcount(x) - 1);
}

int my_modulus( const unsigned int& val, const unsigned int& div ) {
   if ( div > 0 ) {
      // Only use mod (%) if we really HAVE to...
      if ( is_power_of_two( div ) ) {
         unsigned int mask = (1 << (ilog2( div ))) - 1;
         return ( val & mask );
      } else {
         return ( val % div );
      }
   } else {
      return 0;
   }
}

bool is_divisible_by( const unsigned int& val, const unsigned int& div ) {
   return ( 0 == my_modulus( val, div ) );
}


bool string_is_palindrome(const std::string& s) {
   return std::equal(s.begin(), s.begin() + s.size()/2, s.rbegin());
}

// Boost? Hurrumph!
// String splitter from SO:
// https://stackoverflow.com/questions/14265581/parse-split-a-string-in-c-using-string-delimiter-standard-c
void my_string_splitter( std::vector<std::string>& str_strings, std::string& str, const std::string delimiter = " ", const bool debug=false ) {
   size_t pos = 0;
   dout << __func__ << "(): str at start is '" << str << "'\n";
   while ((pos = str.find(delimiter)) != std::string::npos) {
      dout << __func__ << "(): token is '" <<  str.substr(0, pos) << "'\n";
      str_strings.push_back( str.substr(0, pos) );
      str.erase(0, pos + delimiter.length());
      dout << __func__ << "(): str in while loop is '" << str << "'\n";
   }
   dout << "\n";
   // Get the rest of the string if any
   if ( str != "" ) str_strings.push_back( str );
}

// variadic free function!
int free_these(void *arg1, ...) {
   va_list args;
   void *vp;
   if ( arg1 != NULL ) free(arg1);
   va_start(args, arg1);
   while ((vp = va_arg(args, void *)) != 0)
      if ( vp != NULL ) free(vp);
   va_end(args);
   return SUCCESS;
}





// end of C++ file for utils
