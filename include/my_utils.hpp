#pragma once

#define SUCCESS 0
#define FAILURE -2

#include <cstdio>
#include <iostream>
#include <iomanip>

#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <iterator>
#include <stdarg.h>
#include <string>
#include <vector>
#include <complex>
#include <algorithm>
#include <random>
#include <exception>
#include <memory>

// For dealing with functions that only return error codes
#ifndef check_status_throw
#   define check_status_throw(status_var, success_val, msg)                     \
      {                                                  \
         if (status_var != success_val) {                        \
            throw std::runtime_error{ std::string{__func__} + "(): " + ex.what() }                          \
         }                                               \
      }
#endif


// For dealing with functions that only return error codes
#ifndef try_func_throw
#   define try_func_throw(status_var, success_val, msg, func) \
      {                                \
         status = func;                \
         check_status(status_var, success_val, msg);    \
      }
#endif


#ifndef delete_array
#   define delete_array(ptr) \
      {                    \
         if (ptr) delete[] ptr;  \
      }
#endif


#ifndef SWAP
#   define SWAP(a, b) \
      {               \
         (a) ^= (b);  \
         (b) ^= (a);  \
         (a) ^= (b);  \
      }
#endif


#ifndef MAX
#   define MAX(a, b) ((a) > (b)) ? (a) : (b);
#endif

#ifndef MIN
#   define MAX(a, b) ((a) > (b)) ? (a) : (b);
#endif


#ifndef MIN
#   define MIN(a, b) ((a) < (b)) ? (a) : (b);
#endif


#ifndef DIVIDE_ROUND_UP
#   define DIVIDE_ROUND_UP(a, b) ((a) + ((b)-1)) / (b);
#endif


#ifndef dout
#  define dout debug && std::cout
#endif


#ifndef debug_print
#   define debug_printf(debug, fmt, ...) \
      {                                  \
         if (debug) {                    \
            printf(fmt, ##__VA_ARGS__);  \
         }                               \
      }
#endif


using llong = long long;
using ulong = unsigned long;
using uint = unsigned int;


template<typename T>
using complex_vector = std::vector<std::complex<T>>;


template<typename T, size_t size>
using complex_array = std::array<std::complex<T>, size>;

using Steady_Clock = std::chrono::steady_clock;
using Time_Point = std::chrono::time_point<std::chrono::steady_clock>;
using Duration = std::chrono::duration<float, std::milli>;
using Duration_ms = std::chrono::duration<float, std::milli>;
using Duration_us = std::chrono::duration<float, std::micro>;
using Duration_ns = std::chrono::duration<float, std::nano>;

// Example usage for previous aliases
// Time_Point start_point = Steady_Clock::now();
// 
// // Timed code goes here
// 
// Duration_ms duration_ms = Steady_Clock::now() - start_point;
// float seconds = duration_ms.count() * std::chrono::milliseconds::period::num / std::chrono::milliseconds::period::den;
// std::cout << __func__ << "() took " << duration_ms.count() << " milliseconds to process "
//    << num_vals << " values\n";
// std::cout << __func__ << "() That's an average of " << (num_vals/seconds) << " values per second\n\n"; 

// Just in case there is no intrinsic
// From Hacker's Delight
int my_popcount(unsigned int x);

int my_count_leading_zeros(unsigned int x);

int my_ilog2(unsigned int x);

int my_modulus( const unsigned int& val, const unsigned int& div );

inline bool is_divisible_by( const unsigned int& val, const unsigned int& div ) {
   return ( 0 == my_modulus( val, div ) );
}

inline bool is_power_of_two( const int& val ) {
   //return ( my_popcount(val) == 1 );  
   return ( __builtin_popcount(val) == 1 );
}

// Hacker's Delight Second Edition pg 291 ('ilog2')
inline int ilog2( const int& val ) {
   return ( 31 - __builtin_clz(val) );
}


// Already included in C++14
template<typename T, typename... Args>
std::unique_ptr<T> my_make_unique(Args&&... args) {
   return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}


inline bool string_is_palindrome(const std::string& str) {
   return std::equal(str.begin(), str.begin() + str.size()/2, str.rbegin());
}

template <typename T>
std::string to_string_with_precision(const T& a_value, const int places = 6) {

    std::ostringstream out;

    out.precision(places);
    
    out << std::fixed << a_value;
    return out.str();
}


// Erase/Remove Idiom
inline void strip_quotes( std::string& str ) {
   // move the non quote characters to the front
   // then erase the rest of the characters (the quotes)
   str.erase( std::remove( str.begin(), str.end(), '\"' ), str.end() );
}


// From stack overflow:
// https://stackoverflow.com/questions/14265581/parse-split-a-string-in-c-using-string-delimiter-standard-c
// for string delimiter
std::vector<std::string> split_string(std::string str, const std::string& delimiter);


// Example code, not actually meant to be directly used
inline std::string decode_status(int status) {
   char const* status_strings[] = {
      "This is status 0", 
      "This is status 1"
   };

   if (status < 2) {
      return std::string(status_strings[status]);
   }
   return std::string("Unknown status value: " + std::to_string(status) + "\n");
}


void free_these(void* arg1, ...);


// Hacker's Delight Second Edition pg 44 ('doz')
// Only valid for signed integers, -2^30 < a,b <=(2^30)-1
// or unsigned integers, 0 < a,b <= (2^31)-1
inline int difference_or_zero(int a, int b) { return ((a - b) & ~((a - b) >> 31)); }


