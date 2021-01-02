// C++ File for my_args

#include "my_args.hpp"

#include <string>
#include <iostream>

mode_select_t decode_mode_select_string( std::string mode_select_string ) {
   if ( mode_select_string == "Sinusoidal" ) {
      return mode_select_t::Sinusoidal;
   } else if ( mode_select_string == "Increasing" ) {
      return mode_select_t::Increasing;
   } else if ( mode_select_string == "Random" ) {
      return mode_select_t::Random;
   } else if ( mode_select_string == "Filebased" ) {
      return mode_select_t::Filebased;
   } else {
      std::cout << "WARNING: Invalid mode select string: " << mode_select_string << "\n";
      std::cout << "Selecting mode_select_t::Sinusoidal\n"; 
      return mode_select_t::Sinusoidal;
   }
}

std::string get_mode_select_string( mode_select_t mode_select ) {
   if ( mode_select == mode_select_t::Sinusoidal ) {
      return "Sinusoidal";
   } else if ( mode_select == mode_select_t::Increasing ) {
      return "Increasing";
   } else if ( mode_select == mode_select_t::Random ) {
      return "Random";
   } else if ( mode_select == mode_select_t::Filebased ) {
      return "Filebased";
   } else {
      return "Unknown";
   }
} // end of std::string get_mode_select_string( mode_select_t mode_select ) const
