#pragma once

#include <complex>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string.h>

#include "my_utils.hpp"


void check_num_file_bytes(llong& file_size, const char* filename, const bool& debug);


template<typename T>
void write_binary_file_inner(
   T* vals, const char* filename, const int& num_vals, const bool& debug = false) {

   try {
      std::ofstream ofile;
      std::streampos num_file_bytes;
      ofile.open(filename, std::ios::out | std::ios::binary);
      if (ofile.is_open()) {
         std::streamsize num_val_bytes = num_vals * sizeof(T);
         dout << __func__ << "(): Val size is " << num_val_bytes << " bytes\n\n"; 
         ofile.write(reinterpret_cast<char*>(vals), num_val_bytes);
      } else {
         throw std::runtime_error{
            "Unable to open file, " + std::string{filename} + ", for writing."
         };
      }
      ofile.close();

      if ((ofile.rdstate() & std::ofstream::failbit) != 0) {
         throw std::runtime_error{
            "Logical error while writing file, " + std::string{filename}
         };
      }
      if ((ofile.rdstate() & std::ofstream::badbit) != 0) {
         throw std::runtime_error{
            "Write error while writing file, " + std::string{filename}
         };
      }
   } catch (std::exception& ex) {
      throw std::runtime_error{ std::string{__func__} + "(): " + ex.what() };
   }
} // end of write_binary_floats_file()


template<typename T>
void write_binary_file(
   std::vector<T>& vals, const char* filename, const bool& debug = false) {
      
   try {
      write_binary_file_inner(vals.data(), filename, (int)vals.size(), debug);
   } catch (std::exception& ex) {
      throw std::runtime_error{ std::string{__func__} + "(): " + ex.what() };
   }
   
}

template<typename T, size_t num_vals>
void write_binary_file(
   std::array<T, num_vals>& vals, const char* filename, const bool& debug = false) {
      
   try {
      write_binary_file_inner(vals.data(), filename, (int)num_vals, debug);
   } catch (std::exception& ex) {
      throw std::runtime_error{ std::string{__func__} + "(): " + ex.what() };
   }
   
}

template<typename T>
void write_binary_file(
   T* vals, const char* filename, const int& num_vals, const bool& debug = false) {
      
   try {
      write_binary_file_inner(vals, filename, num_vals, debug);
   } catch (std::exception& ex) {
      throw std::runtime_error{ std::string{__func__} + "(): " + ex.what() };
   }
   
}


template<typename T>
void write_text_file_inner(
   T* vals, 
   const char* filename, 
   const int& start_index, 
   const int& num_vals, 
   const std::string& prefix,
   const std::string& line_prefix,
   const std::string& line_suffix,
   const std::string& suffix,
   const bool& debug = false
) {

   try {
      std::ofstream ofile;
      ofile.open(filename, std::ios::out);
      if (ofile.is_open()) {
         dout << __func__ << "(): Num vals is " << num_vals << " vals\n\n";

         ofile << prefix;
         int max_index = start_index + num_vals;
         for( int index = start_index; index < max_index; ++index ) {
            ofile << line_prefix << vals[index] << line_suffix;
         }
         ofile << suffix;
      }
   } catch (std::exception& ex) {
      throw std::runtime_error{ std::string{__func__} + "(): " + ex.what() };
   }
} // end of write_text_fike_inner()


template<typename T>
void write_text_file(
   std::vector<T>& vals, 
   const char* filename, 
   const std::string& prefix,
   const std::string& line_prefix,
   const std::string& line_suffix,
   const std::string& suffix,
   const bool& debug = false
) {
      
   try {
      write_text_file_inner(vals.data(), filename, 0, (int)vals.size(), 
         prefix, line_prefix, line_suffix, suffix, debug);
   } catch (std::exception& ex) {
      throw std::runtime_error{ std::string{__func__} + "(): " + ex.what() };
   }
   
}


template<typename T>
void write_text_file(
   std::vector<T>& vals,
   const char* filename,
   const int& start_index,
   const int& num_vals,
   const std::string& prefix,
   const std::string& line_prefix,
   const std::string& line_suffix,
   const std::string& suffix,
   const bool& debug = false
) {
      
   try {
      write_binary_file_inner(vals.data(), filename, start_index, num_vals, 
         prefix, line_prefix, line_suffix, suffix, debug);
   } catch (std::exception& ex) {
      throw std::runtime_error{ std::string{__func__} + "(): " + ex.what() };
   }
   
}


template<typename T>
void write_text_file(
   T* vals, 
   const char* filename, 
   const int& num_vals, 
   const std::string& prefix,
   const std::string& line_prefix,
   const std::string& line_suffix,
   const std::string& suffix,
   const bool& debug = false
) {
      
   try {
      write_binary_file_inner(vals, filename, 0, num_vals,
         prefix, line_prefix, line_suffix, suffix, debug);
   } catch (std::exception& ex) {
      throw std::runtime_error{ std::string{__func__} + "(): " + ex.what() };
   }
   
}


template<typename T>
void read_binary_file_inner(
   T* vals, const char* filename, const int& num_vals, const bool& debug = false) {

   try {
      std::ifstream ifile;
      std::streampos num_file_bytes;
      ifile.open(filename, std::ios::in | std::ios::binary);
      if (ifile.is_open()) {
         size_t num_val_bytes = num_vals * sizeof(T);
         dout << __func__ << "(): Val size is " << num_val_bytes << " bytes\n"; 
         ifile.seekg(0, ifile.end);
         llong num_file_bytes = (llong)ifile.tellg();
         ifile.seekg(0, ifile.beg);
         dout << __func__ << "(): File size is " << (llong)num_file_bytes << " bytes\n"; 
         if (num_file_bytes < num_val_bytes) {
            throw std::runtime_error{ "Expected file size, " +
               std::to_string(num_file_bytes) + " bytes, less than expected: " +
               std::to_string(num_val_bytes) + " bytes, for file " + std::string{filename} 
            };
         }
         ifile.read(reinterpret_cast<char*>(vals), num_val_bytes);

      } else {
         throw std::runtime_error{
            "Unable to open file: " + std::string{filename}
         };
      } // end of if ( ifile.is_open() ) {
   } catch (std::exception& ex) {
      throw std::runtime_error{std::string{__func__} + "(): " + ex.what()};
   }
}


template<typename T>
void read_binary_file(
   std::vector<T>& vals, const char* filename, const bool& debug = false) {

   try {
      read_binary_file_inner<T>( vals.data(), filename, (int)vals.size(), debug );
   } catch (std::exception& ex) {
      throw std::runtime_error{std::string{__func__} + "(): " + ex.what()};
   }
}


template<typename T>
void read_binary_file(
   std::vector<T>& vals, const char* filename, const int& num_vals_to_read, const bool& debug = false) {

   try {
      read_binary_file_inner<T>( vals.data(), filename, num_vals_to_read, debug );
   } catch (std::exception& ex) {
      throw std::runtime_error{std::string{__func__} + "(): " + ex.what()};
   }
}


template<typename T>
void read_binary_file(
   T* vals, const char* filename, const int& num_vals, const bool& debug = false) {

   try {
      read_binary_file_inner<T>( vals, filename, num_vals, debug );
   } catch (std::exception& ex) {
      throw std::runtime_error{std::string{__func__} + "(): " + ex.what()};
   }
}

template<typename T>
void read_text_file_inner(
   T* vals, const char* filename, const int& start_index, const int& num_vals, const bool& debug = false) {

   try {
      std::ifstream ifile;
      std::streampos num_file_bytes;
      ifile.open(filename, std::ios::in);
      if (ifile.is_open()) {
         int max_index = start_index + num_vals;
         for( int index = start_index; index < max_index; ++index ) {
            ifile >> vals[index];
         } 
      } else {
         throw std::runtime_error{
            "Unable to open file: " + std::string{filename}
         };
      } // end of if ( ifile.is_open() ) {
   } catch (std::exception& ex) {
      throw std::runtime_error{std::string{__func__} + "(): " + ex.what()};
   }
}


template<typename T>
void read_text_file(
   std::vector<T>& vals, const char* filename, const bool& debug = false) {

   try {
      read_text_file_inner<T>( vals.data(), filename, 0, (int)vals.size(), debug );
   } catch (std::exception& ex) {
      throw std::runtime_error{std::string{__func__} + "(): " + ex.what()};
   }
}


template<typename T>
void read_text_file(
   std::vector<T>& vals, const char* filename, const int& num_vals_to_read, const bool& debug = false) {

   try {
      read_text_file_inner<T>( vals.data(), filename, 0, num_vals_to_read, debug );
   } catch (std::exception& ex) {
      throw std::runtime_error{std::string{__func__} + "(): " + ex.what()};
   }
}


template<typename T>
void read_text_file(
   std::vector<T>& vals, const char* filename, const int& start_index, const int& num_vals_to_read, const bool& debug = false) {

   try {
      read_text_file_inner<T>( vals.data(), filename, start_index, num_vals_to_read, debug );
   } catch (std::exception& ex) {
      throw std::runtime_error{std::string{__func__} + "(): " + ex.what()};
   }
}


template<typename T>
void read_text_file(
   T* vals, const char* filename, const int& num_vals, const bool& debug = false) {

   try {
      read_text_file_inner<T>( vals, filename, 0, num_vals, debug );
   } catch (std::exception& ex) {
      throw std::runtime_error{std::string{__func__} + "(): " + ex.what()};
   }
}


template<typename T>
void read_text_file(
   T* vals, const char* filename, const int& start_index, const int& num_vals, const bool& debug = false) {

   try {
      read_text_file_inner<T>( vals, filename, start_index, num_vals, debug );
   } catch (std::exception& ex) {
      throw std::runtime_error{std::string{__func__} + "(): " + ex.what()};
   }
}

void test_my_file_io_funcs(
   std::string& filename, const int& num_vals, const bool& inject_error, const bool& debug);
