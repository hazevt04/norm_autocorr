#pragma once

#include "my_utils.hpp"

#include <fstream>
#include <sstream>

/// My Utility Functions for Text and Binary File I/O

typedef long long llong;

void check_num_file_bytes(llong& file_size, const char* filename, const bool debug);

template<typename T>
void write_binary_file_inner(
   T* vals, const char* filename, const int num_vals, const bool debug = false) {

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
            std::string{"Unable to open file, "} + filename + std::string{", for writing."}};
      }
      ofile.close();

      if ((ofile.rdstate() & std::ofstream::failbit) != 0) {
         throw std::runtime_error{
            std::string{"Logical error while writing file, "} + filename + std::string{"."}};
      }
      if ((ofile.rdstate() & std::ofstream::badbit) != 0) {
         throw std::runtime_error{
            std::string{"Write error while writing file, "} + filename + std::string{"."}};
      }
   } catch (std::exception& ex) {
      throw std::runtime_error{std::string{__func__} + std::string{"(): "} + ex.what()};
   }
} // end of write_binary_floats_file()


template<typename T>
void write_binary_file(
   std::vector<T>& vals, const char* filename, const bool debug = false) {
      
   try {
      write_binary_file_inner(vals.data(), filename, (int)vals.size(), debug);
   } catch (std::exception& ex) {
      throw std::runtime_error{std::string{__func__} + std::string{"(): "} + ex.what()};
   }
   
}


template<typename T>
void write_binary_file(
   T* vals, const char* filename, const int num_vals, const bool debug = false) {
      
   try {
      write_binary_file_inner(vals, filename, num_vals, debug);
   } catch (std::exception& ex) {
      throw std::runtime_error{std::string{__func__} + std::string{"(): "} + ex.what()};
   }
   
}

template<typename T>
void read_text_file_inner(
   T* vals, const char* filename, const int num_vals, const bool debug = false) {

   try {
      std::ifstream ifile;
      std::streampos num_file_bytes;
      dout << __func__ << "(): Text file to be read is " << filename << "\n";
      ifile.open(filename, std::ios::in );
      if (ifile.is_open()) {
         ifile.seekg(0, ifile.end);
         llong num_file_bytes = (llong)ifile.tellg();
         ifile.seekg(0, ifile.beg);
         dout <<  __func__ << "(): File size is " << (llong)num_file_bytes << " bytes\n\n";

         int item_num = 0;
         while( ifile && (item_num < num_vals) ) {
            std::string input_str;
            T t_val;
            ifile >> t_val;
            dout << __func__ << "(): Read item " << item_num << ": "<< t_val << " from file\n";
            vals[item_num] = t_val;
            ++item_num;
         }

      } else {
         throw std::runtime_error{
            std::string{"Unable to open file: "} + filename + std::string{"."}};
      } // end of if ( ifile.is_open() ) {
   } catch (std::exception& ex) {
      throw std::runtime_error{std::string{__func__} + std::string{"(): "} + ex.what()};
   }

}


template<typename T>
void read_text_file(
   std::vector<T>& vals, const char* filename, const bool debug = false) {

   try {
      read_text_file_inner<T>( vals.data(), filename, (int)vals.size(), debug );
   } catch (std::exception& ex) {
      throw std::runtime_error{std::string{__func__} + std::string{"(): "} + ex.what()};
   }
}

template<typename T>
void read_text_file(
   std::vector<T>& vals, const char* filename, const int num_vals_to_read, const bool debug = false) {

   try {
      read_text_file_inner<T>( vals.data(), filename, num_vals_to_read, debug );
   } catch (std::exception& ex) {
      throw std::runtime_error{std::string{__func__} + std::string{"(): "} + ex.what()};
   }
}

template<typename T>
void read_text_file(
   T* vals, const char* filename, const int num_vals, const bool debug = false) {

   try {
      read_text_file_inner<T>( vals, filename, num_vals, debug );
   } catch (std::exception& ex) {
      throw std::runtime_error{std::string{__func__} + std::string{"(): "} + ex.what()};
   }
}


template<typename T>
void read_binary_file_inner(
   T* vals, const char* filename, const int num_vals, const bool debug = false) {

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
         dout <<  __func__ << "(): File size is " << (llong)num_file_bytes << " bytes\n\n";
         if (num_file_bytes < num_val_bytes) {
            throw std::runtime_error{std::string{"Expected file size, "} +
               std::to_string(num_file_bytes) + std::string{" bytes, less than expected: "} +
               std::to_string(num_val_bytes) + std::string{" bytes, for file "} + filename +
               std::string{"."}};
         }
         ifile.read(reinterpret_cast<char*>(vals), num_val_bytes);

      } else {
         throw std::runtime_error{
            std::string{"Unable to open file: "} + filename + std::string{"."}};
      } // end of if ( ifile.is_open() ) {
   } catch (std::exception& ex) {
      throw std::runtime_error{std::string{__func__} + std::string{"(): "} + ex.what()};
   }
}

template<typename T>
void read_binary_file(
   std::vector<T>& vals, const char* filename, const bool debug = false) {

   try {
      read_binary_file_inner<T>( vals.data(), filename, (int)vals.size(), debug );
   } catch (std::exception& ex) {
      throw std::runtime_error{std::string{__func__} + std::string{"(): "} + ex.what()};
   }
}

template<typename T>
void read_binary_file(
   std::vector<T>& vals, const char* filename, const int num_vals_to_read, const bool debug = false) {

   try {
      read_binary_file_inner<T>( vals.data(), filename, num_vals_to_read, debug );
   } catch (std::exception& ex) {
      throw std::runtime_error{std::string{__func__} + std::string{"(): "} + ex.what()};
   }
}

template<typename T>
void read_binary_file(
   T* vals, const char* filename, const int num_vals, const bool debug = false) {

   try {
      read_binary_file_inner<T>( vals, filename, num_vals, debug );
   } catch (std::exception& ex) {
      throw std::runtime_error{std::string{__func__} + std::string{"(): "} + ex.what()};
   }
}


void test_my_file_io_funcs(
   std::string filename, const int num_vals, const bool inject_error, const bool debug);
