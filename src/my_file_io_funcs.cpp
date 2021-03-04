// C++ File for my_file_io_funcs

#include "my_file_io_funcs.hpp"

#include "my_generators.hpp"
#include "my_comparators.hpp"
#include "my_printers.hpp"


void check_num_file_bytes(llong& num_file_bytes, const char* filename, const bool& debug = false) {

   try {
      std::ifstream ifile;
      ifile.open(filename, std::ios::in | std::ios::binary);
      if (ifile.is_open()) {
         // get length of file:
         ifile.seekg(0, ifile.end);
         num_file_bytes = (llong)ifile.tellg();
         ifile.seekg(0, ifile.beg);
         dout << __func__ << "(): File size for " << filename << " is " << num_file_bytes << " bytes\n\n";
      } else {
         throw std::runtime_error{std::string{"Unable to open file, "} + filename +
            std::string{", for checking filesize."}};
      }
   } catch (std::exception& ex) {
      throw std::runtime_error{std::string{__func__} + std::string{"(): "} + ex.what()};
   } // end of catch

} // end of void check_num_file_bytes()


void test_my_file_io_funcs(
   std::string& filename, const int& num_vals, const bool& inject_error, const bool& debug) {

   try {
      llong num_file_bytes = 0;

      std::vector<float> write_vals(num_vals);
      std::vector<float> read_vals(num_vals);

      gen_reals<float>(write_vals, 0, num_vals);
      print_vals<float>(write_vals, "Write Vals:\n");

      dout << "The input text file is " << filename << "\n";
         
      write_binary_file(write_vals, filename.c_str(), debug);

      check_num_file_bytes(num_file_bytes, filename.c_str(), debug);

      if (inject_error) {
         filename = "wrong_file.bin";
      }

      read_binary_file(read_vals, filename.c_str(), debug);

      print_vals<float>(read_vals, "Read Vals:\n");

      int mismatch_index = -1;
      if ( !compare_vals<float>(mismatch_index, read_vals, write_vals) ) { 
         throw std::runtime_error{std::string{"Values read from "} + filename +
            std::string{" don't match values written."}};
      } else {
         std::cout << "All " << num_vals << " values read from " << filename
                   << " matched the values written\n";
      }

   } catch (std::exception& ex) {
      throw std::runtime_error{std::string{__func__} + std::string{"(): "} + ex.what()};
   }
}


// end of C++ file for my_file_io_funcs
