// C++ File for parse_args

// C++ File for parse_args

#include "parse_args.hpp"

#include "my_utils.hpp"

#include <getopt.h>


void print_usage( char* prog_name ) {
   std::cout << "Usage: " << prog_name << ":\n";
   std::cout << "-m/--modeselect <selection> Select the mode for\n";
   std::cout << "                            test samples input. One of\n";
   std::cout << "                               Sinusoidal, Random, or Filebased\n";
   std::cout << "-t/--threadsperblock <num>  Number of threads per block (Must be power of 2)\n"; 
   std::cout << "-n/--nsamples <num>         Number of Samples (ignored for Filebased select)\n";
   std::cout << "-s/--seed <num>             Seed for the random number generator (only used if Random select)\n";
   std::cout << "-f/--filename <name>        Name of input file if Filebased select\n";
   std::cout << "\n"; 
}


void parse_args( my_args_t& my_args, int argc, char** argv ) {
   try {
      const char* const short_options = "m:t:n:s:f:dh";
      const option long_options[] = {
         {"mode", optional_argument, nullptr, 'm'},
         {"threadsperblock", optional_argument, nullptr, 't'},
         {"nsamples", optional_argument, nullptr, 'n'},
         {"seed", optional_argument, nullptr, 's'},
         {"filename", optional_argument, nullptr, 'f'},
         {"debug", no_argument, nullptr, 0}
      };

      while (true) {

         const auto opt = getopt_long( argc, argv, short_options,
            long_options, nullptr );

         char* end_ptr = nullptr;
         int t_threads_per_block;
         int t_num_samples;
         int t_seed;

         if (-1 == opt) break;
         
         switch (opt) {
            case 'm':
               my_args.mode_select = decode_mode_select_string( optarg );
               break;
            case 't':
               t_threads_per_block = strtol(optarg, &end_ptr, 10);
               if ( end_ptr == nullptr ) { 
                  throw std::runtime_error( std::string{__func__} + 
                     std::string{"(): Invalid string for threads per block entered: "} +
                     optarg + std::string{". Must be a number."} 
                  );
               } else if ( !is_power_of_two(t_threads_per_block) ) {
                  throw std::runtime_error( std::string{__func__} + 
                     std::string{"(): Invalid string for threads per block entered: "} +
                     optarg + std::string{". Must be a power of two"} 
                  );
               }
               my_args.threads_per_block = t_threads_per_block;
               break;
            case 'n':
               t_num_samples = strtol(optarg, &end_ptr, 10);
               if ( end_ptr == nullptr ) { 
                  throw std::runtime_error( std::string{__func__} + 
                     std::string{"(): Invalid string for number of samples entered: "} +
                     optarg + std::string{". Must be a number."} 
                  );
               }
               my_args.num_samples = t_num_samples;
               break;
            case 's':
               t_seed = strtol(optarg, &end_ptr, 10);
               if ( end_ptr == nullptr ) { 
                  throw std::runtime_error( std::string{__func__} + 
                     std::string{"(): Invalid string for seed entered: "} +
                     optarg + std::string{". Must be a number."} 
                  );
               }
               my_args.seed = t_seed;
               break;
            case 'f':
               my_args.filename = optarg;
               break;
            case 'd':
               my_args.debug = true;
               break;
            case 'h':
               print_usage(argv[0]);
               my_args.help_shown = true;
               break;
            case '?':
            default:
               print_usage(argv[0]);
               my_args.help_shown = true;
         } // end of switch (c) {
      } // end of while
   } catch( std::exception& ex ) {
      throw std::runtime_error(
         std::string{__func__} + std::string{"(): Error: "} +
         ex.what()
      );
   }
}

// end of C++ file for parse_args

