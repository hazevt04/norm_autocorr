#pragma once

#include "my_args.hpp"

#include <getopt.h>

static void print_usage( char* prog_name ) {
   std::cout << "Usage: " << prog_name << ":\n";
   std::cout << "-s/--select <selection>     Select the mode for\n";
   std::cout << "                            test samples input. One of\n";
   std::cout << "                            Sinousoidal, Random, or Filebased\n";
   std::cout << "-f/--filename <name>        Name of input file if Filebased select\n";
   std::cout << "-e/--efilename <name>       Name of expected output file if Filebased select\n";
   std::cout << "-n/--num_samples <name>     Number of samples\n";
   std::cout << "\n"; 
}

void parse_args( my_args_t& my_args, int argc, char** argv ) {
   try {
      const char* const short_options = "s:f:e:n:dh";
      const option long_options[] = {
         {"select", required_argument, nullptr, 's'},
         {"filename", optional_argument, nullptr, 'f'},
         {"efilename", optional_argument, nullptr, 'e'},
         {"num_samples", optional_argument, nullptr, 'n'},
         {"debug", no_argument, nullptr, 'd'},
         {"help", no_argument, nullptr, 'h'}
      };
      
      my_args.test_select_string = "Filebased";
      my_args.filename = "input_samples.5.180GHz.20MHzBW.560u.LS.dat";
      my_args.exp_norms_filename = "exp_norms.5.180GHz.20MHzBW.560u.LS.dat";
      my_args.num_samples = 128000;
      bool bad_arg = false;
      while (true) {

         const auto opt = getopt_long( argc, argv, short_options,
            long_options, nullptr );

         if (-1 == opt) break;
         
         switch (opt) {
            case 's':
               my_args.test_select_string = optarg;
               break;

            case 'f':
               my_args.filename = optarg;
               break;

            case 'e':
               my_args.exp_norms_filename = optarg;
               break;
            
            case 'n':
               my_args.num_samples = std::atoi(optarg);
               break;

            case 'd':
               my_args.debug = true;
               break;

            case 'h':
               my_args.help_showed = true;
               print_usage(argv[0]);
               break;

            case '?':
            default:
               bad_arg = true;
               print_usage(argv[0]);
               break;
         } // end of switch (c) {
         if ( bad_arg ) {
            throw std::runtime_error{std::string{"Invalid argument: '"} + optarg + std::string{"'"}};
         }
      } // end of while
      
      print_my_args(my_args);

   } catch( std::exception& ex ) {
      throw std::runtime_error(
         std::string{__func__} + std::string{"(): Error: "} +
         ex.what()
      );
   }
}
