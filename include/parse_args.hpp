#pragma once

#include <getopt.h>

#include "my_args.hpp"

static void print_usage( char* prog_name ) {
   std::cout << "Usage: " << prog_name << ":\n";
   std::cout << "-s/--select <selection>     Select the mode for\n";
   std::cout << "                            test samples input. One of\n";
   std::cout << "                            Sinousoidal, Random, or Filebased\n";
   std::cout << "-f/--filename <name>        Name of input file if Filebased select\n";
   std::cout << "\n"; 
}

void parse_args( my_args_t& my_args, int argc, char** argv ) {
   try {
      const char* const short_options = "s:f:dh";
      const option long_options[] = {
         {"select", required_argument, nullptr, 's'},
         {"filename", optional_argument, nullptr, 'f'},
         {"debug", no_argument, nullptr, 0}
      };

      my_args.filename = "input_samples.5.9GHz.10MHzBW.560u.LS.dat";
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

            case 'd':
               my_args.debug = true;
               break;

            case 'h':
               my_args.help_showed = true;
               print_usage(argv[0]);
               break;

            case '?':
            default:
               print_usage(argv[0]);
               break;
         } // end of switch (c) {
      } // end of while
   } catch( std::exception& ex ) {
      throw std::runtime_error(
         std::string{__func__} + std::string{"(): Error: "} +
         ex.what()
      );
   }
}
