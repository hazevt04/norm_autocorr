#include "my_args.hpp"

void print_my_args( const my_args_t& my_args ) {
   std::cout << "Args:\n"; 
   std::cout << "\tTest Select String is '" << my_args.test_select_string << "'\n";
   std::cout << "\tFilename is '" << my_args.filename << "'\n";
   std::cout << "\tExpected Norms Filename is '" << my_args.exp_norms_filename << "'\n";
   std::cout << "\tDelay is " << my_args.delay << "\n";
   std::cout << "\tMag Sqrs Window Size is " << my_args.mag_sqrs_window_size << "\n"; 
   std::cout << "\tConj Sqrs Window Size is " << my_args.conj_sqrs_window_size << "\n";
   std::cout << "\tNum Samples is " << my_args.num_samples << "\n";
   std::cout << "\tMax Num Iters is " << my_args.max_num_iters << "\n";
   std::cout << "\tDebug is " << ( my_args.debug ? "true" : "false" ) << "\n"; 
   std::cout << "\tHelp Showed is " << ( my_args.help_showed ? "true" : "false" ) << "\n\n"; 
}
