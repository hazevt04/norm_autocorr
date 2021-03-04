#pragma once

#include <sys/types.h>
#include <sys/stat.h>
#include <errno.h>

bool mkpath( std::string path ) {
   bool success = false;
   int mkdir_result = mkdir( path.c_str(), 0775 );
   if ( mkdir_result == -1 ) {
      switch( errno ) {
         case ENOENT:
            //parent didn't exist, try to create it
            if ( mkpath( path.substr(0, path.find_last_of('/')) ) ) {
               //Now, try to create again.
               success = ( 0 == mkdir( path.c_str(), 0775 ) );
            } else {
               success = false;
            }
            break;
         case EEXIST:
            //Done!
            success = true;
            break;
         default:
            success = false;
            break;
      } // end of switch
      
   } else {
      success = true;
   }
   return success;
}
