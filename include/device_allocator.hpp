#pragma once

// Device Allocator Class
// Allows use of STL clases (like std::vector) with cudaMalloc() and cudaFree()
// (like thrust's device_vector)
// Based on Jared Hoberock, NVIDIA:
// https://github.com/jaredhoberock/managed_allocator/blob/master/managed_allocator.hpp

template<class T>
class device_allocator {
  public:
    using value_type = T;
    using reference = T&;
    using const_reference = const T&;

    // Make sure that only 1 allocation is done
    // per instance of this class
    bool memory_is_allocated;
    device_allocator():
      memory_is_allocated( false ) {}

    template<class U>
    device_allocator(const device_allocator<U>&):
      memory_is_allocated( false ) {}
  
    value_type* allocate(size_t n) {
      try {
         value_type* result = nullptr;
         if ( !memory_is_allocated ) {
     
            cudaError_t error = cudaMalloc(&result, n*sizeof(T));
        
            if(error != cudaSuccess) {
              throw std::runtime_error("device_allocator::allocate(): cudaMalloc()");
            }
            memory_is_allocated = true;
         }
         return result;
      } catch ( std::exception& ex ) {
         std::cerr << __func__ << "(): ERROR: " << ex.what() << "\n";
         return nullptr;
      }
    }
    
    void deallocate(value_type* ptr, size_t size) {
       if ( ptr ) {
         cudaFree( ptr );
         ptr = nullptr;
       }
    } 
};

template<class T1, class T2>
bool operator==(const device_allocator<T1>&, const device_allocator<T2>&) {
  return true;
}

template<class T1, class T2>
bool operator!=(const device_allocator<T1>& lhs, const device_allocator<T2>& rhs) {
  return !(lhs == rhs);
}

template<class T>
using device_vector = std::vector<T, device_allocator<T>>;


template <class T>
void print_vec( const device_vector<T>& vals, const int num_vals, const char* prefix = "", const char* delim = "\n" ) {
   std::cout << prefix;
   for( int index = 0; index != num_vals; ++index ) {
      std::cout << vals[ index ] << delim;
   }
   std::cout << "\n";
}
