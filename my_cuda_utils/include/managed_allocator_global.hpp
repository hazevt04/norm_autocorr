#pragma once

#include <cuda_runtime.h>

#include <exception>
#include <iostream>
#include <stdexcept>
#include <vector>

// Managed Allocator Class
// Allows use of STL clases (like std::vector) with cudaMallocManaged() and cudaFree()
// From Jared Hoberock, NVIDIA:
// https://github.com/jaredhoberock/managed_allocator/blob/master/managed_allocator.hpp

template<class T>
class managed_allocator_global {
  public:
    using value_type = T;
    using reference = T&;
    using const_reference = const T&;

    // Make sure that only 1 allocation is done
    // per instance of this class
    bool memory_is_allocated;
    managed_allocator_global():
      memory_is_allocated( false ) {}

    template<class U>
    managed_allocator_global(const managed_allocator_global<U>&):
      memory_is_allocated( false ) {}
  
    value_type* allocate(size_t n) {
      try {
         value_type* result = nullptr;
         if ( !memory_is_allocated ) {
     
            cudaError_t error = cudaMallocManaged(&result, n*sizeof(T), cudaMemAttachGlobal);
        
            if(error != cudaSuccess) {
              throw std::runtime_error("managed_allocator_global::allocate(): cudaMallocManaged( cudaMemAttachGlobal )");
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
bool operator==(const managed_allocator_global<T1>&, const managed_allocator_global<T2>&) {
  return true;
}

template<class T1, class T2>
bool operator!=(const managed_allocator_global<T1>& lhs, const managed_allocator_global<T2>& rhs) {
  return !(lhs == rhs);
}

template<class T>
using managed_vector_global = std::vector<T, managed_allocator_global<T>>;


