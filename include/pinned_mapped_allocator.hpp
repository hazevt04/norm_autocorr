#pragma once

// Pinned Mapped Allocator Class
// Allows use of STL clases (like std::vector) with cudaMalloc() and cudaFree()
// (like thrust's device_vector)
// Based on Jared Hoberock, NVIDIA:
// https://github.com/jaredhoberock/managed_allocator/blob/master/managed_allocator.hpp

template<class T>
class pinned_mapped_allocator {
  public:
    using value_type = T;
    using reference = T&;
    using const_reference = const T&;

    // Make sure that only 1 allocation is done
    // per instance of this class
    bool memory_is_allocated;
    pinned_mapped_allocator():
      memory_is_allocated( false ) {}

    template<class U>
    pinned_mapped_allocator(const pinned_mapped_allocator<U>&):
      memory_is_allocated( false ) {}
  
    value_type* allocate(size_t n) {
      try {
         value_type* result = nullptr;
         if ( !memory_is_allocated ) {
     
            cudaError_t error = cudaHostAlloc(&result, n*sizeof(T), cudaHostAllocMapped);
        
            if(error != cudaSuccess) {
              throw std::runtime_error("pinned_mapped_allocator::allocate(): cudaHostAlloc(..., cudaHostAllocMapped)");
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
         cudaFreeHost( ptr );
         ptr = nullptr;
       }
    } 
};

template<class T1, class T2>
bool operator==(const pinned_mapped_allocator<T1>&, const pinned_mapped_allocator<T2>&) {
  return true;
}

template<class T1, class T2>
bool operator!=(const pinned_mapped_allocator<T1>& lhs, const pinned_mapped_allocator<T2>& rhs) {
  return !(lhs == rhs);
}

template<class T>
using pinned_mapped_vector = std::vector<T, pinned_mapped_allocator<T>>;

