#pragma once

// Managed Allocator Class
// Allows use of STL clases (like std::vector) with cudaMallocManaged() and cudaFree()
// From Jared Hoberock, NVIDIA:
// https://github.com/jaredhoberock/managed_allocator/blob/master/managed_allocator.hpp

template<class T>
class managed_allocator_host {
  public:
    using value_type = T;
    using reference = T&;
    using const_reference = const T&;

    // Make sure that only 1 allocation is done
    // per instance of this class
    bool memory_is_allocated;
    managed_allocator_host():
      memory_is_allocated( false ) {}

    template<class U>
    managed_allocator_host(const managed_allocator_host<U>&):
      memory_is_allocated( false ) {}
  
    value_type* allocate(size_t n) {
      try {
         value_type* result = nullptr;
         if ( !memory_is_allocated ) {
     
            cudaError_t error = cudaMallocManaged(&result, n*sizeof(T), cudaMemAttachHost);
        
            if(error != cudaSuccess) {
              throw std::runtime_error("managed_allocator_host::allocate(): cudaMallocManaged( cudaMemAttachHost )");
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
bool operator==(const managed_allocator_host<T1>&, const managed_allocator_host<T2>&) {
  return true;
}

template<class T1, class T2>
bool operator!=(const managed_allocator_host<T1>& lhs, const managed_allocator_host<T2>& rhs) {
  return !(lhs == rhs);
}

template<class T>
using managed_vector_host = std::vector<T, managed_allocator_host<T>>;

