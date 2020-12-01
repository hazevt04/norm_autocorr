#include "my_cuda_utils.hpp"

cudaError_t checkAndPrint(const char* name, int sync) {
    cudaError_t cerror = cudaGetLastError();
    if (cerror != cudaSuccess) {
        const char * errorMessage = cudaGetErrorString(cerror);
        fprintf(stderr, "CUDA error check \"%s\" returned ERROR code: %d (%s) %s \n", name, cerror, errorMessage, (sync) ? "after sync" : "");
    }
    return cerror;
}
 
cudaError_t checkCUDAError(const char * name, int sync) {
    cudaError_t cerror = cudaSuccess;
    if (sync) {
        cerror = checkAndPrint(name, 0);
        cudaDeviceSynchronize();
        cerror = checkAndPrint(name, 1);
    } else {
        cerror = checkAndPrint(name, 0);
    }
    return cerror;
}
