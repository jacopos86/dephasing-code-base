#include <pycuda-complex.hpp>
#include <cuComplex.h>
typedef pycuda::complex<double> cmplx;

__global__ void compute_T1_oneph_w_resolved() {
    /* internal variables */
    const int i = threadIdx.x + blockDim.x * blockIdx.x;
    const int j = threadIdx.y + blockDim.y * blockIdx.y;
    const int k = threadIdx.z + blockDim.z * blockIdx.z;
    const int idx = i + j * blockDim.x * gridDim.x + k * blockDim.x * gridDim.x * blockDim.y * gridDim.y;

    /* local variables */



}


__global__ void compute_T1_oneph_time_resolved(int n, int *INIT_INDEX, int *SIZE_LIST) {
    /* internal variables */
    const int i = threadIdx.x + blockDim.x * blockIdx.x;
    const int j = threadIdx.y + blockDim.y * blockIdx.y;
    const int k = threadIdx.z + blockDim.z * blockIdx.z;
    const int idx = i + j * blockDim.x * gridDim.x + k * blockDim.x * gridDim.x * blockDim.y * gridDim.y;

    /* local variables */
    const int sx = SIZE_LIST[idx];
    const int i0x = INIT_INDEX[idx];



}