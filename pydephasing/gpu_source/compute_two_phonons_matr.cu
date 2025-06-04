#include <pycuda-complex.hpp>
#include <math.h>
#include <cuComplex.h>
typedef pycuda::complex<double> cmplx;

/*
    compute g_qqp
*/

__global__ void compute_gqqp(int nat, int *INIT_INDEX, int *SIZE_LIST, int *MODES_LIST, cmplx *FX, 
cmplx *FXXP) {
    /* internal variables */
    const int i = threadIdx.x + blockDim.x * blockIdx.x;
    const int j = threadIdx.y + blockDim.y * blockIdx.y;
    const int k = threadIdx.z + blockDim.z * blockIdx.z;
    const int idx = i + j * blockDim.x * gridDim.x + k * blockDim.x * gridDim.x * blockDim.y * gridDim.y;
    const int sx = SIZE_LIST[idx];
    const int i0x = INIT_INDEX[idx];
    /* cycle over modes pairs */
    for (int ix=i0x; ix<i0x+sx; ix++) {
        /* modes pair indexes */
        int IL = MODES_LIST[2*ix];
        int ILP = MODES_LIST[2*ix+1];
        /* iterate over atomic indexes */
        for (int JX=0; JX<3*nat; JX++) {
            for (int JXP=0; JXP<3*nat; JXP++) {
                /* iterate over bands to compute FXXP */
            }
        }
    }
























}