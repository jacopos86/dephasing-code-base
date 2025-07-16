#include <pycuda-complex.hpp>
#include <math.h>
#include <cuComplex.h>
typedef pycuda::complex<double> cmplx;

/*
    compute P_eph^{(1)} -> e-ph scattering first order
*/

__global__ void compute_P1_eph(int nm, int nst, cmplx *P) {
    /*internal variables */
    const int i = threadIdx.x + blockDim.x * blockIdx.x;
    const int j = threadIdx.y + blockDim.y * blockIdx.y;
    const int k = threadIdx.z + blockDim.z * blockIdx.z;
    const int idx = i + j * blockDim.x * gridDim.x + k * blockDim.x * gridDim.x * blockDim.y * gridDim.y;
    if (idx == 0) {
        printf("%d    -     %d\n", nm, nst);
    } 
}