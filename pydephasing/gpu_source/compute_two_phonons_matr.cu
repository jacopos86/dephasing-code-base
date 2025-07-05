#include <pycuda-complex.hpp>
#include <math.h>
#include <cuComplex.h>
typedef pycuda::complex<double> cmplx;

/*
    compute g_qqp
*/

__global__ void compute_gqqp(int nat, int nl, int nst, int nmd, int *INIT_INDEX, int *SIZE_LIST, int *MODES_LIST, 
double *AQL, double *AQPL, double *WQL, double *WQPL, double *EIG, cmplx *FX, cmplx *EQ, cmplx *EQP, cmplx *EIQR,
cmplx *EIQPR, cmplx *GQQP) {
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
        if (ix == 850) {
            printf("%d      %d       %d      %d\n", IL, ILP, nl, nst);
        }
        for (int jax=0; jax<3*nat; jax++) {
            cmplx eq_1 = EQ[IL+jax*nmd];
            if (ix == 0 && jax == 10) {
                printf("%d        %d         %.10E        %.10E\n", IL, jax, real(eq_1), imag(eq_1));
            }
            for (int jby=0; jby<3*nat; jby++) {
                cmplx eq_2 = EQP[ILP+jby*nmd];
                for (int n1=0; n1<nl; n1++) {
                    for (int n2=0; n2<nl; n2++) {
                        for (int a=0; a<nst; a++) {
                            for (int b=0; b<nst; b++) {
                                size_t INDG = ILP + IL*nmd + b*nmd*nmd + a*nst*nmd*nmd;
                                GQQP[INDG] = (0, 0);
                            }
                        }
                    }
                }
            }
        }
    }
}