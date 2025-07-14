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
    for (int ix=i0x; ix<i0x+1; ix++) {
        /* modes pair indexes */
        int IL = MODES_LIST[2*ix];
        int ILP = MODES_LIST[2*ix+1];
        //if (ix == 850) {
        //    printf("%d      %d       %d      %d\n", IL, ILP, nl, nst);
        //}
        for (int a=0; a<nst; a++) {
            for (int ap=0; ap<nst; ap++) {
                size_t INDG = ILP+IL*nmd+ap*nmd*nmd+a*nst*nmd*nmd;
                GQQP[INDG] = (0, 0);
                for (int jax=0; jax<3*nat; jax++) {
                    cmplx eq_1 = EQ[IL+jax*nmd];
                    //if (ix == 0 && jax == 10) {
                    //   printf("%d        %d         %.10E        %.10E\n", IL, jax, real(eq_1), imag(eq_1));
                    //}
                    for (int jby=0; jby<3*nat; jby++) {
                        cmplx eq_2 = EQP[ILP+jby*nmd];
                        for (int n1=0; n1<nl; n1++) {
                            for (int n2=0; n2<nl; n2++) {
                                cmplx F = (0, 0);
                                for (int b=0; b<nst; b++) {
                                    int INX = jax+ap*3*nat+b*nst*3*nat;
                                    int INXP = jby+b*3*nat+a*nst*3*nat;
                                    F += FX[INXP] * FX[INX] * (1/(-WQPL[ILP]-EIG[b]+EIG[a]) + 1/(-WQL[IL]-EIG[b]+EIG[ap]));
                                    INX = jax+b*3*nat+a*nst*3*nat;
                                    INXP = jby+ap*3*nat+b*nst*3*nat;
                                    F += FX[INX] * FX[INXP] * (1/(WQL[IL]-EIG[b]+EIG[a]) + 1/(WQPL[ILP]-EIG[b]+EIG[ap]));
                                }
                                GQQP[INDG] += 0.5 * AQL[IL] * eq_1 * EIQR[n1] * F * EIQPR[n2] * eq_2 * AQPL[ILP];
                            }
                        }
                    }
                }
            }
        }
        if (i0x == 0) {
            printf("%d   -> END\n", ix);
        }
    }
}

/*
    compute g_qqp with second order Raman term
*/

__global__ void compute_gqqp_2nd_Raman(int nat, int nl, int nst, int nmd, int *INIT_INDEX, int *SIZE_LIST, int *MODES_LIST,
double *AQL, double *AQPL, double *WQL, double *WQPL, double *EIG, cmplx *FX, cmplx *FXY, cmplx *EQ, cmplx *EQP, cmplx *EIQR,
cmplx *EIQPR, cmplx *GQQP) {
    /* internal variables */
    const int i = threadIdx.x + blockDim.x * blockIdx.x;
    const int j = threadIdx.y + blockDim.y * blockIdx.y;
    const int k = threadIdx.z + blockDim.z * blockIdx.z;
    const int idx = i + j * blockDim.x * gridDim.x + k * blockDim.x * gridDim.x * blockDim.y * gridDim.y;
    const int sx = SIZE_LIST[idx];
    const int i0x = INIT_INDEX[idx];
    /* cycle over mode pairs */
    for (int ix=i0x; ix<i0x+sx; ix++) {
        /* modes index */
        int IL = MODES_LIST[2*ix];
        int ILP = MODES_LIST[2*ix+1];
        /* states index (a,a' )*/
        for (int a=0; a<nst; a++) {
            for (int ap=0; ap<nst; ap++) {
                size_t INDG = ILP+IL*nmd+ap*nmd*nmd+a*nst*nmd*nmd;
                GQQP[INDG] = (0, 0);
                for (int jax=0; jax<3*nat; jax++) {
                    cmplx eq_1 = EQ[IL+jax*nmd];
                    for (int jby=0; jby<3*nat; jby++) {
                        cmplx eq_2 = EQP[ILP+jby*nmd];
                        for (int n1=0; n1<nl; n1++) {
                            for (int n2=0; n2<nl; n2++) {
                                cmplx F = (0, 0);
                                /* compute force */
                                for (int b=0; b<nst; b++) {
                                    int INX = jax+ap*3*nat+b*nst*3*nat;
                                    int INXP = jby+b*3*nat+a*nst*3*nat;
                                    F += FX[INXP] * FX[INX] * (1/(-WQPL[ILP]-EIG[b]+EIG[a]) + 1/(-WQL[IL]-EIG[b]+EIG[ap]));
                                    INX = jax+b*3*nat+a*nst*3*nat;
                                    INXP = jby+ap*3*nat+b*nst*3*nat;
                                    F += FX[INX] * FX[INXP] * (1/(WQL[IL]-EIG[b]+EIG[a]) + 1/(WQPL[ILP]-EIG[b]+EIG[ap]));
                                }            
                                GQQP[INDG] += 0.5 * AQL[IL] * eq_1 * EIQR[n1] * F * EIQPR[n2] * eq_2 * AQPL[ILP];
                            }
                        }
                    }
                }
            }
        }
    }
}