#include "extern_func.cuh"
#include <pycuda-complex.hpp>
#include <cuComplex.h>
typedef pycuda::complex<double> cmplx;

__global__ void compute_T1_oneph_w_resolved(int n, int nw, double KT, int *INIT_INDEX, int *SIZE_LIST,
double *WGR, double *WQ, double *WQL) {
    /* internal variables */
    const int i = threadIdx.x + blockDim.x * blockIdx.x;
    const int j = threadIdx.y + blockDim.y * blockIdx.y;
    const int k = threadIdx.z + blockDim.z * blockIdx.z;
    const int idx = i + j * blockDim.x * gridDim.x + k * blockDim.x * gridDim.x * blockDim.y * gridDim.y;

    /* local variables */
    const int sx = SIZE_LIST[idx];
    const int i0x = INIT_INDEX[idx];

    for (int ix=i0x; ix<i0x+sx; ix++) {
        double x = WQL[ix] / KT;
        double nq = bose_occup(x);
        for (int iw=0; iw<nw; iw++) {
            double w = WGR[iw];
            for (int a1=0; a1<n; a1++) {
                for (int a=0; a<n; a++) {
                    cmplx gq = 0.0;
                }
            }
        }
    }


}


__global__ void compute_T1_oneph_time_resolved(int n, int nql, int nt, double KT, double eta, int *INIT_INDEX, int *SIZE_LIST, 
double *TIME, double *WQ, double *WQL, cmplx *GQL, double *GOFT, double *INTGOFT) {
    /* internal variables */
    const int i = threadIdx.x + blockDim.x * blockIdx.x;
    const int j = threadIdx.y + blockDim.y * blockIdx.y;
    const int k = threadIdx.z + blockDim.z * blockIdx.z;
    const int idx = i + j * blockDim.x * gridDim.x + k * blockDim.x * gridDim.x * blockDim.y * gridDim.y;

    /* local variables */
    const int sx = SIZE_LIST[idx];
    const int i0x = INIT_INDEX[idx];

    for (int ix=i0x; ix<i0x+sx; ix++) {
        if (ix == 1) {
            printf("%.10E       %.10E\n", real(GQL[ix+1*nql+1*n*nql]), imag(GQL[ix+1*nql+1*n*nql]));
        }
        double x = WQL[ix] / KT;
        double nq = bose_occup(x);
        for (int it=0; it<nt; it++) {
            double t = TIME[it];
            for (int a1=0; a1<n; a1++) {
                int INGT = it + a1*nt;
                for (int a=0; a<n; a++) {
                    int INGQL = ix + a1*nql + a*n*nql;
                    cmplx g = GQL[INGQL];
                    cmplx g_conj = (g.real(), -g.imag());
                    GOFT[INGT] += (g * g_conj).real();
                }
            }
        }
    }


}