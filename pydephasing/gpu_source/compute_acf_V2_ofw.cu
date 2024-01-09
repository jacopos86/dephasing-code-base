#include <pycuda-complex.hpp>
#include <math.h>
#include "/home/jacopo/PROJECTS/dephasing-code-base/pydephasing/gpu_source/extern_func.cuh"
#define PI 3.141592653589793
typedef pycuda::complex<double> cmplx;

/* acf V2 (w) function */

__global__ void compute_acf_V2_ofw(int *qlp_init, int *lgth, int *qlp_lst, const int SIZE,
double *wg, double wq, double *wqp, double wuq, double *wuqp, double Alq, double *Alqp,
cmplx *Flqlqp, double T, double DE, double MINFREQ, double THZTOEV, double KB, 
const double TOLER, double ETA, cmplx *acfw) {
    const int i = threadIdx.x + blockDim.x * blockIdx.x;
    const int j = threadIdx.y + blockDim.y * blockIdx.y;
    const int k = threadIdx.z + blockDim.z * blockIdx.z;
    int idx = i + j * blockDim.x * gridDim.x + k * blockDim.x * gridDim.x * blockDim.y * gridDim.y;
    int iwx = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
    int iqlpx= blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
    /* internal variables */
    int iqlp0, n, ii, iqlp;
    double Eql, x, nql, Eqlp, nqlp;
    double LTZ, fw;
    /* check iwx */
    if (iwx < SIZE) {
        iqlp0 = qlp_init[iqlpx];
        n = lgth[iqlpx];
        /* ph. occup.*/
        Eql = wuq * THZTOEV;
        x = Eql / (KB * T);
        nql = bose_occup(x, T, TOLER);
        /* cycle (q',l') */
        for (ii=iqlp0; ii<iqlp0+n; ii++) {
            iqlp = qlp_lst[ii];
            if (wuqp[iqlp] > MINFREQ) {
                Eqlp = wuqp[iqlp] * THZTOEV;
                /* ph. occup. */
                x = Eqlp / (KB * T);
                nqlp = bose_occup(x, T, TOLER);
                /* compute lorentzian */
                x = DE + Eqlp - Eql + wg[iwx];
                LTZ = lorentzian(x, ETA);
                fw = nql * (1. + nqlp) * LTZ;
                /* compute ACF */
                acfw[idx] += wq * wqp[iqlp] * Alq * Alq * Alqp[iqlp] * Alqp[iqlp] * fw * Flqlqp[iqlp] * conj(Flqlqp[iqlp]);
            }
        }
    }
}

/* acf V2 at/ph (w) function */

__global__ void compute_acf_V2_atr_ofw(int *at_lst, int NA_SIZE, double *wg, const int SIZE, int NMODES,
int NAT, double DE, double ETA, double wq, double wuq, double Alq, double *wqp, double *wuqp,
double *Alqp, cmplx *Fjax_lqlqp, double T, double MINFREQ, double THZTOEV, double KB, double TOLER, cmplx *acfw) {
    const int i = threadIdx.x + blockDim.x * blockIdx.x;
    const int j = threadIdx.y + blockDim.y * blockIdx.y;
    const int k = threadIdx.z + blockDim.z * blockIdx.z;
    int idx = i + j * blockDim.x * gridDim.x + k * blockDim.x * gridDim.x * blockDim.y * gridDim.y;
    int iwx = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
    int ax = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
    /* internal vars. */
    double wql, Eql, x, wqlp, Eqlp;
    double nql, nqlp;
    double LTZ, fw;
    int dx, iFx, iqlp;
    /* tx < SIZE : PROCEED */
    if (iwx < SIZE && ax < NA_SIZE) {
        int ia = at_lst[ax];
        if (wuq > MINFREQ) {
            /* ph. occup. */
            Eql = wuq * THZTOEV;
            x = Eql / (KB * T);
            nql = bose_occup(x, T, TOLER);
            /* iterate over (q',l')*/
            for (iqlp=0; iqlp<NMODES; iqlp++) {
                if (wuqp[iqlp] > MINFREQ) {
                    Eqlp = wuqp[iqlp] * THZTOEV;
                    /* ph. occup. number */
                    x = Eqlp / (KB * T);
                    nqlp = bose_occup(x, T, TOLER);
                    /* compute lorentzian func. */
                    x = DE + Eqlp - Eql + wg[iwx];
                    LTZ = lorentzian(x, ETA);
                    fw = nql * (1. + nqlp) * LTZ;
                    /* atoms iteration */
                    for (dx=0; dx<3; dx++) {
                        iFx = 3*NAT*iqlp+3*ia+dx;
                        acfw[idx] += wq * wqp[iqlp] * Alq * Alq * Alqp[iqlp] * Alqp[iqlp] * fw * Fjax_lqlqp[iFx] * conj(Fjax_lqlqp[iFx]);
                    }
                }
            }
        }
    }
}