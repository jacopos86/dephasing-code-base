#include <pycuda-complex.hpp>
#include <math.h>
#include "/home/jacopo/PROJECTS/dephasing-code-base/pydephasing/gpu_source/extern_func.cuh"
#define PI 3.141592653589793
typedef pycuda::complex<double> cmplx;

/* acf V2 (t) -> fit calculation */

__global__ void compute_acf_V2_oft(int *qlp_init, int *lgth, int *qlp_lst, const int SIZE,
double *time, double wq, double *wqp, double wuq, double *wuqp, double Alq, double *Alqp, 
cmplx *Flqlqp, double T, double DE, const double NU, const double MINFREQ, const double THZTOEV, 
const double KB, const double TOLER, cmplx *acf) {
    const int i = threadIdx.x + blockDim.x * blockIdx.x;
    const int j = threadIdx.y + blockDim.y * blockIdx.y;
    const int k = threadIdx.z + blockDim.z * blockIdx.z;
    //printf("%d ", blockIdx.z);
    int idx = i + j * blockDim.x * gridDim.x + k * blockDim.x * gridDim.x * blockDim.y * gridDim.y;
    int tx  = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
    int iqlx= blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
    /* internal variables */
    int iqlp0, n, iqlp, ii;
    double wql, Eql, x, wqlp, Eqlp;
    double re, im;
    double nql, nqlp;
    cmplx ft;
    /* check tx index */
    if (tx < SIZE) {
        iqlp0 = qlp_init[iqlx];
        n = lgth[iqlx];
        wql = 2.*PI*wuq;
        /* ph. occup. */
        Eql = wuq * THZTOEV;
        x = Eql / (KB * T);
        nql = bose_occup(x, T, TOLER);
        /* run over (q',l') */
        for (ii=iqlp0; ii<iqlp0+n; ii++) {
            iqlp = qlp_lst[ii]; 
            if (wuqp[iqlp] > MINFREQ) {
                wqlp = 2.*PI*wuqp[iqlp];
                Eqlp = wuqp[iqlp] * THZTOEV;
                /* set e^iwt */
                re = cos((DE + wqlp - wql)*time[tx]);
                im = sin((DE + wqlp - wql)*time[tx]);
                cmplx eiwt(re, -im);
                /* ph. occup.*/
                x = Eqlp / (KB * T);
                nqlp = bose_occup(x, T, TOLER);
                /* acf^2(t) */
                ft = nql * (1.+nqlp) * eiwt * exp(-NU*time[tx]);
                acf[idx] += wq * wqp[iqlp] * Alq * Alq * Alqp[iqlp] * Alqp[iqlp] * ft * Flqlqp[iqlp] * conj(Flqlqp[iqlp]);
            }
        }
    }
}

/* acf V2 at/ph (t) -> fit calculation */

__global__ void compute_acf_V2_atr_oft(int *at_lst, const int NA_SIZE, double *time, const int SIZE, 
const int NMODES, const int NAT, double DE, double NU, double wq, double wuq, double Alq, double *wqp, 
double *wuqp, double *Alqp, cmplx *Fjax_lqlqp, double T, const double MINFREQ, const double THZTOEV, 
const double KB, const double TOLER, cmplx *acf) {
    const int i = threadIdx.x + blockDim.x * blockIdx.x;
    const int j = threadIdx.y + blockDim.y * blockIdx.y;
    const int k = threadIdx.z + blockDim.z * blockIdx.z;
    int idx = i + j * blockDim.x * gridDim.x + k * blockDim.x * gridDim.x * blockDim.y * gridDim.y;
    int tx = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
    int ax = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
    /* internal vars. */
    double wql, Eql, x, wqlp, Eqlp;
    double nql, nqlp;
    double re, im;
    int ia, iqlp, iFx, dx;
    cmplx ft;
    /* tx < SIZE : PROCEED */
    if (tx < SIZE && ax < NA_SIZE) {
        ia = at_lst[ax];
        if (wuq > MINFREQ) {
            wql = 2.*PI*wuq;
            /* ph. occup. */
            Eql = wuq * THZTOEV;
            x = Eql / (KB * T);
            nql = bose_occup(x, T, TOLER);
            /* iterate over (q',l') */
            for (iqlp=0; iqlp<NMODES; iqlp++) {
                if (wuqp[iqlp] > MINFREQ) {
                    wqlp = 2.*PI*wuqp[iqlp];
                    Eqlp = wuqp[iqlp] * THZTOEV;
                    /* EXP ^ iwt */
                    re = cos((DE + wqlp - wql)*time[tx]);
                    im = sin((DE + wqlp - wql)*time[tx]);
                    cmplx eiwt(re, -im);
                    /* ph. occup. */
                    x = Eqlp / (KB * T);
                    nqlp = bose_occup(x, T, TOLER);
                    /* acf^(2)(t) */
                    ft = nql * (1.+nqlp) * eiwt * exp(-NU*time[tx]);
                    for (dx=0; dx<3; dx++) {
                        iFx = 3*NAT*iqlp+3*ia+dx;
                        acf[idx] += wq * wqp[iqlp] * Alq * Alq * Alqp[iqlp] * Alqp[iqlp] * ft * Fjax_lqlqp[iFx] * conj(Fjax_lqlqp[iFx]);
                    }
                }
            }
        }
    }
}