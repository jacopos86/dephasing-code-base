#include <pycuda-complex.hpp>
#include <math.h>
#include "/home/jacopo/PROJECTS/dephasing-code-base/pydephasing/gpu_source/extern_func.cuh"
#define PI 3.141592653589793
typedef pycuda::complex<double> cmplx;

/* time ACF first order */

__global__ void compute_acf_V1_oft(int *ql_init, int *lgth, int *ql_lst, const int SIZE,
double *time, double *wq, double *wuq, double *Alq, cmplx *Flq, double T, double DE, double NU,
double MINFREQ, double THZTOEV, double KB, const double TOLER, cmplx *acf) {
    const int i = threadIdx.x + blockDim.x * blockIdx.x;
    const int j = threadIdx.y + blockDim.y * blockIdx.y;
    const int k = threadIdx.z + blockDim.z * blockIdx.z;
    int idx = i + j * blockDim.x * gridDim.x + k * blockDim.x * gridDim.x * blockDim.y * gridDim.y;
    int tx  = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
    int iqlx= blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
    /* internal variables */
    double x, wql, Eql, nql;
    int iql0, iql, n, ii;
    double re, im;
    cmplx ft;
    const cmplx IU(0., 1.);
    /* run over t and iql index */
    if (tx < SIZE) {
        iql0= ql_init[iqlx];
        n = lgth[iqlx];
        for (ii=iql0; ii<iql0+n; ii++) {
            iql = ql_lst[ii];
            if (wuq[iql] > MINFREQ) {
                wql= 2.*PI*wuq[iql]; 
                re = cos((wql+DE)*time[tx]);
                im = sin((wql+DE)*time[tx]);
                cmplx eiwt(re, -im);
                re = cos((wql-DE)*time[tx]);
                im = sin((wql-DE)*time[tx]);
                cmplx cc_eiwt(re, im);
                /* ph. occup */
                Eql = wuq[iql] * THZTOEV;
                x = Eql / (KB * T);
                nql = bose_occup(x, T, TOLER);
                /* compute auto correl functions (eV^2) units */
                ft = ((1. + nql) * eiwt + nql * cc_eiwt) * exp(-NU*time[tx]);
                acf[idx] += wq[iql] * Alq[iql] * Alq[iql] * ft * Flq[iql] * conj(Flq[iql]);
            }
        }
    }
}

/* atom res. ACF of t*/

__global__ void compute_acf_V1_atr_oft(int *at_lst, double *wq, double *wuq, double *time,
double DE, double NU, cmplx *Fjax_lq, double *Alq, int SIZE, int NA_SIZE, int NMODES, int NAT, 
double T, double MINFREQ, double THZTOEV, double KB, double TOLER, cmplx *acf) {
    const int i = threadIdx.x + blockDim.x * blockIdx.x;
    const int j = threadIdx.y + blockDim.y * blockIdx.y;
    const int k = threadIdx.z + blockDim.z * blockIdx.z;
    int idx = i + j * blockDim.x * gridDim.x + k * blockDim.x * gridDim.x * blockDim.y * gridDim.y;
    int tx = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
    int ax = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
    /* internal vars*/
    int iql;
    int dx, iFx;
    double re, im;
    double wql, Eql, x, nql;
    cmplx ft;
    const cmplx IU(0., 1.);
    /* check tx < SIZE */
    if (tx < SIZE && ax < NA_SIZE) {
        int ia = at_lst[ax];
        /* iterate over (q,l) */
        for (iql=0; iql<NMODES; iql++) {
            if (wuq[iql] > MINFREQ) {
                wql = 2.*PI*wuq[iql];
                re = cos((wql+DE)*time[tx]);
                im = sin((wql+DE)*time[tx]);
                cmplx eiwt(re, -im);
                re = cos((wql-DE)*time[tx]);
                im = sin((wql-DE)*time[tx]);
                cmplx cc_eiwt(re, im);
                /* ph. occup. */
                Eql = wuq[iql] * THZTOEV;
                x = Eql / (KB * T);
                nql = bose_occup(x, T, TOLER);
                /* compute auto correl functions (eV^2) units */
                ft = ((1. + nql) * eiwt + nql * cc_eiwt) * exp(-NU*time[tx]);
                for (dx=0; dx<3; dx++) {
                    iFx = 3*NAT*iql+3*ia+dx;
                    acf[idx] += wq[iql] * Alq[iql] * Alq[iql] * ft * Fjax_lq[iFx] * conj(Fjax_lq[iFx]);
                }
            }
        }
    }
}

/* atom res. ACF of t*/

__global__ void compute_acf_V1_phr_oft(int NPH, int *ph_lst, int SIZE, double *time, double *wq,
double *wuq, double *Alq, cmplx *Flq, double T, double DE, double NU, double MINFREQ, double THZTOEV, 
double KB, double TOLER, cmplx *acf) {
    const int i = threadIdx.x + blockDim.x * blockIdx.x;
    const int j = threadIdx.y + blockDim.y * blockIdx.y;
    const int k = threadIdx.z + blockDim.z * blockIdx.z;
    int idx= i + j * blockDim.x * gridDim.x + k * blockDim.x * gridDim.x * blockDim.y * gridDim.y;
    int tx = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
    int phx= blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
    int iql = ph_lst[phx];
    /* internal vars*/
    double wql, Eql, x, nql;
    double re, im;
    cmplx ft;
    const cmplx IU(0., 1.);
    /* check tx < SIZE */
    if (tx < SIZE && phx < NPH) {
        if (wuq[iql] > MINFREQ) {
            wql = 2.*PI*wuq[iql];
            re = cos((wql+DE)*time[tx]);
            im = sin((wql+DE)*time[tx]);
            cmplx eiwt(re, -im);
            re = cos((wql-DE)*time[tx]);
            im = sin((wql-DE)*time[tx]);
            cmplx cc_eiwt(re, im);
            /* ph. occup.*/
            Eql = wuq[iql] * THZTOEV;
            x = Eql / (KB * T);
            nql = bose_occup(x, T, TOLER);
            /* compute ACF (eV^2) units*/
            ft = ((1. + nql) * eiwt + nql * cc_eiwt) * exp(-NU*time[tx]);
            acf[idx] += wq[iql] * Alq[iql] * Alq[iql] * ft * Flq[iql] * conj(Flq[iql]);
        }
    }
}