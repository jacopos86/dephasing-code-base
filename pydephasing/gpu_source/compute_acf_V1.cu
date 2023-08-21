#include <pycuda-complex.hpp>
#include <math.h>
#define PI 3.141592653589793
typedef pycuda::complex<double> cmplx;

/* bose occupation internal function */

__device__ double bose_occup(double x, double T, const double TOLER) {
    double nql;
    if (T < TOLER) {
        nql = 0.;
    }
    else {
        if (x > 100.) {
            nql = 0.;
        }
        else {
            nql = 1./(exp(x) - 1.);
        }
    }
    return nql;
}

/* lorentzian function */

__device__ double lorentzian(double x, double eta) {
    double ltz;
    ltz = 1./PI * eta / 2. / (x * x + eta * eta / 4.);
    return ltz;
}

/* time ACF first order */

__global__ void compute_acf_V1_oft(int *ql_init, int *lgth, int *ql_lst, const int SIZE,
double *time, double *wq, double *wuq, double *Alq, cmplx *Flq, double T, double DE, double NU,
double MINFREQ, double THZTOEV, double KB, const double TOLER, cmplx *acf, cmplx *acf_int) {
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
                /* compute cumulative sum auto correl function (eV^2 ps) units */
                ft = IU * (1. + nql) * (eiwt * exp(-NU*time[tx]) - 1.) / (wql+DE-IU*NU) - IU * nql * (cc_eiwt * exp(-NU*time[tx]) - 1.) / (wql-DE+IU*NU);
                acf_int[idx] += wq[iql] * Alq[iql] * Alq[iql] * ft * Flq[iql] * conj(Flq[iql]);
            }
        }
    }
}

/* freq. ACF order 1 */
__global__ void compute_acf_V1_ofw(int *ql_init, int *lgth, int *ql_lst, const int SIZE, double *wg,
double *wq, double *wuq, double *Alq, cmplx *Flq, double T, double DE, 
double MINFREQ, double THZTOEV, double KB, const double TOLER, double ETA, cmplx *acfw) {
    const int i = threadIdx.x + blockDim.x * blockIdx.x;
    const int j = threadIdx.y + blockDim.y * blockIdx.y;
    const int k = threadIdx.z + blockDim.z * blockIdx.z;
    int idx = i + j * blockDim.x * gridDim.x + k * blockDim.x * gridDim.x * blockDim.y * gridDim.y;
    int iwx = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
    int iqlx= blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
    /* internal variables */
    int iql0, iql, ii, n;
    double x, Eql, nql;
    double LTZ1, LTZ2, fw;
    /* run over w - ql index */
    if (iwx < SIZE) {
        iql0 = ql_init[iqlx];
        n = lgth[iqlx];
        for (ii=iql0; ii<iql0+n; ii++) {
            iql = ql_lst[ii];
            if (wuq[iql] > MINFREQ) {
                /* Eql in eV */
                Eql= wuq[iql] * THZTOEV;
                x = Eql / (KB * T);
                nql = bose_occup(x, T, TOLER);
                /* compute lorentzians */
                x = DE + Eql + wg[iwx];
                LTZ1 = lorentzian(x, ETA);
                x = DE - Eql + wg[iwx];
                LTZ2 = lorentzian(x, ETA);
                fw = (1.+nql) * LTZ1 + nql * LTZ2;
                /* compute ACF */
                acfw[idx] += wq[iql] * Alq[iql] * Alq[iql] * fw * Flq[iql] * conj(Flq[iql]);
            }
        }
    }
}

/* atom res. ACF of t*/

__global__ void compute_acf_V1_atr_oft(int *at_lst, double *wq, double *wuq, double *time,
double DE, double NU, cmplx *Fjax_lq, double *Alq, int SIZE, int NA_SIZE, int NMODES, int NAT, 
double T, double MINFREQ, double THZTOEV, double KB, double TOLER, cmplx *acf, cmplx *acf_int) {
    const int i = threadIdx.x + blockDim.x * blockIdx.x;
    const int j = threadIdx.y + blockDim.y * blockIdx.y;
    const int k = threadIdx.z + blockDim.z * blockIdx.z;
    int idx = i + j * blockDim.x * gridDim.x + k * blockDim.x * gridDim.x * blockDim.y * gridDim.y;
    int tx = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
    int ax = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
    int ia = at_lst[ax];
    /* internal vars*/
    int iql;
    int dx, iFx;
    double re, im;
    double wql, Eql, x, nql;
    cmplx ft;
    const cmplx IU(0., 1.);
    /* check tx < SIZE */
    if (tx < SIZE && ax < NA_SIZE) {
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
                /* compute cumulative sum auto correl function (eV^2 ps) units */
                ft = IU * (1. + nql) * (eiwt * exp(-NU*time[tx]) - 1.) / (wql+DE-IU*NU) - IU * nql * (cc_eiwt * exp(-NU*time[tx]) - 1.) / (wql-DE+IU*NU);
                for (dx=0; dx<3; dx++) {
                    iFx = 3*NAT*iql+3*ia+dx;
                    acf_int[idx] += wq[iql] * Alq[iql] * Alq[iql] * ft * Fjax_lq[iFx] * conj(Fjax_lq[iFx]);
                }
            }
        }
    }
}

/* atom res. ACF of t*/

__global__ void compute_acf_V1_phr_oft(int NPH, int *ph_lst, int SIZE, double *time, double *wq,
double *wuq, double *Alq, cmplx *Flq, double T, double DE, double NU, double MINFREQ, double THZTOEV, 
double KB, double TOLER, cmplx *acf, cmplx *acf_int) {
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
            /* compute cumulative sum auto correl function (eV^2 ps) units */
            ft = IU * (1. + nql) * (eiwt * exp(-NU*time[tx]) - 1.) / (wql+DE-IU*NU) - IU * nql * (cc_eiwt * exp(-NU*time[tx]) - 1.) / (wql-DE+IU*NU);
            acf_int[idx] += wq[iql] * Alq[iql] * Alq[iql] * ft * Flq[iql] * conj(Flq[iql]);
        }
    }
}

/* freq. ACF order 1 -> ATR */
__global__ void compute_acf_V1_atr_ofw(int NA_SIZE, int *at_lst, const int SIZE, double *wg, double *wq, 
double *wuq, double *Alq, cmplx *Fjax_lq, double T, double DE, int NMODES, int NAT,
double MINFREQ, double THZTOEV, double KB, const double TOLER, double ETA, cmplx *acfw) {
    const int i = threadIdx.x + blockDim.x * blockIdx.x;
    const int j = threadIdx.y + blockDim.y * blockIdx.y;
    const int k = threadIdx.z + blockDim.z * blockIdx.z;
    int idx = i + j * blockDim.x * gridDim.x + k * blockDim.x * gridDim.x * blockDim.y * gridDim.y;
    int iwx = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
    int ax = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
    int ia = at_lst[ax];
    /* internal variables */
    int iql;
    int dx, iFx;
    double Eql, x, nql;
    double LTZ1, LTZ2, fw;
    /* tx - na SIZE*/
    if (iwx < SIZE && ax < NA_SIZE) {
        for (iql=0; iql<NMODES; iql++) {
            if (wuq[iql] > MINFREQ) {
                /* Eql in eV*/
                Eql = wuq[iql] * THZTOEV;
                x = Eql / (KB * T);
                nql = bose_occup(x, T, TOLER);
                /* compute lorentzians */
                x = DE + Eql + wg[iwx];
                LTZ1 = lorentzian(x, ETA);
                x = DE - Eql + wg[iwx];
                LTZ2 = lorentzian(x, ETA);
                fw = (1.+nql) * LTZ1 + nql * LTZ2;
                /* compute ACF(w) - eV*/
                for (dx=0; dx<3; dx++) {
                    iFx = 3*NAT*iql+3*ia+dx;
                    acfw[idx] += wq[iql] * Alq[iql] * Alq[iql] * fw * Fjax_lq[iFx] * conj(Fjax_lq[iFx]);
                }
            }
        }
    }
}

/* freq. ACF order 1 -> PHR */
__global__ void compute_acf_V1_phr_ofw(int NPH, int *ph_lst, const int SIZE, double *wg, double *wq,
double *wuq, double *Alq, cmplx *Flq, double T, double DE, double MINFREQ, double THZTOEV, 
double KB, const double TOLER, double ETA, cmplx *acfw) {
    const int i = threadIdx.x + blockDim.x * blockIdx.x;
    const int j = threadIdx.y + blockDim.y * blockIdx.y;
    const int k = threadIdx.z + blockDim.z * blockIdx.z;
    int idx = i + j * blockDim.x * gridDim.x + k * blockDim.x * gridDim.x * blockDim.y * gridDim.y;
    int iwx = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
    int phx = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
    int iql = ph_lst[phx];
    /* internal variables*/
    double Eql, x, nql;
    double LTZ1, LTZ2, fw;
    /* check tx size*/
    if (iwx < SIZE && phx < NPH) {
        if (wuq[iql] > MINFREQ) {
            /* Eql in eV*/
            Eql = wuq[iql] * THZTOEV;
            x = Eql / (KB * T);
            nql = bose_occup(x, T, TOLER);
            /* compute lorentzians*/
            x = DE + Eql + wg[iwx];
            LTZ1 = lorentzian(x, ETA);
            x = DE - Eql + wg[iwx];
            LTZ2 = lorentzian(x, ETA);
            /* compute ACF(w) - eV*/
            fw = (1.+nql) * LTZ1 + nql * LTZ2;
            acfw[idx] += wq[iql] * Alq[iql] * Alq[iql] * fw * Flq[iql] * conj(Flq[iql]);
        }
    }
}