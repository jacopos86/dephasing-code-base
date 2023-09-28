#include <pycuda-complex.hpp>
#define PI 3.141592653589793
typedef pycuda::complex<double> cmplx;

__global__ void compute_acf_V2_oft(int *qlp_init, int *lgth, int *qlp_lst, const int SIZE,
double *time, double wq, double *wqp, double wuq, double *wuqp, double Alq, double *Alqp, 
cmplx *Flqlqp, double T, double DE, double NU, double MINFREQ, double THZTOEV, double KB, 
const double TOLER, cmplx *acf, cmplx *acf_int) {
    const int i = threadIdx.x + blockDim.x * blockIdx.x;
    const int j = threadIdx.y + blockDim.y * blockIdx.y;
    const int k = threadIdx.z + blockDim.z * blockIdx.z;
    //printf("%d ", blockIdx.z);
    int idx = i + j * blockDim.x * gridDim.x + k * blockDim.x * gridDim.x * blockDim.y * gridDim.y;
    int tx  = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
    int iqlx= blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
    /* internal variables */

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
                ft = nql * (1.+nqlp) * eiwt * EXP(-NU*time[tx]);
                acf[idx] += wq * wqp[iqlp] * Alq * Alq * Alqp[iqlp] * Alqp[iqlp] * ft * Flqlqp[iqlp] * conj(Flqlqp[iqlp]);
                /* \int acf^2(t) */
                ft(0.,0.);
                ft = nql * (1.+nqlp) * eiwt * EXP(-NU*time[tx]);
            }
        }
    }
}