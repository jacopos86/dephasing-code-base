#include <pycuda-complex.hpp>
#define PI 3.141592653589793
typedef pycuda::complex<double> cmplx;

__global__ void compute_acf_V2_oft(int *qlp_init, int *lgth, int *qlp_lst, const int t_size,
double *time, double wq, double *wqp, double wuq, double *wuqp, double Alq, double *Alqp, 
cmplx *Flqlqp, double T, double DE, double NU, double MINFREQ, double THZTOEV, double KB, 
const double TOLER, cmplx *acf, cmplx *acf_int) {
    const int i = threadIdx.x + blockDim.x * blockIdx.x;
    const int j = threadIdx.y + blockDim.y * blockIdx.y;
    const int k = threadIdx.z + blockDim.z * blockIdx.z;
    //printf("%d ", blockIdx.z);
    int idx = i + j * blockDim.x * gridDim.x + k * blockDim.x * gridDim.x * blockDim.y * gridDim.y;
    int tx  = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
    int ipx = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
    int imp, ii0, ii, nmp;
    double x, nlqp, Eqlp;
    double re, im;
    double A1, A2, A3;
    cmplx ft;
    if (tx < t_size) {
        ii0 = qlp0_lst[ipx];
        nmp = lgth_lst[ipx];
        for (ii=ii0; ii<ii0+nmp; ii++) {
            imp = qlp_lst[ii]; 
            if (wuq[imp] > min_freq) {
                Eqlp = wuq[imp] * THz_to_ev;
                re = cos(2.*PI*wuq[imp]*time[tx]);
                im = sin(2.*PI*wuq[imp]*time[tx]);
                cmplx eiwpt(re, -im);
                cmplx cc_eiwpt = conj(eiwpt);
                if (T < toler) {
                    nlqp = 0.;
                }
                else {
                    x = Eqlp / (kb * T);
                    if (x > 100.) {
                        nlqp = 0.;
                    }
                    else {
                        nlqp = 1./(exp(x) - 1.);
                    }
                }
                A1 = 1. + nlq + nlqp + nlq * nlqp;
                A2 = nlq * nlqp;
                A3 = 2.*(1. + nlq) * nlqp;
                ft = A1 * eiwt[tx] * eiwpt + A2 * cc_eiwpt * conj(eiwt[tx]) + A3 * eiwt[tx] * cc_eiwpt * eint[tx];
                acf[idx] += wq * wqp[imp] * Alq * Alq * Alqp[imp] * Alqp[imp] * ft * Flqp[imp] * conj(Flqp[imp]);
            }
        }
    }
}