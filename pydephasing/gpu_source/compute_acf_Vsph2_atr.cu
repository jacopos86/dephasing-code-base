#include <pycuda-complex.hpp>
#define PI 3.141592653589793
typedef pycuda::complex<double> cmplx;

__global__ void compute_acf_Vsph2_atr(double *wqp, double *wuq, cmplx *eiwt, double *eint,
double *time, cmplx *Flqp, double *Alqp, int *iax_lst, double Alq, double T, double wq, 
double nlq, double THz_to_ev, double min_freq, double kb, const double toler,
const int t_size, const int na_size, int nmodes, int nat, cmplx *acf) {
    const int i = threadIdx.x + blockDim.x * blockIdx.x;
    const int j = threadIdx.y + blockDim.y * blockIdx.y;
    const int k = threadIdx.z + blockDim.z * blockIdx.z;
    int idx = i + j * blockDim.x * gridDim.x + k * blockDim.x * gridDim.x * blockDim.y * gridDim.y;
    int tx = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
    int ax = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
    int imp;
    int iax, dx, iFx;
    double re, im, Eqlp, x, nlqp;
    double A1, A2, A3;
    cmplx ft;
    if (ax < na_size && tx < t_size) {
        for (imp=0; imp<nmodes; imp++) {
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
                iax = iax_lst[ax];
                for (dx=0; dx<3; dx++) {
                    iFx = 3*nat*imp + 3*iax + dx;
                    acf[idx] += wq * wqp[imp] * Alq * Alq * Alqp[imp] * Alqp[imp] * ft * Flqp[iFx] * conj(Flqp[iFx]);
                }
            }
        }
    }
}