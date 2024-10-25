#include <pycuda-complex.hpp>
#define PI 3.141592653589793
typedef pycuda::complex<double> cmplx;

__global__ void compute_acf_Vsph2_dyndec(double *wqp, double *wuq, double *time,
cmplx *Flqp, double *Alqp, double *wk, int *qlp_lst, int *qlp0_lst, int *lgth_lst, 
int *it0_lst, int *it1_lst, double *factorial_k, double Alq, double T, double wq, double wql, double nlq, double THz_to_ev, 
double min_freq, double kb, const double toler, double nu, int nt0, int ndkt, const int t_size, cmplx *acf) {
    const int i = threadIdx.x + blockDim.x * blockIdx.x;
    const int j = threadIdx.y + blockDim.y * blockIdx.y;
    const int k = threadIdx.z + blockDim.z * blockIdx.z;
    //printf("%d ", blockIdx.z);
    int idx = i + j * blockDim.x * gridDim.x + k * blockDim.x * gridDim.x * blockDim.y * gridDim.y;
    int tx  = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
    int ipx = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
    int imp, ii0, ii, nmp;
    int it, t0, t1, kk;
    double x, nlqp, Eqlp, wqlp, rho, th;
    double A1, A2, A3;
    cmplx eiw1, eiw2, eiw3;
    cmplx fk, dtk0_acf, ak;
    // check time
    if (tx < t_size) {
        ii0 = qlp0_lst[ipx];
        nmp = lgth_lst[ipx];
        for (ii=ii0; ii<ii0+nmp; ii++) {
            imp = qlp_lst[ii]; 
            if (wuq[imp] > min_freq) {
                Eqlp = wuq[imp] * THz_to_ev;
                wqlp = wuq[imp] * 2. * PI;
                // phonon amplitudes
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
                // (wlq-wlqp-inu)^k = rho^k e^{ik theta}
                rho = sqrt((wql-wqlp)*(wql-wqlp) + nu*nu);
                th = atanf(-nu/(wql-wqlp));
                // compute taylor coefficients
                for (it=0;it<nt0;it++) {
                    t0 = it0_lst[it];
                    t1 = it1_lst[it];
                    if (tx >= t0 && tx < t1) {
                        eiw1 = cmplx(cos((wqlp+wql)*time[t0]),-sin((wqlp+wql)*time[t0]));
                        eiw2 = conj(eiw1);
                        eiw3 = cmplx(cos((-wqlp+wql)*time[t0]),-sin((-wqlp+wql)*time[t0]))*exp(-nu*time[t0]);
                        for (kk=0;kk<ndkt;kk++) {
                            dtk0_acf = cmplx(0., 0.);
                            fk = pow(-1.,kk) * pow(wql+wqlp,kk) * eiw1 * A1 + pow(wql+wqlp,kk) * eiw2 * A2 + pow(-1.,kk) * pow(rho,kk) * cmplx(cos(kk*th),sin(kk*th)) * eiw3 * A3;
                            dtk0_acf = cmplx(cos(kk*PI/2),sin(kk*PI/2)) * wq * wqp[imp] * Alq * Alq * Alqp[imp] * Alqp[imp] * fk * Flqp[imp] * conj(Flqp[imp]);
                            ak = dtk0_acf * wk[kk] / factorial_k[kk];
                            acf[idx] += ak * pow(time[tx]-time[t0],kk);
                        }
                    }
                    //acf[idx] += wq * wqp[imp] * Alq * Alq * Alqp[imp] * Alqp[imp] * ft * Flqp[imp] * conj(Flqp[imp]);
                }
            }
        }
    }
}