#include <pycuda-complex.hpp>
#include <math.h>
#include <cuComplex.h>
typedef pycuda::complex<double> cmplx;
#define PI 3.141592653589793

/* compute phr forces */
__global__ void compute_Flq_lqp(int *qp_lst, int *ilp_lst, int *jax_lst, const int size, const int nax, const int nat,
double wql, double *wqlp, cmplx *euq, cmplx *euqlp, double *r_lst, double *qv_lst, double *m_lst, cmplx *eiqR, 
const int nqs, const int qs0, const int qs1, double *eig, cmplx *Fax, const int calc_raman, const int calc_typ,
cmplx *F_lqlqp) {
    // cmplx *Fax
    // cmplx *Faxby
    // cmplx *F_lmqlqp, cmplx *F_lqlmqp, cmplx *F_lmqlmqp
    /* internal variables */
    const int i = threadIdx.x + blockDim.x * blockIdx.x;
    const int j = threadIdx.y + blockDim.y * blockIdx.y;
    const int k = threadIdx.z + blockDim.z * blockIdx.z;
    const int idx = i + j * blockDim.x * gridDim.x + k * blockDim.x * gridDim.x * blockDim.y * gridDim.y;
    const int iqlx= threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
    const int jx  = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
    int jby, ib;
    int ms, i0m, i1m, im0, im1;
    double re, im, qpR;
    double Ma, Mb;
    cmplx F;
    /* local (q',l') pair */
    const int iqp = qp_lst[iqlx];
    const int ilp = ilp_lst[iqlx];
    /* jax */
    const int jax = jax_lst[jx];
    /* check if thread is allowed to compute */
    if (iqlx < size && jx < nax) {
        /* Ma */
        Ma = m_lst[jax];
        /* run over jby index */
        for (jby=0; jby<3*nat; jby++) {
            ib = jby / 3;
            /* atom mass */
            Mb = m_lst[jby];
            /*e^iq'R*/
            qpR  = qv_lst[iqp*3]*r_lst[ib*3];
            qpR += qv_lst[iqp*3+1]*r_lst[ib*3+1];
            qpR += qv_lst[iqp*3+2]*r_lst[ib*3+2];
            re = cos(2.*PI*qpR);
            im = sin(2.*PI*qpR);
            cmplx eiqpR(re, im);
            /*       eff. force                   */
            //  F = Faxby[3*nat*jax+jby];
            /* ---------------------------------- */
            /*     Raman calculation              */
            /* ---------------------------------- */
            if (calc_raman == 1) {
                cmplx Fr(0.,0.);
                if (calc_typ == 0) {
                    /* deph calc. */
                    for (ms=0; ms<nqs; ms++) {
                        i0m = jax*nqs*nqs+qs0*nqs+ms;
                        i1m = jax*nqs*nqs+qs1*nqs+ms;
                        im0 = jby*nqs*nqs+ms*nqs+qs0;
                        im1 = jby*nqs*nqs+ms*nqs+qs1;
                    //    Fr += Fax[i0m] * Fax[im0] / (eig[qs0]-eig[ms]+wql);
                    //    Fr -= Fax[i1m] * Fax[im1] / (eig[qs1]-eig[ms]+wql);
                    //    Fr += Fax[i0m] * Fax[im0] / (eig[qs0]-eig[ms]-wqlp[iqlx]);
                    //    Fr -= Fax[i1m] * Fax[im1] / (eig[qs1]-eig[ms]-wqlp[iqlx]);
                    }
                }
                else if (calc_typ == 1) {
                    /* relax calc. */
                    for (ms=0; ms<nqs; ms++) {
                        i0m = jax*nqs*nqs+qs0*nqs+ms;
                        im1 = jby*nqs*nqs+ms*nqs+qs1;
                    //    Fr += Fax[i0m] * Fax[im1] / (eig[qs1]-eig[ms]+wqlp[iqlx]);
                    //    Fr += Fax[i0m] * Fax[im1] / (eig[qs1]-eig[ms]-wql);
                    }
                }
                F += Fr;
            }
            F_lqlqp[idx] += F * eiqpR * euqlp[3*nat*iqlx+jby] / sqrt (Mb);
            //F_lqlmqp[idx]+= F * eiqpR * euqlp[3*nat*iqlx+jby] / sqrt (Mb);
            //F_lmqlqp[idx]+= F * conj(eiqpR) * conj(euqlp[3*nat*iqlx+jby]) / sqrt (Mb);
            //F_lmqlmqp[idx]+= F * conj(eiqpR) * conj(euqlp[3*nat*iqlx+jby]) / sqrt (Mb);
        }
        /* multiply with l.h.s*/
        F_lqlqp[idx] = eiqR[jax] * euq[jax] / sqrt (Ma) * F_lqlqp[idx];
        //F_lmqlqp[idx]= conj(eiqR[jax]) * conj(euq[jax]) / sqrt (Ma) * F_lmqlqp[idx];
        //F_lqlmqp[idx] = eiqR[jax] * euq[jax] / sqrt (Ma) * F_lqlmqp[idx];
        //F_lmqlmqp[idx]= conj(eiqR[jax]) * conj(euq[jax]) / sqrt (Ma) * F_lmqlmqp[idx];
    }
}

/* Raman function calculation */
//__global__ cmplx compute_raman_force2(int qs0, int qs1, int nqs, cmplx *Fjax, cmplx *Fjby,
//double *eig, double wql, double wqlp, int calc_typ) {
    /* internal variables */
 //   int ms;
 //   cmplx Fr(0.,0.);
    /* if deph/rel */
 //   if (calc_typ == 0) {
        /* deph calculation*/
 //       for (ms=0; ms<nqs; ms++) {
 //           Fr += Fjax[qs0*nqs+ms] * Fjby[ms*nqs+qs0] / (eig[qs0] - eig[ms] + wql);
 //           Fr -= Fjax[qs1*nqs+ms] * Fjby[ms*nqs+qs1] / (eig[qs1] - eig[ms] + wql);
 //           Fr += Fjax[qs0*nqs+ms] * Fjby[ms*nqs+qs0] / (eig[qs0] - eig[ms] - wqlp);
 //           Fr -= Fjax[qs1*nqs+ms] * Fjby[ms*nqs+qs1] / (eig[qs1] - eig[ms] - wqlp);
 //       }
 //   }
 //   else {
        /* relax calculation */
 //       for (ms=0; ms<nqs; ms++) {
 //           Fr += Fjax[qs0*nqs+ms] * Fjby[ms*nqs+qs1] / (eig[qs1] - eig[ms] + wqlp);
 //           Fr += Fjax[qs0*nqs+ms] * Fjby[ms*nqs+qs1] / (eig[qs1] - eig[ms] - wql);
 //       }
 //   }
 //   return Fr;
//}

/*

Flq_lqp force calculation

*/

__global__ void compute_Flqlqp(int *qp_lst, int *ilp_lst, int *jby_lst, int nqlp,
int nby, int nat, cmplx *euqlp, double *r_lst, double *qv_lst, double *m_lst, double *f_axby,
cmplx *f_lqlqp, cmplx *f_lmqlqp, cmplx *f_lqlmqp, cmplx *f_lmqlmqp) {
    /* internal variables */
    const int i = threadIdx.x + blockDim.x * blockIdx.x;
    const int j = threadIdx.y + blockDim.y * blockIdx.y;
    const int k = threadIdx.z + blockDim.z * blockIdx.z;
    const int idx = i + j * blockDim.x * gridDim.x + k * blockDim.x * gridDim.x * blockDim.y * gridDim.y;
    const int iqlx= threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
    const int jx= blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
    /* local (q',l') pair */
    const int iqp = qp_lst[iqlx];
    const int ilp = ilp_lst[iqlx];
    /* jby index */
    const int jby = jby_lst[jx];
    /* start calculation on the thread */
    if (iqlx < nqlp && jx < nby) {
        /* mass Mb */
        Mb = m_lst[jby];
        /* atom's index */
        ib = jby / 3;
        /* e^iq'Rb */
        qpRb = qv_lst[iqp*3]*r_lst[ib*3];
        qpRb+= qv_lst[iqp*3+1]*r_lst[ib*3+1];
        qpRb+= qv_lst[iqp*3+2]*r_lst[ib*3+2];
        re = cos(2.*PI*qpRb);
        im = sin(2.*PI*qpRb);
        cmplx eiqpRb(re, im);
        /* compute force */
        r1 = f_axby[jx] * eiqpRb * euqlp[3*nat*iqlx+jby] / sqrt(Mb);
        f_lqlqp[idx]  += r1;
        f_lmqlqp[idx] += r1;
        r2 = f_axby[jx] * conj(eiqpRb) * conj(euqlp[3*nat*iqlx+jby]) / sqrt(Mb);
        f_lqlmqp[idx] += r2;
        f_lmqlmqp[idx]+= r2;
    }
}

/*

Flq_lqp Raman force calculation

*/

__global__ void compute_Flqlqp_raman(int *qp_lst, int *ilp_lst, int *jbyr_lst, int *jaxby_lst,
int nqlp, int nby, cmplx *euqlp, double *r_lst, double *qv_lst, double *m_lst, double *f_axby,
double *fr_axby, cmplx *f_lqlqp, cmplx *f_lmqlqp, cmplx *f_lqlmqp, cmplx *f_lmqlmqp) {
    /* internal variables */
    const int i = threadIdx.x + blockDim.x * blockIdx.x;
    const int j = threadIdx.y + blockDim.y * blockIdx.y;
    const int k = threadIdx.z + blockDim.z * blockIdx.z;
    const int idx = i + j * blockDim.x * gridDim.x + k * blockDim.x * gridDim.x * blockDim.y * gridDim.y;
    const int iqlx= threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
    const int jx= blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
    /* local (q',l') pair */
    const int iqp = qp_lst[iqlx];
    const int ilp = ilp_lst[iqlx];
    /* jby index */
    const int jby = jaxby_lst[jx];
    const int jrby= jbyr_lst[jx];
    cmplx F(0.,0.);
    /* compute effective force */
    if (jby > -1) {
        F += f_axby[jx];
    }
    if (jrby > -1) {
        F += fr_axby[idx];
    }
    /* start calculation on the thread */
    if (iqlx < nqlp && jx < nby) {
        /* mass Mb */
        Mb = m_lst[jby];
        /* atom's index */
        ib = jby / 3;
        /* e^iq'Rb */
        qpRb = qv_lst[iqp*3]*r_lst[ib*3];
        qpRb+= qv_lst[iqp*3+1]*r_lst[ib*3+1];
        qpRb+= qv_lst[iqp*3+2]*r_lst[ib*3+2];
        re = cos(2.*PI*qpRb);
        im = sin(2.*PI*qpRb);
        cmplx eiqpRb(re, im);
        /* compute force */
        f_lqlqp[idx] += F * eiqpRb * euqlp[3*nat*iqlx+jby] / sqrt(Mb);
    }
}

/*

Raman function calculation 

*/
__global__ void compute_raman_force(int qs0, int qs1, int nqs, int iax, int *iFby_lst, int nby, int *qlp_lst,
double wql, double *wqlp, int size, cmplx *Fjax, double *eig, int calc_typ, cmplx *Fr) {
    /* internal variables */
    const int i = threadIdx.x + blockDim.x * blockIdx.x;
    const int j = threadIdx.y + blockDim.y * blockIdx.y;
    const int k = threadIdx.z + blockDim.z * blockIdx.z;
    const int idx = i + j * blockDim.x * gridDim.x + k * blockDim.x * gridDim.x * blockDim.y * gridDim.y;
    const int iqlx= threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
    const int jjby= blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
    // local variables
    int ms;
    int iby = iFby_lst[jjby];
    /* iqlx < size and jx < ndof */
    if (iqlx < size && jjby < nby) {
        /* if deph/rel */
        if (calc_typ == 0) {
            /* deph. calculation*/
            for (ms=0; ms<nqs; ms++) {
                Fr[idx] += Fjax[iax+qs0*nqs+ms] * Fjax[iby+ms*nqs+qs0] / (eig[qs0] - eig[ms] + wql);
                Fr[idx] -= Fjax[iax+qs1*nqs+ms] * Fjax[iby+ms*nqs+qs1] / (eig[qs1] - eig[ms] + wql);
                Fr[idx] += Fjax[iax+qs0*nqs+ms] * Fjax[iby+ms*nqs+qs0] / (eig[qs0] - eig[ms] - wqlp[iqlx]);
                Fr[idx] -= Fjax[iax+qs1*nqs+ms] * Fjax[iby+ms*nqs+qs1] / (eig[qs1] - eig[ms] - wqlp[iqlx]);
            }
        }
        else {
            /* relax calculation */
            for (ms=0; ms<nqs; ms++) {
                Fr[idx] += Fjax[iax+qs0*nqs+ms] * Fjax[iby+ms*nqs+qs1] / (eig[qs1] - eig[ms] + wqlp[iqlx]);
                Fr[idx] += Fjax[iax+qs0*nqs+ms] * Fjax[iby+ms*nqs+qs1] / (eig[qs1] - eig[ms] - wql);
            }
        }
    }
}