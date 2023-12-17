#include <pycuda-complex.hpp>
#include <math.h>
#include <cuComplex.h>
typedef pycuda::complex<double> cmplx;
#define PI 3.141592653589793

/* compute phr forces */
//__global__ void compute_Flq_lqp(int *qp_lst, int *ilp_lst, int *jax_lst, const int size, const int nax, const int nat,
//double wql, double *wqlp, cmplx *euq, cmplx *euqlp, double *r_lst, double *qv_lst, double *m_lst, cmplx *eiqR, 
//const int nqs, const int qs0, const int qs1, double *eig, cmplx *Fax, const int calc_raman, const int calc_typ,
//cmplx *F_lqlqp) {
    // cmplx *Fax
    // cmplx *Faxby
    // cmplx *F_lmqlqp, cmplx *F_lqlmqp, cmplx *F_lmqlmqp
    /* internal variables */
//    const int i = threadIdx.x + blockDim.x * blockIdx.x;
//    const int j = threadIdx.y + blockDim.y * blockIdx.y;
//    const int k = threadIdx.z + blockDim.z * blockIdx.z;
//    const int idx = i + j * blockDim.x * gridDim.x + k * blockDim.x * gridDim.x * blockDim.y * gridDim.y;
//    const int iqlx= threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
//    const int jx  = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
//    int jby, ib;
//    int ms, i0m, i1m, im0, im1;
//    double re, im, qpR;
//    double Ma, Mb;
//    cmplx F;
    /* local (q',l') pair */
//    const int iqp = qp_lst[iqlx];
//    const int ilp = ilp_lst[iqlx];
    /* jax */
//    const int jax = jax_lst[jx];
    /* check if thread is allowed to compute */
//    if (iqlx < size && jx < nax) {
        /* Ma */
//        Ma = m_lst[jax];
        /* run over jby index */
//        for (jby=0; jby<3*nat; jby++) {
//            ib = jby / 3;
            /* atom mass */
//            Mb = m_lst[jby];
            /*e^iq'R*/
//            qpR  = qv_lst[iqp*3]*r_lst[ib*3];
//            qpR += qv_lst[iqp*3+1]*r_lst[ib*3+1];
//            qpR += qv_lst[iqp*3+2]*r_lst[ib*3+2];
//            re = cos(2.*PI*qpR);
//            im = sin(2.*PI*qpR);
//            cmplx eiqpR(re, im);
            /*       eff. force                   */
            //  F = Faxby[3*nat*jax+jby];
            /* ---------------------------------- */
            /*     Raman calculation              */
            /* ---------------------------------- */
//            if (calc_raman == 1) {
//                cmplx Fr(0.,0.);
//                if (calc_typ == 0) {
                    /* deph calc. */
 //                   for (ms=0; ms<nqs; ms++) {
//                        i0m = jax*nqs*nqs+qs0*nqs+ms;
//                        i1m = jax*nqs*nqs+qs1*nqs+ms;
//                        im0 = jby*nqs*nqs+ms*nqs+qs0;
//                        im1 = jby*nqs*nqs+ms*nqs+qs1;
                    //    Fr += Fax[i0m] * Fax[im0] / (eig[qs0]-eig[ms]+wql);
                    //    Fr -= Fax[i1m] * Fax[im1] / (eig[qs1]-eig[ms]+wql);
                    //    Fr += Fax[i0m] * Fax[im0] / (eig[qs0]-eig[ms]-wqlp[iqlx]);
                    //    Fr -= Fax[i1m] * Fax[im1] / (eig[qs1]-eig[ms]-wqlp[iqlx]);
//                    }
//                }
//                else if (calc_typ == 1) {
                    /* relax calc. */
//                    for (ms=0; ms<nqs; ms++) {
//                        i0m = jax*nqs*nqs+qs0*nqs+ms;
//                        im1 = jby*nqs*nqs+ms*nqs+qs1;
                    //    Fr += Fax[i0m] * Fax[im1] / (eig[qs1]-eig[ms]+wqlp[iqlx]);
                    //    Fr += Fax[i0m] * Fax[im1] / (eig[qs1]-eig[ms]-wql);
//                    }
//                }
//                F += Fr;
//            }
//            F_lqlqp[idx] += F * eiqpR * euqlp[3*nat*iqlx+jby] / sqrt (Mb);
            //F_lqlmqp[idx]+= F * eiqpR * euqlp[3*nat*iqlx+jby] / sqrt (Mb);
            //F_lmqlqp[idx]+= F * conj(eiqpR) * conj(euqlp[3*nat*iqlx+jby]) / sqrt (Mb);
            //F_lmqlmqp[idx]+= F * conj(eiqpR) * conj(euqlp[3*nat*iqlx+jby]) / sqrt (Mb);
//        }
        /* multiply with l.h.s*/
//        F_lqlqp[idx] = eiqR[jax] * euq[jax] / sqrt (Ma) * F_lqlqp[idx];
        //F_lmqlqp[idx]= conj(eiqR[jax]) * conj(euq[jax]) / sqrt (Ma) * F_lmqlqp[idx];
        //F_lqlmqp[idx] = eiqR[jax] * euq[jax] / sqrt (Ma) * F_lqlmqp[idx];
        //F_lmqlmqp[idx]= conj(eiqR[jax]) * conj(euq[jax]) / sqrt (Ma) * F_lmqlmqp[idx];
//    }
//}

/*

Flq_lqp force calculation

*/

__global__ void compute_Flqlqp(int nat, int nby, int nqlp, int *qp_lst, int *qlp_lst, 
int *jby_lst, cmplx *euqlp, double *r_lst, double *qv_lst, double *m_lst, double *f_axby,
cmplx *f_lqlqp, cmplx *f_lmqlqp, cmplx *f_lqlmqp, cmplx *f_lmqlmqp) {
    /* internal variables */
    const int i = threadIdx.x + blockDim.x * blockIdx.x;
    const int j = threadIdx.y + blockDim.y * blockIdx.y;
    const int k = threadIdx.z + blockDim.z * blockIdx.z;
    const int idx = i + j * blockDim.x * gridDim.x + k * blockDim.x * gridDim.x * blockDim.y * gridDim.y;
    const int iqlx= threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
    const int jx= blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
    /* local variables */
    double Mb, qpRb;
    double re, im;
    cmplx r1(0.,0.);
    cmplx r2(0.,0.);
    int ib, jby;
    int iqlp, iqp;
    /* local (q',l') pair */
    if (iqlx < nqlp) {
        iqlp = qlp_lst[iqlx];
        iqp  = qp_lst[iqlp];
    }
    /* jby index */
    if (jx < nby) {
        jby = jby_lst[jx];
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
        r1 = f_axby[jx] * eiqpRb * euqlp[3*nat*iqlp+jby] / sqrt(Mb);
        f_lqlqp[idx]  += r1;
        f_lmqlqp[idx] += r1;
        r2 = f_axby[jx] * conj(eiqpRb) * conj(euqlp[3*nat*iqlp+jby]) / sqrt(Mb);
        f_lqlmqp[idx] += r2;
        f_lmqlmqp[idx]+= r2;
    }
}

/*

Flq_lqp Raman force calculation

*/

__global__ void compute_Flqlqp_raman(int nat, int nby, int nqlp, cmplx *euqlp, double *r_lst, 
double *qv_lst, double *m_lst, int *qp_lst, int *qlp_lst, int *faxby_ind, int *jby_lst, 
cmplx *fr, double *f_axby, cmplx *f_lqlqp, cmplx *f_lmqlqp, cmplx *f_lqlmqp, cmplx *f_lmqlmqp) {
    /* internal variables */
    const int i = threadIdx.x + blockDim.x * blockIdx.x;
    const int j = threadIdx.y + blockDim.y * blockIdx.y;
    const int k = threadIdx.z + blockDim.z * blockIdx.z;
    const int idx = i + j * blockDim.x * gridDim.x + k * blockDim.x * gridDim.x * blockDim.y * gridDim.y;
    const int iqlx= threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
    const int jx= blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
    int iqlp, iqp;
    int jby, ib;
    double Mb;
    double qpRb, re, im;
    cmplx r1(0., 0.);
    cmplx r2(0., 0.);
    /* local (q',l') pair */
    if (iqlx < nqlp) {
        iqlp= qlp_lst[iqlx];
        iqp = qp_lst[iqlp];
    }
    // effective force
    cmplx F(0.,0.);
    if (jx < nby) {
        /* jby index */
        jby= jby_lst[jx];
        /* compute eff. force */
        F = fr[idx];
        if (faxby_ind[jx] > -1) {
            F += f_axby[faxby_ind[jx]];
        }
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
        r1 = F * eiqpRb * euqlp[3*nat*iqlp+jby] / sqrt(Mb);
        f_lqlqp[idx]  += r1;
        f_lmqlqp[idx] += r1;
        r2 = F * conj(eiqpRb) * conj(euqlp[3*nat*iqlp+jby]) / sqrt(Mb);
        f_lqlmqp[idx] += r2;
        f_lmqlmqp[idx]+= r2;
    }
}

/*

Raman function calculation 

*/
__global__ void compute_raman_force(int qs0, int qs1, int nqs, int iax, int *iFby_ind, int nby, int *qlp_lst,
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
    int iby;
    int iqlp;
    /* local jby index */
    if (jjby < nby) {
        iby = iFby_ind[jjby];
    }
    /* iqlx < size and jx < ndof */
    if (iqlx < size && jjby < nby && iby > -1) {
        /* set local qlp index */
        iqlp = qlp_lst[iqlx];
        /* if deph/rel */
        if (calc_typ == 0) {
            /* deph. calculation*/
            for (ms=0; ms<nqs; ms++) {
                Fr[idx] += Fjax[iax+qs0*nqs+ms] * Fjax[iby+ms*nqs+qs0] / (eig[qs0] - eig[ms] + wql);
                Fr[idx] -= Fjax[iax+qs1*nqs+ms] * Fjax[iby+ms*nqs+qs1] / (eig[qs1] - eig[ms] + wql);
                Fr[idx] += Fjax[iax+qs0*nqs+ms] * Fjax[iby+ms*nqs+qs0] / (eig[qs0] - eig[ms] - wqlp[iqlp]);
                Fr[idx] -= Fjax[iax+qs1*nqs+ms] * Fjax[iby+ms*nqs+qs1] / (eig[qs1] - eig[ms] - wqlp[iqlp]);
            }
        }
        else {
            /* relax calculation */
            for (ms=0; ms<nqs; ms++) {
                Fr[idx] += Fjax[iax+qs0*nqs+ms] * Fjax[iby+ms*nqs+qs1] / (eig[qs1] - eig[ms] + wqlp[iqlp]);
                Fr[idx] += Fjax[iax+qs0*nqs+ms] * Fjax[iby+ms*nqs+qs1] / (eig[qs1] - eig[ms] - wql);
            }
        }
    }
}