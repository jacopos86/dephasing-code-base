#include <pycuda-complex.hpp>
#include <math.h>
typedef pycuda::complex<double> cmplx;

/* compute phr forces */
__global__ void compute_Flq_lqp(int *qp_lst, int *ilp_lst, int *jax_lst, const int size, const int nax, const int nat,
double wql, double *wqlp, cmplx *euq, cmplx *euqlp, double *r_lst, double *qv_lst, double *m_lst,
cmplx *eiqR, const int nqs, double *eig, cmplx *Fax, cmplx *Faxby, cmplx *F_lqlqp,
cmplx *F_lmqlqp, cmplx *F_lqlmqp, cmplx *F_lmqlmqp) {
    /* internal variables */
    const int i = threadIdx.x + blockDim.x * blockIdx.x;
    const int j = threadIdx.y + blockDim.y * blockIdx.y;
    const int k = threadIdx.z + blockDim.z * blockIdx.z;
    const int idx = i + j * blockDim.x * gridDim.x + k * blockDim.x * gridDim.x * blockDim.y * gridDim.y;
    const int iqlx= threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
    const int jx  = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
    /* local (q',l') pair */
    const int iqp = qp_lst[iqlx];
    const int ilp = ilp_lst[iqlx];
    /* jax */
    const int jax = jax_lst[jx];
    /* check if thread is allowed to compute */
    if (iqlx < size && jx < nax) {
        /* set local Q vector */
        for (id=0; id<3; id++) {
            qpv[id] = qv_lst[iqp*3+id];
        }
        /* set Fjax force */
        i = 0;
        for (msr=0; msr<nqs; msr++) {
            for (msc=0; msc<nqs; msc++) {
                Fjax[i] = Fax[jax*nqs*nqs+msr*nqs+msc];
                i += 1;
            }
        }
        /* run over jby index */
        for (jby=0; jby<3*nat; jby++) {
            ib = jby / 3;
            /* atom coordinates*/
            for (id=0; id<3; id++) {
                R[id] = 0.0;
                R[id] = r_lst[ib*3+id];
            }
            /* atom mass */
            Mb = m_lst[jby];
            /*e^iq'R*/
            qpR = qpv[0]*R[0] + qpv[1]*R[1] + qpv[2]*R[2];
            re = cos(2.*PI*qpR);
            im = sin(2.*PI*qpR);
            cmplx eiqpR(re, im);
            /* eff. force */
            F = Faxby[3*nat*jax+jby];
            i = 0;
            for (msr=0; msr<nqs; msr++) {
                for (msc=0; msc<nqs; msc++) {
                    Fjby[i] = Fax[jby*nqs*nqs+msr*nqs+msc];
                    i += 1;
                }
            }
            if (calc_raman) {
                Fr= compute_raman_force(qs0, qs1, nqs, Fjax, Fjby, eig, wql, wqlp[idx]);
                F += Fr;
            }
            F_lqlqp[idx] += F * eiqpR * euqlp[3*nat*iqlx+jby] / SQRT (Mb);
            F_lqlmqp[idx]+= F * eiqpR * euqlp[3*nat*iqlx+jby] / SQRT (Mb);
            F_lmqlqp[idx]+= F * conj(eiqpR) * conj(euqlp[3*nat*iqlx+jby]) / SQRT (Mb);
            F_lmqlmqp[idx]+= F * conj(eiqpR) * conj(euqlp[3*nat*iqlx+jby]) / SQRT (Mb);
        }
        /* multiply with l.h.s*/
        F_lqlqp[idx] = eiqR[jax] * euq[jax] / SQRT (m_lst[jax]) * F_lqlqp[idx];
        F_lmqlqp[idx]= conj(eiqR[jax]) * conj(euq[jax]) / SQRT (m_lst[jax]) * F_lmqlqp[idx];
        F_lqlmqp[idx] = eiqR[jax] * euq[jax] / SQRT (m_lst[jax]) * F_lqlmqp[idx];
        F_lmqlmqp[idx]= conj(eiqR[jax]) * conj(euq[jax]) / SQRT (m_lst[jax]) * F_lmqlmqp[idx];
    }
}

/* Raman function calculation */
__device__ cmplx compute_raman_force(int qs0, int qs1, int nqs, cmplx *Fjax, cmplx *Fjby,
double *eig, double wql, double wqlp, int calc_type) {
    cmplx Fr(0.,0.);
    if (calc_type == 0) {
        /* deph calculation*/
        for (ms=0; ms<nqs; ms++) {
            Fr += Fjax[qs0*nqs+ms] * Fjby[ms*nqs+qs0] / (eig[qs0] - eig[ms] + wql);
            Fr -= Fjax[qs1*nqs+ms] * Fjby[ms*nqs+qs1] / (eig[qs1] - eig[ms] + wql);
            Fr += Fjax[qs0*nqs+ms] * Fjby[ms*nqs+qs0] / (eig[qs0] - eig[ms] - wqlp);
            Fr -= Fjax[qs1*nqs+ms] * Fjby[ms*nqs+qs1] / (eig[qs1] - eig[ms] - wqlp);
        }
    }
    else {
        /* relax calculation */
        for (ms=0; ms<nqs; ms++) {
            Fr += Fjax[qs0*nqs+ms] * Fjby[ms*nqs+qs1] / (eig[qs1] - eig[ms] + wqlp);
            Fr += Fjax[qs0*nqs+ms] * Fjby[ms*nqs+qs1] / (eig[qs1] - eig[ms] - wql);
        }
    }
    return Fr;
}