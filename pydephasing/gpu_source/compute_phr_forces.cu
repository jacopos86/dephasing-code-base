#include <pycuda-complex.hpp>
#include <math.h>
typedef pycuda::complex<double> cmplx;

/* compute phr forces */
__global__ void compute_Flq_lqp(int *qp_lst, int *ilp_lst, int *jax_lst, const int size, const int nax, double wql, double *wqlp,
cmplx *euq, cmplx *euqlp, double *r_lst, double *qv_lst, double *m_lst,
cmplx *eiqr, const int nqs, double *eig, double *Fax, double *Faxby, cmplx *F_lqlqp,
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
                    Fjax[i] = Fax[jax*nqs*nqs+msr*nqs+msc];
                    Fjby[i] = Fax[jby*nqs*nqs+msr*nqs+msc];
                    i += 1;
                }
            }
            if (calc_raman) {
                Fr= compute_raman_force(qs0, qs1, nqs, Fjax, Fjby, eig, wql, wqlp[idx]);
                F += Fr;
            }
        }




    }


}