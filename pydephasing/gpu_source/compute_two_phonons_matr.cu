#include <cstdio>
#include <math.h>
#include <cuComplex.h>

/*
    compute g_qqp
*/

__global__ void compute_gqqp_1st_Raman(int nst, int nmd, int *INIT_INDEX, int *SIZE_LIST, int *MODES_LIST,
                                       double *WQL, double *WQPL, double *EIG,
                                       cuDoubleComplex *GQL,
                                       cuDoubleComplex *GQPL,
                                       cuDoubleComplex *GQQP)
{
    /* internal variables */
    const int i = threadIdx.x + blockDim.x * blockIdx.x;
    const int j = threadIdx.y + blockDim.y * blockIdx.y;
    const int k = threadIdx.z + blockDim.z * blockIdx.z;
    const int idx = i + j * blockDim.x * gridDim.x + k * blockDim.x * gridDim.x * blockDim.y * gridDim.y;
    const int sx = SIZE_LIST[idx];
    const int i0x = INIT_INDEX[idx];
    /* cycle over modes pairs */
    for (int ix = i0x; ix < i0x + sx; ix++)
    {
        /* modes pair indexes */
        int IL = MODES_LIST[2 * ix];
        int ILP = MODES_LIST[2 * ix + 1];
        // if (ix == 0)
        //{
        // printf("%d      %d       %d\n", IL, ILP, nst);
        //}
        for (int a = 0; a < nst; a++)
        {
            for (int ap = 0; ap < nst; ap++)
            {
                size_t INDG = ILP + IL * nmd + ap * nmd * nmd + a * nst * nmd * nmd;
                size_t INDG_M = 2 * INDG;
                size_t INDG_P = 2 * INDG + 1;
                /* b iteration */
                for (int b = 0; b < nst; b++)
                {
                    size_t ING = IL + b * nmd + a * nst * nmd;
                    size_t INGP = ILP + ap * nmd + b * nst * nmd;
                    // G^-(q,q')
                    cuDoubleComplex gq_m = GQL[ING];
                    cuDoubleComplex gqp_m = GQPL[INGP];
                    double denom =
                        (1.0 / (-WQL[IL] - EIG[b] + EIG[a])) +
                        (1.0 / (-WQPL[ILP] - EIG[b] + EIG[ap]));
                    GQQP[INDG_M] = cuCadd(
                        GQQP[INDG_M],
                        cuCmul(cuCmul(gq_m, gqp_m), make_cuDoubleComplex(denom, 0.0)));
                    // G^+(q,q')
                    cuDoubleComplex gq_p = cuConj(gq_m);
                    cuDoubleComplex gqp_p = cuConj(gqp_m);
                    // if (ix == 0)
                    //{
                    //     printf("(%d, %d, %d, %d, %e, %e, %e, %e, %e)\n", IL, b, a, ap, cuCreal(gq_m), cuCimag(gq_m), cuCreal(gq_p), cuCimag(gq_p), cuCreal(GQQP[INDG_M]));
                    // }
                    denom =
                        (1.0 / (WQL[IL] - EIG[b] + EIG[a])) +
                        (1.0 / (WQPL[ILP] - EIG[b] + EIG[ap]));
                    GQQP[INDG_P] = cuCadd(
                        GQQP[INDG_P],
                        cuCmul(cuCmul(gq_p, gqp_p), make_cuDoubleComplex(denom, 0.0)));
                }
            }
        }
    }
}

/*
    compute g_qqp with second order Raman term
*/

__global__ void compute_gqqp_2nd_Raman(int nat, int nst, int nmd, int *INIT_INDEX, int *SIZE_LIST, int *MODES_LIST,
                                       double *AQL, double *AQPL,
                                       cuDoubleComplex *FXY,
                                       cuDoubleComplex *EQ,
                                       cuDoubleComplex *EQP,
                                       cuDoubleComplex *EIQR,
                                       cuDoubleComplex *EIQPR,
                                       cuDoubleComplex *GQQP)
{
    /* internal variables */
    const int i = threadIdx.x + blockDim.x * blockIdx.x;
    const int j = threadIdx.y + blockDim.y * blockIdx.y;
    const int k = threadIdx.z + blockDim.z * blockIdx.z;
    const int idx = i + j * blockDim.x * gridDim.x + k * blockDim.x * gridDim.x * blockDim.y * gridDim.y;
    const int sx = SIZE_LIST[idx];
    const int i0x = INIT_INDEX[idx];
    /* cycle over mode pairs */
    for (int ix = i0x; ix < i0x + sx; ix++)
    {
        /* modes index */
        int IL = MODES_LIST[2 * ix];
        int ILP = MODES_LIST[2 * ix + 1];
        /* states index (a,a' )*/
        for (int a = 0; a < nst; a++)
        {
            for (int ap = 0; ap < nst; ap++)
            {
                size_t INDG = ILP + IL * nmd + ap * nmd * nmd + a * nst * nmd * nmd;
                size_t INDG_M = 2 * INDG;
                size_t INDG_P = 2 * INDG + 1;
                for (int jax = 0; jax < 3 * nat; jax++)
                {
                    cuDoubleComplex eq_1 = EQ[IL + jax * nmd];
                    for (int jby = 0; jby < 3 * nat; jby++)
                    {
                        cuDoubleComplex eq_2 = EQP[ILP + jby * nmd];
                        size_t INDXXP = jby + jax * 3 * nat + ap * 9 * nat * nat + a * nst * 9 * nat * nat;
                        /* Gqqp^- */
                        cuDoubleComplex fact_1 = make_cuDoubleComplex(0.5 * AQL[IL] * AQPL[ILP], 0.0);
                        cuDoubleComplex fact_2 = cuCmul(fact_1, eq_1);
                        cuDoubleComplex fact_3 = cuCmul(fact_2, EIQR[jax]);
                        cuDoubleComplex fact_4 = cuCmul(fact_3, FXY[INDXXP]);
                        cuDoubleComplex fact_5 = cuCmul(fact_4, EIQPR[jby]);
                        cuDoubleComplex fact_6 = cuCmul(fact_5, eq_2);
                        GQQP[INDG_M] = cuCadd(
                            GQQP[INDG_M],
                            fact_6);
                        /* Gqqp^+ -> c.c. */
                        fact_2 = cuCmul(fact_1, cuConj(eq_1));
                        fact_3 = cuCmul(fact_2, cuConj(EIQR[jax]));
                        fact_4 = cuCmul(fact_3, cuConj(FXY[INDXXP]));
                        fact_5 = cuCmul(fact_4, cuConj(EIQPR[jby]));
                        fact_6 = cuCmul(fact_5, cuConj(eq_2));
                        GQQP[INDG_P] = cuCadd(
                            GQQP[INDG_P],
                            fact_6);
                    }
                }
            }
        }
    }
    if (idx == 0)
    {
        printf("%d   -> END\n", i0x);
    }
}