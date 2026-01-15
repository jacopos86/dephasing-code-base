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
                GQQP[INDG_M] = make_cuDoubleComplex(0.0, 0.0);
                GQQP[INDG_P] = make_cuDoubleComplex(0.0, 0.0);
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
        // if (i0x == 0)
        //{
        //     printf("%d   -> END\n", ix);
        // }
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
    //    for (int ix = i0x; ix < i0x + sx; ix++)
    //    {
    //        /* modes index */
    //        int IL = MODES_LIST[2 * ix];
    //        int ILP = MODES_LIST[2 * ix + 1];
    //        /* states index (a,a' )*/
    //        for (int a = 0; a < nst; a++)
    //        {
    //            for (int ap = 0; ap < nst; ap++)
    //            {
    //                size_t INDG = ILP + IL * nmd + ap * nmd * nmd + a * nst * nmd * nmd;
    //                GQQP[INDG] = (0, 0);
    //                for (int jax = 0; jax < 3 * nat; jax++)
    //                {
    //                    cmplx eq_1 = EQ[IL + jax * nmd];
    //                    for (int jby = 0; jby < 3 * nat; jby++)
    //                    {
    //                        cmplx eq_2 = EQP[ILP + jby * nmd];
    //                        for (int n1 = 0; n1 < nl; n1++)
    //                        {
    //                            for (int n2 = 0; n2 < nl; n2++)
    //                            {
    //                                cmplx F = (0, 0);
    //                                /* compute force */
    //                                for (int b = 0; b < nst; b++)
    //                                {
    //                                    int INX = jax + ap * 3 * nat + b * nst * 3 * nat;
    //                                    int INXP = jby + b * 3 * nat + a * nst * 3 * nat;
    //                                    F += FX[INXP] * FX[INX] * (1 / (-WQPL[ILP] - EIG[b] + EIG[a]) + 1 / (-WQL[IL] - EIG[b] + EIG[ap]));
    //                                    INX = jax + b * 3 * nat + a * nst * 3 * nat;
    //                                    INXP = jby + ap * 3 * nat + b * nst * 3 * nat;
    //                                    F += FX[INX] * FX[INXP] * (1 / (WQL[IL] - EIG[b] + EIG[a]) + 1 / (WQPL[ILP] - EIG[b] + EIG[ap]));
    //                                }
    //                                int INXXP = jby + jax * 3 * nat + ap * 9 * nat * nat + a * nst * 9 * nat * nat;
    //                                F += FXY[INXXP];
    //                                GQQP[INDG] += 0.5 * AQL[IL] * eq_1 * EIQR[n1] * F * EIQPR[n2] * eq_2 * AQPL[ILP];
    //                            }
    //                        }
    //                    }
    //                }
    //            }
    //        }
    //    }
    if (idx == 0)
    {
        printf("%d   -> END\n", i0x);
    }
}