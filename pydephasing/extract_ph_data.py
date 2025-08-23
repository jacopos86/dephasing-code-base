# This routine extract the phonon data information
# and return it for processing
import numpy as np
import h5py
from pydephasing.set_param_object import p
from common.phys_constants import eps, THz_to_ev
from common.special_functions import lorentzian
from pydephasing.log import log
from parallelization.mpi import mpi
import logging
#
# set ql' list
def set_ql_list_red_qgrid(qpts, nat, wu):
    nq = len(qpts)
    # q -> -q map
    qmq_list = set_q_to_mq_list(qpts, nq)
    # only q>0 list
    qlp_list = []
    qmq_map = []
    for iqpair1 in qmq_list:
        iq1 = iqpair1[0]
        for il in range(3*nat):
            if wu[iq1][il] > p.min_freq:
                qlp_list.append((iq1,il))
        miq1 = iqpair1[1]
        qmq_map.append((iq1,miq1))
    # make dict q -> -q
    qmq_map = dict(qmq_map)
    # make local ql_list
    ql_list = mpi.split_list(qlp_list)
    return ql_list, qlp_list, qmq_map
#
# list of (l',q') pairs
def set_iqlp_list(il, iq, qlp_list_full, wu, H):
    T = np.max(p.temperatures)
    # dE
    dE = 0.0
    if p.relax:
        # quantum states
        iqs0 = p.index_qs0
        iqs1 = p.index_qs1
        dE = H.eig[iqs1] - H.eig[iqs0]
        # eV
    # set w_ql (eV)
    E_ql = wu[iq][il] * THz_to_ev
    nql = bose_occup(E_ql, T)
    # if freq. resolved calculation
    if p.w_resolved:
        ltz_max = 0.
        for iqp, ilp in qlp_list_full:
            if iqp != iq or ilp != il:
                E_qlp = wu[iqp][ilp] * THz_to_ev
                # eV
                x = dE + E_qlp - E_ql
                if lorentzian(x, p.eta) > ltz_max:
                    ltz_max = lorentzian(x, p.eta)
    # run over (q',l')
    qlp_list = []
    for iqp, ilp in qlp_list_full:
        if iqp != iq or ilp != il:
            E_qlp = wu[iqp][ilp] * THz_to_ev
            # E (eV)
            nqlp = bose_occup(E_qlp, T)
            A_th = nql * (1. + nqlp)
            if p.w_resolved:
                x = dE + E_qlp - E_ql
                if lorentzian(x, p.eta) / ltz_max > p.lorentz_thres and A_th > 1.E-7:
                    #print(dE, E_ql, E_qlp, x, lorentzian(x, p.eta))
                    qlp_list.append((iqp,ilp))
            else:
                if A_th > 1.E-7:
                    qlp_list.append((iqp,ilp))
    return qlp_list