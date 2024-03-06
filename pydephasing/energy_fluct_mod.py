#  This module defines the auto correlation
#  functions to be used in the different
#  calculations
#
import numpy as np
from pydephasing.mpi import mpi
from pydephasing.log import log
from pydephasing.input_parameters import p
from pydephasing.phys_constants import THz_to_ev
from pydephasing.utility_functions import bose_occup
from pydephasing.extract_ph_data import set_q_to_mq_list
from pydephasing.ph_resolved_quant import compute_ph_amplitude_q
#
# energy levels fluctuations
class energy_level_fluctuations_oft:
    # initialization
    def __init__(self):
        self.deltaE_oft = np.zeros((p.nt,p.ntmp))
    # compute energy fluctuations
    def compute_deltaE_oft(self, wq, wu, ql_list, A_lq, F_lq):
        #
        # compute deltaE(t)
        #
        iql = 0
        # iterate over ph. modes
        for iq, il in ql_list:
            if mpi.rank == mpi.root:
                log.debug(str(il) + ',' + str(iq) + ' -> ' + str(len(ql_list)))
            wuq = wu[iq]
            if wuq[il] > p.min_freq:
                # Eql
                Eql = wuq[il] * THz_to_ev
                wql = 2.*np.pi*wuq[il]
                # time part
                eiwt = np.zeros(p.nt, dtype=np.complex128)
                eiwt[:] = np.exp(1j*wql*p.time[:])
                # run over the temperature list
                for iT in range(p.ntmp):
                    T = p.temperatures[iT]
                    # bose occup.
                    nph = bose_occup(Eql, T)
                    ft = np.zeros(p.nt, dtype=np.complex128)
                    ft[:] = (1.+nph) * eiwt[:] + nph * eiwt[:].conjugate()
                    # compute energy fluct.
                    self.deltaE_oft[:,iT] += wq[iq] * A_lq[iql] ** 2 * ft[:] * F_lq[iql] * F_lq[iql].conjugate()
                    # eV units
                    #
            iql += 1
#
# static spin fluctuations class
class ZFS_ph_fluctuations:
    # expression 2nd order ZFS energy
    # fluctuations
    # dV = \sum_lambda \sum_q w(q) A_lambda(q) A_lambda(-q) (1+2n_lambda(q)) Delta F_lambda,q;lambda,-q
    def __init__(self):
        self.dE_eV = 0.
        # eV units
    # set (q,-q,l) transitions
    # list
    def set_ql_trans_list(self, qpts, nat, wu):
        nq = len(qpts)
        # q -> -q map
        qmq_list = set_q_to_mq_list(qpts, nq)
        qql_list = []
        for iqpair in qmq_list:
            iq1 = iqpair[0]
            iq2 = iqpair[1]
            for il in range(3*nat):
                if wu[iq1][il] > p.min_freq and wu[iq2][il] > p.min_freq:
                    qql_list.append((iq1,iq2,il))
        # split list between procs.
        qql_list = mpi.split_list(qql_list)
        return qql_list
    # compute eff. force
    def transf_2nd_order_force_phr(self, wu, u, nat, qql_list):
        return None
    # compute energy 
    # fluctuations
    def compute_fluctuations(self, qpts, nat, wu, u):
        # first set transitions list
        qql_list = self.set_ql_trans_list(qpts, nat, wu)
        # compute amplitudes
        ql_list = []
        for iqql in qql_list:
            iq = iqql[0]
            il = iqql[2]
            ql_list.append((iq,il))
        A_lq = compute_ph_amplitude_q(wu, nat, ql_list)
        # compute eff. forces
        F_lqqp = self.transf_2nd_order_force_phr(wu, u, nat, qql_list)