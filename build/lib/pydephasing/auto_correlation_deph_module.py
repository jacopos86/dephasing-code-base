#
# This module sets up the methods
# needed to compute the energy fluctuations
# auto correlation function
#
import numpy as np
import cmath
import math
from tqdm import tqdm
from pydephasing.phys_constants import THz_to_ev, eps, kb
from pydephasing.input_parameters import p
from pydephasing.atomic_list_struct import atoms
from pydephasing.utility_functions import bose_occup, lorentzian
from pydephasing.global_params import GPU_ACTIVE
from pathlib import Path
from pydephasing.auto_correlation_driver import acf_ph
import matplotlib.pyplot as plt
import sys
# pycuda
if GPU_ACTIVE:
    from pycuda.compiler import SourceModule
    import pycuda.driver as cuda
    import pycuda.autoinit
    from pydephasing.global_params import gpu
#
# spin-exc-ph dephasing class -> homogeneous
class acf_ph_deph(acf_ph):
    def __init__(self):
        super(acf_ph_deph, self).__init__()
    #
    # acf V1 (w=0)
    def compute_acf_V1_w0(self, wq, wu, ql_list, A_lq, F_lq):
        # Delta_w0 = sum_l,q A_l,q^2 [1 + 2 n_lq] |F_lq|^2
        # eV units
        Delta_w0 = np.zeros(p.ntmp)
        # compute partial value
        iql = 0
        for iq, il in ql_list:
            wuq = wu[iq]
            if wuq[il] > p.min_freq:
                # E in eV
                E_ql = wuq[il] * THz_to_ev
                ltz  = lorentzian(E_ql, p.eta)
                # eV^-1
                # temperatures
                for iT in range(p.ntmp):
                    T = p.temperatures[iT]
                    # bose occ.
                    nph = bose_occup(E_ql, T)
                    Delta_w0[iT] += wq[iq] * A_lq[iql] ** 2 * (1.+2.*nph) * ltz * (F_lq[iql] * F_lq[iql].conjugate()).real
            iql += 1
        return Delta_w0
# ------------------------------------------------------------------------
#
#               CPU CLASS
#
# ------------------------------------------------------------------------
class CPU_acf_ph_deph(acf_ph_deph):
    def __init__(self):
        super(CPU_acf_ph_deph, self).__init__()
    #
    # compute <Delta V^(1)(t) Delta V^(1)(t')>
    def compute_acf_V1_oft(self, wq, wu, ql_list, A_lq, F_lq):
        # initialize acf_sp -> 0 acf -> 1 integral
        self.acf_sp = np.zeros((p.nt, 2, p.ntmp), dtype=np.complex128)
        # compute partial acf
        # acf(t) = \sum_q \sum_l
        iql = 0
        for iq, il in ql_list:
            wuq = wu[iq]
            if wuq[il] > p.min_freq:
                # E in (eV)
                Eql = wuq[il] * THz_to_ev
                wql = 2.*np.pi*wuq[il]
                # set e^{-iwt}
                exp_iwt = np.zeros(p.nt, dtype=np.complex128)
                cc_exp_iwt = np.zeros(p.nt, dtype=np.complex128)
                for t in range(p.nt):
                    exp_iwt[t] = cmath.exp(-1j*wql*p.time[t])
                    cc_exp_iwt[t] = cmath.exp(1j*wql*p.time[t])
                # run over temperatures
                for iT in range(p.ntmp):
                    T = p.temperatures[iT]
                    # bose occupation
                    nph = bose_occup(Eql, T)
                    ft = np.zeros(p.nt, dtype=np.complex128)
                    ft[:] = (1.+nph) * exp_iwt[:] + nph * cc_exp_iwt[:]
                    # (eV^2) units
                    self.acf_sp[:,0,iT] += wq[iq] * A_lq[iql] ** 2 * ft[:] * F_lq[iql] * F_lq[iql].conjugate()
                    # (eV^2 ps) units
                    ft[:] = 0.
                    ft[:] = (1.+nph) * (exp_iwt[:] - 1.)/(-1j*wql) + nph * (cc_exp_iwt[:] - 1.)/(1j*wql)
                    self.acf_sp[:,1,iT] += wq[iq] * A_lq[iql] ** 2 * ft[:] * F_lq[iql] * F_lq[iql].conjugate()
            iql += 1
    #
    # compute <Delta V(1) \Delta V(1)>(w)
    def compute_acf_V1_ofw(self, wq, wu, ql_list, A_lq, F_lq):
        # initialize acf_sp -> 0 acf -> 1 integral
        self.acf_sp = np.zeros((p.nwg, p.ntmp), dtype=np.complex128)
        ltz = np.zeros(p.nwg)
        # compute partial acf \sum_ql
        iql = 0
        for iq, il in ql_list:
            wuq = wu[iq]
            if wuq[il] > p.min_freq:
                # E in eV
                Eql = wuq[il] * THz_to_ev
                ltz[:] = 0.
                for iw in range(p.nwg):
                    w = p.w_grid[iw]
                    # eV
                    ltz[iw] = lorentzian(Eql+w, p.eta)
                    # eV^-1
                # run over temperatures
                for iT in range(p.ntmp):
                    T = p.temperatures[iT]
                    # bose occ.
                    nph = bose_occup(Eql, T)
                    # eV/ps^2*eV*ps^2*eV^-1 = eV
                    self.acf_sp[:,iT] += wq[iq] * A_lq[iql] ** 2 * (1.+2.*nph) * ltz[:] * F_lq[iql] * F_lq[iql].conjugate()
            iql += 1
    #
    # compute <Delta V^(1)(t) Delta V^(1)(t')> -> ph / at resolved
    def compute_acf_V1_atphr_oft(self, nat, wq, wu, ql_list, A_lq, Fjax_lq):
        if p.ph_resolved:
            self.acf_phr_sp = np.zeros((p.nt2,2,p.nphr,p.ntmp), dtype=np.complex128)
            self.acf_wql_sp = np.zeros((p.nt2,2,p.nwbn,p.ntmp), dtype=np.complex128)
        if p.at_resolved:
            self.acf_atr_sp = np.zeros((p.nt2,2,nat,p.ntmp), dtype=np.complex128)
        if not p.ph_resolved and not p.at_resolved:
            return
        # compute partial acf
        iql = 0
        for iq, il in tqdm(ql_list):
            wuq = wu[iq]
            if wuq[il] > p.min_freq:
                # E in (eV)
                Eql = wuq[il] * THz_to_ev
                wql = 2.*np.pi*wuq[il]
                # set e^{-iwt}
                exp_iwt = np.zeros(p.nt2, dtype=np.complex128)
                cc_exp_iwt = np.zeros(p.nt2, dtype=np.complex128)
                for t in range(p.nt2):
                    exp_iwt[t] = cmath.exp(-1j*wql*p.time2[t])
                    cc_exp_iwt[t] = cmath.exp(1j*wql*p.time2[t])
                # run over temperatures
                for iT in range(p.ntmp):
                    T = p.temperatures[iT]
                    # bose occupation
                    nph = bose_occup(Eql, T)
                    ft = np.zeros(p.nt2, dtype=np.complex128)
                    ft[:] = (1.+nph) * exp_iwt[:] + nph * cc_exp_iwt[:]
                    for jax in range(3*nat):
                        # (eV^2) units
                        # ph. resolved
                        if p.ph_resolved:
                            ii = p.wql_grid_index[iq,il]
                            self.acf_wql_sp[:,0,ii,iT] += wq[iq] * A_lq[iql] ** 2 * ft[:] * Fjax_lq[jax,iql] * Fjax_lq[jax,iql].conjugate()
                            if il in p.phm_list:
                                iph = p.phm_list.index(il)
                                self.acf_phr_sp[:,0,iph,iT] += wq[iq] * A_lq[iql] ** 2 * ft[:] * Fjax_lq[jax,iql] * Fjax_lq[jax,iql].conjugate()
                        # at. resolved
                        if p.at_resolved:
                            ia = atoms.index_to_ia_map[jax] - 1
                            self.acf_atr_sp[:,0,ia,iT] += wq[iq] * A_lq[iql] ** 2 * ft[:] * Fjax_lq[jax,iql] * Fjax_lq[jax,iql].conjugate()
                    # integral
                    # (eV^2 ps) units
                    ft[:] = 0.
                    ft[:] = (1.+nph) * (exp_iwt[:] - 1.)/(-1j*wql) + nph * (cc_exp_iwt[:] - 1.)/(1j*wql)
                    for jax in range(3*nat):
                        # (eV^2) units
                        # ph. resolved
                        if p.ph_resolved:
                            ii = p.wql_grid_index[iq,il]
                            self.acf_wql_sp[:,1,ii,iT] += wq[iq] * A_lq[iql] ** 2 * ft[:] * Fjax_lq[jax,iql] * Fjax_lq[jax,iql].conjugate()
                            if il in p.phm_list:
                                iph = p.phm_list.index(il)
                                self.acf_phr_sp[:,1,iph,iT] += wq[iq] * A_lq[iql] ** 2 * ft[:] * Fjax_lq[jax,iql] * Fjax_lq[jax,iql].conjugate()
                        # at. resolved
                        if p.at_resolved:
                            ia = atoms.index_to_ia_map[jax] - 1
                            self.acf_atr_sp[:,1,ia,iT] += wq[iq] * A_lq[iql] ** 2 * ft[:] * Fjax_lq[jax,iql] * Fjax_lq[jax,iql].conjugate()
            iql += 1
    #
    # compute <Delta V(1) \Delta V(1)>(w) / at-ph res.
    def compute_acf_V1_atphr_ofw(self, nat, wq, wu, ql_list, A_lq, Fjax_lq):
        if p.ph_resolved:
            self.acf_phr_sp = np.zeros((p.nwg,p.nphr,p.ntmp), dtype=np.complex128)
            self.acf_wql_sp = np.zeros((p.nwg,p.nwbn,p.ntmp), dtype=np.complex128)
        if p.at_resolved:
            self.acf_atr_sp = np.zeros((p.nwg,nat,p.ntmp), dtype=np.complex128)
        if not p.ph_resolved and not p.at_resolved:
            return
        # lorentzian
        ltz = np.zeros(p.nwg)
        # compute partial acf
        iql = 0
        for iq, il in tqdm(ql_list):
            wuq = wu[iq]
            if wuq[il] > p.min_freq:
                # E in (eV)
                Eql = wuq[il] * THz_to_ev
                for iw in range(p.nwg):
                    w = p.w_grid[iw]
                    # eV
                    ltz[iw] = lorentzian(Eql+w, p.eta)
                    # eV^-1
                # run over temperatures
                for iT in range(p.ntmp):
                    T = p.temperatures[iT]
                    # bose occupation
                    nph = bose_occup(Eql, T)
                    for jax in range(3*nat):
                        # eV/ps^2*eV*ps^2*eV^-1 = eV
                        # ph. resolved
                        if p.ph_resolved:
                            ii = p.wql_grid_index[iq,il]
                            self.acf_wql_sp[:,ii,iT] += wq[iq] * A_lq[iql] ** 2 * (1.+2.*nph) * ltz[:] * Fjax_lq[jax,iql] * Fjax_lq[jax,iql].conjugate()
                            if il in p.phm_list:
                                iph = p.phm_list.index(il)
                                self.acf_phr_sp[:,iph,iT] += wq[iq] * A_lq[iql] ** 2 * (1.+2.*nph) * ltz[:] * Fjax_lq[jax,iql] * Fjax_lq[jax,iql].conjugate()
                        # at. resolved
                        if p.at_resolved:
                            ia = atoms.index_to_ia_map[jax] - 1
                            self.acf_atr_sp[:,ia,iT] += wq[iq] * A_lq[iql] ** 2 * (1.+2.*nph) * ltz[:] * Fjax_lq[jax,iql] * Fjax_lq[jax,iql].conjugate()
            iql += 1
    # compute <Delta V^(2)(t) Delta V^(2)(t')>_c
    def compute_acf_Vph2(self, wq, wu, iq, il, qlp_list, A_lq, A_lqp, F_lqlqp):
        # update acf_sp data
        # set wu[q]
        wuq = wu[iq]
        if wuq[il] > p.min_freq:
            Eql = wuq[il] * THz_to_ev
            # e^{-i wt}
            exp_iwt = np.zeros(p.nt, dtype=np.complex128)
            exp_int = np.zeros(p.nt)
            for t in range(p.nt):
                exp_iwt[t] = cmath.exp(-1j*2.*np.pi*wuq[il]*p.time[t])
                exp_int[t] = math.exp(-p.eta*p.time[t])
            # run over temperatures
            n_qlT = np.zeros(p.ntmp)
            for iT in range(p.ntmp):
                T = p.temperatures[iT]
                n_qlT[iT] = bose_occup(Eql, T)
            # run over (qp,ilp)
            iqlp = 0
            for iqp, ilp in qlp_list:
                # wuqp
                wuqp = wu[iqp]
                Eqlp = wuqp[ilp] * THz_to_ev
                if wuqp[ilp] > p.min_freq:
                    # e^{-w't}
                    exp_iwpt = np.zeros(p.nt, dtype=np.complex128)
                    for t in range(p.nt):
                        exp_iwpt[t] = cmath.exp(-1j*2.*np.pi*wuqp[ilp]*p.time[t])
                    # run over temperatures
                    n_qlpT = np.zeros(p.ntmp)
                    for iT in range(p.ntmp):
                        T = p.temperatures[iT]
                        n_qlpT[iT] = bose_occup(Eqlp, T)
                    A_ph1 = np.zeros(p.ntmp)
                    A_ph2 = np.zeros(p.ntmp)
                    A_ph3 = np.zeros(p.ntmp)
                    A_ph1[:] = 1. + n_qlT[:] + n_qlpT[:] + n_qlT[:] * n_qlpT[:]
                    A_ph2[:] = n_qlT[:] * n_qlpT[:]
                    A_ph3[:] = 2.*(1. + n_qlT[:]) * n_qlpT[:]
                    # compute time fluctuations
                    for iT in range(p.ntmp):
                        ft = np.zeros(p.nt, dtype=np.complex128)
                        ft[:] = A_ph1[iT] * exp_iwt[:] * exp_iwpt[:] + A_ph2[iT] * exp_iwt[:].conjugate() * exp_iwpt[:].conjugate() + A_ph3[iT] * exp_iwt[:] * exp_iwpt[:].conjugate() * exp_int[:]
                        # (eV^2) units
                        self.acf_sp[:,iT] += wq[iq] * wq[iqp] * A_lq ** 2 * A_lqp[iqlp] ** 2 * ft[:] * F_lqlqp[iqlp] * F_lqlqp[iqlp].conjugate()
                iqlp += 1
    #
    # compute <V(1)V(1)> dynamical decoupling
    #
    def compute_acf_Vph1_dyndec(self, wq, wu, ql_list, A_lq, F_lq):
        # compute partial sum acf
        npl = len(p.n_pulses)
        self.acfdd_sp = np.zeros((p.nt, npl, p.ntmp), dtype=np.complex128)
        # iql
        iql = 0
        # \sum_q \sum_l
        for iq, il in ql_list:
            wuq = wu[iq]
            if wuq[il] > p.min_freq:
                # E in eV
                E = wuq[il] * THz_to_ev
                wql = wuq[il]*2.*np.pi
                # iterate over pulses
                for n in p.n_pulses:
                    ni = p.n_pulses.index(n)
                    # run over temperatures
                    for iT in range(p.ntmp):
                        T = p.temperatures[iT]
                        # bose occupation
                        nph = bose_occup(E, T)
                        fw = np.zeros(p.nw)
                        # run over all frequencies
                        for iw in range(p.nw):
                            # (1)
                            Lw1 = lorentzian(wql/(2.*n)-p.wg[iw], p.eta)
                            Lw2 = lorentzian(wql/(2.*n)+p.wg[iw], p.eta)
                            fw[iw] = (1.+nph)/(2.*n) ** 3 * Lw1 + nph/(2.*n) ** 3 * Lw2
                            # (2)
                            Lw1 = lorentzian(wql-p.wg[iw], p.eta)
                            Lw2 = lorentzian(wql+p.wg[iw], p.eta)
                            fw[iw] += (-1) ** n / 2. * ((1.+nph) * Lw1 + nph * Lw2)
                            # (3)
                            if n > 1:
                                for j in range(1, n):
                                    Lw1 = lorentzian((2.*j+1)/(2.*n)*wql-p.wg[iw], p.eta)
                                    Lw2 = lorentzian((2.*j+1)/(2.*n)*wql+p.wg[iw], p.eta)
                                    fw[iw] += (1.+nph)*(-1) ** j * ((2.*j+1)/(2.*n)) ** 3 * Lw1 + nph * (-1) ** j * ((2.*j+1)/(2.*n)) ** 3 * Lw2
                        # (eV^2) units
                        self.acfdd_sp[:,ni,iT] += wq[iq] * 2.*np.pi * A_lq[iql] ** 2 * fw[:] * F_lq[iql] * F_lq[iql].conjugate()
            iql += 1
    #
    # dyndec calculation acf (2)
    def compute_acf_Vph2_dyndec(self, wq, wu, iq, il, qlp_list, A_lq, A_lqp, F_lqlqp, w_k):
        # update acfdd_sp data (order 1 calc.)
        # set wu[q]
        wuq = wu[iq]
        if wuq[il] > p.min_freq:
            wql = wuq[il]*2.*np.pi
            Eql = wuq[il] * THz_to_ev
            # run over temperatures
            n_qlT = np.zeros(p.ntmp)
            for iT in range(p.ntmp):
                T = p.temperatures[iT]
                n_qlT[iT] = bose_occup(Eql, T)
            # run over (qp,ilp)
            iqlp = 0
            for iqp, ilp in qlp_list:
                # wuqp
                wuqp = wu[iqp]
                Eqlp = wuqp[ilp] * THz_to_ev
                if wuqp[ilp] > p.min_freq:
                    wqlp = wuqp[ilp]*2.*np.pi
                    # run over temperatures
                    n_qlpT = np.zeros(p.ntmp)
                    for iT in range(p.ntmp):
                        T = p.temperatures[iT]
                        n_qlpT[iT] = bose_occup(Eqlp, T)
                    A_ph1 = np.zeros(p.ntmp)
                    A_ph2 = np.zeros(p.ntmp)
                    A_ph3 = np.zeros(p.ntmp)
                    A_ph1[:] = 1. + n_qlT[:] + n_qlpT[:] + n_qlT[:] * n_qlpT[:]
                    A_ph2[:] = n_qlT[:] * n_qlpT[:]
                    A_ph3[:] = 2.*(1. + n_qlT[:]) * n_qlpT[:]
                    e_iw1 = np.zeros(p.nt0, dtype=np.complex128)
                    e_iw2 = np.zeros(p.nt0, dtype=np.complex128)
                    e_iw3 = np.zeros(p.nt0, dtype=np.complex128)
                    for it in range(p.nt0):
                        it0 = p.it0_seq[it]
                        e_iw1[it] = cmath.exp(-1j*(wql+wqlp)*p.time[it0])
                        e_iw2[it] = cmath.exp(1j*(wql+wqlp)*p.time[it0])
                        e_iw3[it] = cmath.exp(-1j*(wql-wqlp-1j*p.eta)*p.time[it0])
                    # compute exp coeff.
                    for iT in range(p.ntmp):
                        # compute taylor coefficients exp.
                        dtk0_acf2 = np.zeros((p.n_dkt, p.nt0), dtype=np.complex128)
                        for it in range(p.nt0):
                            for k in range(p.n_dkt):
                                fk = (-1)**k * (wql+wqlp)**k * e_iw1[it] * A_ph1[iT] + (wql+wqlp)**k * e_iw2[it] * A_ph2[iT] + (-1)**k * (wql-wqlp-1j*p.eta)**k * e_iw3[it] * A_ph3[iT]
                                dtk0_acf2[k,it] += (1j)**k * wq[iq] * wq[iqp] * A_lq ** 2 * A_lqp[iqlp] ** 2 * fk * F_lqlqp[iqlp] * F_lqlqp[iqlp].conjugate()
                        max_dtk0 = np.max(dtk0_acf2)
                        dtk0_acf2 = dtk0_acf2 / max_dtk0
                        # update coeff
                        for n in p.n_pulses:
                            ni = p.n_pulses.index(n)
                            ft = np.zeros(p.nt, dtype=np.complex128)
                            for it in range(p.nt0):
                                it0 = p.it0_seq[it]
                                it1 = p.it1_seq[it]
                                for k in range(p.n_dkt):
                                    ak = dtk0_acf2[k,it] * w_k[ni,k] / math.factorial(k)
                                    ft[it0:it1] += ak * (p.time[it0:it1] - p.time[it0]) ** k
                            self.acfdd_sp[:,ni,iT] += ft[:] * max_dtk0
                            # (eV^2) units
                iqlp += 1
    #
    # ph / at resolved (2nd order)
    def compute_acf_Vph2_atphr(self, nat, wq, wu, iq, il, qlp_list, A_lq, A_lqp, Fjax_lqlqp):
        if not p.ph_resolved and not p.at_resolved:
            return
        if p.ph_resolved and il in p.phm_list:
            iph = p.phm_list.index(il)
        if p.ph_resolved:
            ii = p.wql_grid_index[iq,il]
        # compute acf
        wuq = wu[iq]
        Eql = wuq[il] * THz_to_ev
        if wuq[il] > p.min_freq:
            # e^{-i wt}
            exp_iwt = np.zeros(p.nt2, dtype=np.complex128)
            exp_int = np.zeros(p.nt2)
            for t in range(p.nt2):
                exp_iwt[t] = cmath.exp(-1j*2.*np.pi*wuq[il]*p.time2[t])
                exp_int[t] = math.exp(-p.eta*p.time2[t])
            # run over T
            n_qlT = np.zeros(p.ntmp)
            for iT in range(p.ntmp):
                T = p.temperatures[iT]
                n_qlT[iT] = bose_occup(Eql, T)
            # run over (qp.ilp)
            iqlp = 0
            for iqp, ilp in qlp_list:
                # wuqp
                wuqp = wu[iqp]
                if wuqp[ilp] > p.min_freq:
                    Eqlp = wuqp[ilp] * THz_to_ev
                    # e^{-w' t}
                    exp_iwpt = np.zeros(p.nt2, dtype=np.complex128)
                    for t in range(p.nt2):
                        exp_iwpt[t] = cmath.exp(-1j*2.*np.pi*wuqp[ilp]*p.time2[t])
                    # run over T
                    n_qlpT = np.zeros(p.ntmp)
                    for iT in range(p.ntmp):
                        T = p.temperatures[iT]
                        n_qlpT[iT] = bose_occup(Eqlp, T)
                    A_ph1 = np.zeros(p.ntmp)
                    A_ph2 = np.zeros(p.ntmp)
                    A_ph3 = np.zeros(p.ntmp)
                    A_ph1[:] = 1. + n_qlT[:] + n_qlpT[:] + n_qlT[:] * n_qlpT[:]
                    A_ph2[:] = n_qlT[:] * n_qlpT[:]
                    A_ph3[:] = 2.*(1. + n_qlT[:]) * n_qlpT[:]
                    # compute fluctuations
                    for iT in range(p.ntmp):
                        ft = np.zeros(p.nt2, dtype=np.complex128)
                        ft[:] = A_ph1[iT] * exp_iwt[:] * exp_iwpt[:] + A_ph2[iT] * exp_iwt[:].conjugate() * exp_iwpt[:].conjugate() + A_ph3[iT] * exp_iwt[:] * exp_iwpt[:].conjugate() * exp_int[:]
                        # (eV^2) units
                        for jax in range(3*nat):
                            if p.ph_resolved:
                                self.acf_wql_sp[:,ii,iT] += wq[iq] * wq[iqp] * A_lq ** 2 * A_lqp[iqlp] ** 2 * ft[:] * Fjax_lqlqp[jax,iqlp] * Fjax_lqlqp[jax,iqlp].conjugate()
                                if il in p.phm_list:
                                    self.acf_phr_sp[:,iph,iT] += wq[iq] * wq[iqp] * A_lq ** 2 * A_lqp[iqlp] ** 2 * ft[:] * Fjax_lqlqp[jax,iqlp] * Fjax_lqlqp[jax,iqlp].conjugate()
                            if p.at_resolved:
                                ia = atoms.index_to_ia_map[jax] - 1
                                self.acf_atr_sp[:,ia,iT] += wq[iq] * wq[iqp] * A_lq ** 2 * A_lqp[iqlp] ** 2 * ft[:] * Fjax_lqlqp[jax,iqlp] * Fjax_lqlqp[jax,iqlp].conjugate()
                iqlp += 1
# -----------------------------------------------------------------------
#
#             GPU CLASS
#
# -----------------------------------------------------------------------
class GPU_acf_ph_deph(acf_ph_deph):
    def __init__(self):
        super(GPU_acf_ph_deph, self).__init__()
        # set up constants
        self.THz_to_ev = np.double(THz_to_ev)
        self.min_freq = np.double(p.min_freq)
        self.kb = np.double(kb)
        self.toler = np.double(eps)
    # GPU equivalent function
    def compute_acf_Vph2(self, wq, wu, iq, il, qlp_list, A_lq, A_lqp, F_lqlqp):
        '''
        compute ACF partial sum
        '''
        gpu_src = Path('./pydephasing/gpu_source/compute_acf_Vsph2.cu').read_text()
        mod = SourceModule(gpu_src)
        compute_acf = mod.get_function("compute_acf_Vsph2")
        # split modes on grid
        qlp_gpu, init_gpu, lgth_gpu = gpu.split_data_on_grid(range(len(qlp_list)))
        # check energy Eql
        if wu[iq][il] > p.min_freq:
            Eql = wu[iq][il] * THz_to_ev
            # frequency
            wuq = np.zeros(len(qlp_list), dtype=np.double)
            w_q = np.double(wq[iq])
            w_qp= np.zeros(len(qlp_list), dtype=np.double)
            ind = 0
            for iqp, ilp in qlp_list:
                w_qp[ind] = wq[iqp]
                wuq[ind] = wu[iqp][ilp]
                ind += 1
            # effective force
            Flqp = np.zeros(len(qlp_list), dtype=np.complex128)
            Alqp = np.zeros(len(qlp_list), dtype=np.double)
            for iqlp in range(len(qlp_list)):
                Flqp[iqlp] = F_lqlqp[iqlp]
                Alqp[iqlp] = A_lqp[iqlp]
            Alq = np.double(A_lq)
            # temperature
            for iT in range(p.ntmp):
                nlq = 0.
                T = p.temperatures[iT]
                nlq = np.double(bose_occup(Eql, T))
                T = np.double(T)
                # iterate over time variable
                t0 = 0
                while (t0 < p.nt):
                    t1 = t0 + gpu.BLOCK_SIZE[0]*gpu.BLOCK_SIZE[1]*gpu.BLOCK_SIZE[2]
                    size = min(t1, p.nt)-t0
                    size_32 = np.int32(size)
                    # set e^{-iwt}
                    exp_iwt = np.zeros(size, dtype=np.complex128)
                    exp_int = np.zeros(size, dtype=np.double)
                    acf = np.zeros(gpu.gpu_size, dtype=np.complex128)
                    time = np.zeros(size, dtype=np.double)
                    for t in range(t0, min(t1,p.nt)):
                        exp_iwt[t-t0] = cmath.exp(-1j*2.*np.pi*wu[iq][il]*p.time[t])
                        exp_int[t-t0] = math.exp(-p.eta*p.time[t])
                        time[t-t0] = p.time[t]
                    # call function
                    compute_acf(cuda.In(w_qp), cuda.In(wuq), cuda.In(exp_iwt), cuda.In(exp_int),
                        cuda.In(time), cuda.In(Flqp), cuda.In(Alqp), cuda.In(qlp_gpu), cuda.In(init_gpu),
                        cuda.In(lgth_gpu), Alq, T, w_q, nlq, self.THz_to_ev, self.min_freq, self.kb, 
                        self.toler, size_32, cuda.Out(acf), block=gpu.block, grid=gpu.grid)
                    acf = gpu.recover_data_from_grid(acf)
                    for t in range(t0, min(t1,p.nt)):
                        self.acf_sp[t,iT] += acf[t-t0]
                    t0 = t1
    #
    # dyndec calculation acf (2)
    def compute_dkt0_acf_Vph2(self, wq, wu, iq, il, qlp_list, A_lq, A_lqp, F_lqlqp, w_k):
        '''
        compute ACF partial sum
        '''
        gpu_src = Path('./pydephasing/gpu_source/compute_acf_Vsph_dyndec.cu').read_text()
        mod = SourceModule(gpu_src)
        compute_acf = mod.get_function("compute_acf_Vsph2_dyndec")
        # split modes on grid
        qlp_gpu, init_gpu, lgth_gpu = gpu.split_data_on_grid(range(len(qlp_list)))
        # check energy Eql
        if wu[iq][il] > p.min_freq:
            Eql = wu[iq][il] * THz_to_ev
            wql = 2.*np.pi*wu[iq][il]
            wql = np.double(wql)
            # frequency
            wuq = np.zeros(len(qlp_list), dtype=np.double)
            w_q = np.double(wq[iq])
            w_qp= np.zeros(len(qlp_list), dtype=np.double)
            ind = 0
            for iqp, ilp in qlp_list:
                w_qp[ind] = wq[iqp]
                wuq[ind] = wu[iqp][ilp]
                ind += 1
            # effective force
            Flqp = np.zeros(len(qlp_list), dtype=np.complex128)
            Alqp = np.zeros(len(qlp_list), dtype=np.double)
            for iqlp in range(len(qlp_list)):
                Flqp[iqlp] = F_lqlqp[iqlp]
                Alqp[iqlp] = A_lqp[iqlp]
            Alq = np.double(A_lq)
            # time array
            time = np.zeros(p.nt, dtype=np.double)
            for t in range(p.nt):
                time[t] = p.time[t]
            nu = np.double(p.eta)
            # w_k array
            npl = len(p.n_pulses)
            ndkt_32 = np.int32(p.n_dkt)
            factorial_k = np.zeros(p.n_dkt, dtype=np.double)
            for k in range(p.n_dkt):
                factorial_k[k] = math.factorial(k)
            for ni in range(npl):
                w_kn = np.zeros(p.n_dkt, dtype=np.double)
                for k in range(p.n_dkt):
                    w_kn[k] = w_k[ni,k]
                # temperature
                #
                for iT in range(p.ntmp):
                    nlq = 0.
                    T = p.temperatures[iT]
                    nlq = np.double(bose_occup(Eql, T))
                    T = np.double(T)
                    # iterate over time variable
                    t0 = 0
                    while (t0 < p.nt):
                        t1 = t0 + gpu.BLOCK_SIZE[0]*gpu.BLOCK_SIZE[1]*gpu.BLOCK_SIZE[2]
                        t_size = min(t1, p.nt) - t0
                        t_size = np.int32(t_size)
                        # build it0lst - it1lst
                        it0lst = []
                        it1lst = []
                        for it in range(p.nt0):
                            it0 = p.it0_seq[it]
                            it1 = p.it1_seq[it]
                            if it0 >= t0 and it0 <= t1:
                                it0lst.append(it0)
                                it1lst.append(min(t1,it1))
                            elif it0 < t0 and it1 >= t0:
                                it0lst.append(it0)
                                it1lst.append(min(it1,t1))
                        it0lst = np.array(it0lst, dtype=np.int32)
                        it1lst = np.array(it1lst, dtype=np.int32)
                        nt0 = np.int32(len(it0lst))
                        # set output function
                        acf = np.zeros(gpu.gpu_size, dtype=np.complex128)
                        # call function
                        compute_acf(cuda.In(w_qp), cuda.In(wuq), cuda.In(time), cuda.In(Flqp), 
                            cuda.In(Alqp), cuda.In(w_kn), cuda.In(qlp_gpu), cuda.In(init_gpu), cuda.In(lgth_gpu),
                            cuda.In(it0lst), cuda.In(it1lst), cuda.In(factorial_k), Alq, T, w_q, wql, nlq, self.THz_to_ev, self.min_freq,
                            self.kb, self.toler, nu, nt0, ndkt_32, t_size, cuda.Out(acf), block=gpu.block, grid=gpu.grid)
                        acf = gpu.recover_data_from_grid(acf)
                        for t in range(t0, min(t1,p.nt)):
                            self.acfdd_sp[t,ni,iT] += acf[t-t0]
                        t0 = t1
    #
    # ph / at resolved (2nd order)
    def compute_acf_Vph2_atphr(self, nat, wq, wu, iq, il, qlp_list, A_lq, A_lqp, Fjax_lqlqp):
        if not p.at_resolved and not p.ph_resolved:
            return
        '''
        compute ACF partial sum
        '''
        if p.at_resolved:
            gpu_src = Path('./pydephasing/gpu_source/compute_acf_Vsph2_atr.cu').read_text()
            mod = SourceModule(gpu_src)
            compute_acf_atr = mod.get_function("compute_acf_Vsph2_atr")
        if p.ph_resolved:
            gpu_src = Path('./pydephasing/gpu_source/compute_acf_Vsph2.cu').read_text()
            mod = SourceModule(gpu_src)
            compute_acf_phr = mod.get_function("compute_acf_Vsph2")
            if il in p.phm_list:
                iph = p.phm_list.index(il)
            ii = p.wql_grid_index[iq,il]
            # split second modes on the grid
            qlp_gpu, init_gpu, lgth_gpu = gpu.split_data_on_grid(range(len(qlp_list)))
        #
        # first check energy Eql
        wuq = wu[iq]
        if wuq[il] > p.min_freq:
            Eql = wuq[il] * THz_to_ev
            # set parameters
            if p.at_resolved:
                nmodes = np.int32(len(qlp_list))
                nat_32 = np.int32(nat)
            # wuq / wq
            wuq = np.zeros(len(qlp_list), dtype=np.double)
            w_q = np.double(wq[iq])
            w_qp = np.zeros(len(qlp_list), dtype=np.double)
            ind = 0
            for iqp, ilp in qlp_list:
                w_qp[ind] = wq[iqp]
                wuq[ind] = wu[iqp][ilp]
                ind += 1
            # effective force
            if p.at_resolved:
                Fjax_lqp = np.zeros(3*nat*len(qlp_list), dtype=np.complex128)
                ind = 0
                for iqlp in range(len(qlp_list)):
                    for jax in range(3*nat):
                        Fjax_lqp[ind] = Fjax_lqlqp[jax,iqlp]
                        ind += 1
            if p.ph_resolved:
                F_lqp = np.zeros(len(qlp_list), dtype=np.complex128)
                for jax in range(3*nat):
                    F_lqp[:] += Fjax_lqlqp[jax,:]
            Alqp = np.zeros(len(qlp_list), dtype=np.double)
            for iqlp in range(len(qlp_list)):
                Alqp[iqlp] = A_lqp[iqlp]
            Alq = np.double(A_lq)
            # run over T
            for iT in range(p.ntmp):
                nlq = 0.
                T = p.temperatures[iT]
                nlq = np.double(bose_occup(Eql, T))
                T = np.double(T)
                # iterate over t
                t0 = 0
                while (t0 < p.nt2):
                    t1 = t0 + gpu.BLOCK_SIZE[0]*gpu.BLOCK_SIZE[1]*gpu.BLOCK_SIZE[2]
                    nt = min(t1, p.nt2)-t0
                    t_size = np.int32(nt)
                    # e^{-i wt}
                    exp_iwt = np.zeros(nt, dtype=np.complex128)
                    exp_int = np.zeros(nt, dtype=np.double)
                    time = np.zeros(nt, dtype=np.double)
                    for t in range(t0, min(t1,p.nt2)):
                        exp_iwt[t-t0] = cmath.exp(-1j*2.*np.pi*wu[iq][il]*p.time2[t])
                        exp_int[t-t0] = math.exp(-p.eta*p.time2[t])
                        time[t-t0] = p.time2[t]
                    if p.at_resolved:
                        ia0 = 0
                        while ia0 < nat:
                            acf = np.zeros(gpu.gpu_size, dtype=np.complex128)
                            ia1 = ia0 + gpu.GRID_SIZE[0]*gpu.GRID_SIZE[1]
                            na = min(ia1, nat)-ia0
                            atr_gpu = np.zeros(na, dtype=np.int32)
                            for a in range(ia0, min(ia1,nat)):
                                atr_gpu[a-ia0] = a
                            na_size = np.int32(na)
                            # compute fluctuations
                            compute_acf_atr(cuda.In(w_qp), cuda.In(wuq), cuda.In(exp_iwt), cuda.In(exp_int),
                                cuda.In(time), cuda.In(Fjax_lqp), cuda.In(Alqp), cuda.In(atr_gpu),
                                Alq, T, w_q, nlq, self.THz_to_ev, self.min_freq, self.kb,
                                self.toler, t_size, na_size, nmodes, nat_32, cuda.Out(acf),
                                block=gpu.block, grid=gpu.grid)
                            # (eV^2) units
                            acf = gpu.recover_data_from_grid_atr(acf, na, nt)
                            for t in range(t0, min(t1,p.nt2)):
                                for a in range(ia0, min(ia1,nat)):
                                    self.acf_atr_sp[t,a,iT] += acf[t-t0,a-ia0]
                            ia0 = ia1
                    if p.ph_resolved:
                        acf = np.zeros(gpu.gpu_size, dtype=np.complex128)
                        compute_acf_phr(cuda.In(w_qp), cuda.In(wuq), cuda.In(exp_iwt), cuda.In(exp_int),
                            cuda.In(time), cuda.In(F_lqp), cuda.In(Alqp), cuda.In(qlp_gpu), cuda.In(init_gpu),
                            cuda.In(lgth_gpu), Alq, T, w_q, nlq, self.THz_to_ev, self.min_freq, self.kb, 
                            self.toler, t_size, cuda.Out(acf), block=gpu.block, grid=gpu.grid)
                        acf = gpu.recover_data_from_grid(acf)
                        for t in range(t0, min(t1,p.nt2)):
                            self.acf_wql_sp[t,ii,iT] += acf[t-t0]
                        if il in p.phm_list:
                            for t in range(t0, min(t1,p.nt2)):
                                self.acf_phr_sp[t,iph,iT] += acf[t-t0]
                    t0 = t1