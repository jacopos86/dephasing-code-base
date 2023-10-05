#
# This module sets up the methods
# needed to compute the energy / spin fluctuations
# auto correlation function
#
import numpy as np
import cmath
import math
from pydephasing.phys_constants import THz_to_ev, eps, kb, hbar
from pydephasing.input_parameters import p
from pydephasing.atomic_list_struct import atoms
from pydephasing.utility_functions import bose_occup, lorentzian
from tqdm import tqdm
from pydephasing.global_params import GPU_ACTIVE
from pathlib import Path
import matplotlib.pyplot as plt
from pydephasing.auto_correlation_driver import acf_ph
# pycuda
if GPU_ACTIVE:
    from pycuda.compiler import SourceModule
    import pycuda.driver as cuda
    import pycuda.autoinit
    from pydephasing.global_params import gpu
# --------------------------------------------------------------
#
#    spin-exc-ph   RELAXATION CLASS
#
# --------------------------------------------------------------
class acf_sp_ph(acf_ph):
    def __init__(self):
        super(acf_sp_ph, self).__init__()
        self.dE = 0.0
    # set dE
    def set_dE(self, H):
        if p.relax:
            # quantum states
            iqs0 = p.index_qs0
            iqs1 = p.index_qs1
            self.dE = H.eig[iqs1] - H.eig[iqs0]
            # eV
        elif p.deph:
            pass
    #
    # <Delta V^(1)(t) Delta V^(1)(t)>
    def compute_acf_V1_t0(self, wq, wu, ql_list, A_lq, F_lq):
        # Delta_2 = sum_l,q A_l,q^2 [1 + 2 n_lq] |F_lq|^2
        # eV^2 units
        Delta_2 = np.zeros(p.ntmp)
        # compute partial Delta_2
        iql = 0
        for iq, il in ql_list:
            wuq = wu[iq]
            # E in eV
            E_ql = wuq[il] * THz_to_ev
            if wuq[il] > p.min_freq:
                # run over temperatures
                for iT in range(p.ntmp):
                    T = p.temperatures[iT]
                    # bose occup.
                    nph = bose_occup(E_ql, T)
                    Delta_2[iT] += wq[iq] * A_lq[iql] ** 2 * (1.+2.*nph) * (F_lq[iql] * F_lq[iql].conjugate()).real
            iql += 1
        return Delta_2
    #
    # acf V1 (w=0)
    def compute_acf_V1_w0(self, wq, wu, ql_list, A_lq, F_lq):
        # Delta_w0 = sum_l,q A_l,q^2 [1 + 2 n_lq] |F_lq|^2
        dE = self.dE
        # eV units
        Delta_w0 = np.zeros(p.ntmp)
        # compute partial value
        iql = 0
        for iq, il in ql_list:
            wuq = wu[iq]
            if wuq[il] > p.min_freq:
                # E in eV
                E_ql = wuq[il] * THz_to_ev
                ltza = lorentzian(dE+E_ql, p.eta)
                ltzb = lorentzian(dE-E_ql, p.eta)
                # eV^-1
                # temperatures
                for iT in range(p.ntmp):
                    T = p.temperatures[iT]
                    # bose occ.
                    nph = bose_occup(E_ql, T)
                    Delta_w0[iT] += wq[iq] * A_lq[iql] ** 2 * ((1.+nph)*ltza + nph*ltzb) * (F_lq[iql] * F_lq[iql].conjugate()).real
            iql += 1
        return Delta_w0
    #
    # acf V2 (t=0)
    def compute_acf_V2_t0(self, wq, wu, iq, il, qlp_list, A_lq, A_lqp, F_lqlqp):
        # eV^2 units
        Delta_2 = np.zeros(p.ntmp, dtype=np.complex128)
        # check freq. value
        if wu[iq][il] > p.min_freq:
            Eql = wu[iq][il] * THz_to_ev
            # ph. occup.
            nql_T = np.zeros(p.ntmp)
            for iT in range(p.ntmp):
                T = p.temperatures[iT]
                nql_T[iT] = bose_occup(Eql, T)
            # iterate over (q',l')
            iqlp = 0
            for iqp, ilp in qlp_list:
                # E in eV
                Eqlp = wu[iqp][ilp] * THz_to_ev
                # e^{-i w t} = 1
                # total force
                F_llp = sum(F_lqlqp[:,iqlp])
                # run over T
                for iT in range(p.ntmp):
                    T = p.temperatures[iT]
                    nqlp_T = bose_occup(Eqlp, T)
                    #
                    # compute Delta^2
                    A_th = nql_T[iT] * (1. + nqlp_T)
                    Delta_2[iT] += wq[iq] * wq[iqp] * A_lq ** 2 * A_lqp[iqlp] ** 2 * A_th * F_llp[iqlp] * np.conjugate(F_llp[iqlp])
                iqlp += 1
        # final result
        Delta_2r = np.zeros(p.ntmp)
        for iT in range(p.ntmp):
            Delta_2r[iT] = Delta_2[iT].real
        return Delta_2r
    #
    # acf V2 (w=0)
    def compute_acf_V2_w0(self, wq, wu, iq, il, qlp_list, A_lq, A_lqp, F_lqlqp):
        # eV^2 units
        Delta_w0 = np.zeros(p.ntmp, dtype=np.complex128)
        # check freq. value
        if wu[iq][il] > p.min_freq:
            Eql = wu[iq][il] * THz_to_ev
            # ph. occup.
            nql_T = np.zeros(p.ntmp)
            for iT in range(p.ntmp):
                T = p.temperatures[iT]
                nql_T[iT] = bose_occup(Eql, T)
            # iterate over (q',l')
            iqlp = 0
            for iqp, ilp in qlp_list:
                # E in eV
                Eqlp = wu[iqp][ilp] * THz_to_ev
                # e^{-i w t} = 1
                # total force
                F_llp = sum(F_lqlqp[:,iqlp])
                # run over T
                for iT in range(p.ntmp):
                    T = p.temperatures[iT]
                    nqlp_T = bose_occup(Eqlp, T)
                    #
                    # compute Delta^2
                    A_th = nql_T[iT] * (1. + nqlp_T)
                    Delta_w0[iT] += wq[iq] * wq[iqp] * A_lq ** 2 * A_lqp[iqlp] ** 2 * A_th * F_llp[iqlp] * np.conjugate(F_llp[iqlp])
                iqlp += 1
        # final result
        Delta_w0r = np.zeros(p.ntmp)
        for iT in range(p.ntmp):
            Delta_w0r[iT] = Delta_w0[iT].real
        return Delta_w0r
# ---------------------------------------------------------------------
#
#     CPU class
#
# ---------------------------------------------------------------------
class CPU_acf_sp_ph(acf_sp_ph):
    def __init__(self):
        super(CPU_acf_sp_ph, self).__init__()
    #
    # compute <Delta V^(1)(t) Delta V^(1)(t')>
    def compute_acf_V1_oft(self, wq, wu, ql_list, A_lq, F_lq):
        # initialize acf_sp -> 0 acf -> 1 integral
        self.acf_sp = np.zeros((p.nt, 2, p.ntmp), dtype=np.complex128)
        # dE
        dE = self.dE / hbar
        nu = p.eta / hbar
        # ps^-1
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
                    exp_iwt[t] = cmath.exp(-1j*(wql+dE-1j*nu)*p.time[t])
                    cc_exp_iwt[t] = cmath.exp(1j*(wql-dE+1j*nu)*p.time[t])
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
                    ft[:] = (1.+nph) * (exp_iwt[:] - 1.)/(-1j*(wql+dE-1j*nu)) + nph * (cc_exp_iwt[:] - 1.)/(1j*(wql-dE+1j*nu))
                    self.acf_sp[:,1,iT] += wq[iq] * A_lq[iql] ** 2 * ft[:] * F_lq[iql] * F_lq[iql].conjugate()
            iql += 1
    #
    # compute <Delta V(1) \Delta V(1)>(w)
    def compute_acf_V1_ofw(self, wq, wu, ql_list, A_lq, F_lq):
        # initialize acf_sp -> 0 acf -> 1 integral
        self.acf_sp = np.zeros((p.nwg, p.ntmp), dtype=np.complex128)
        # dE (eV)
        dE = self.dE
        ltza = np.zeros(p.nwg)
        ltzb = np.zeros(p.nwg)
        # compute partial acf \sum_ql
        iql = 0
        for iq, il in ql_list:
            wuq = wu[iq]
            if wuq[il] > p.min_freq:
                # E in eV
                Eql = wuq[il] * THz_to_ev
                ltza[:] = 0.
                ltzb[:] = 0.
                for iw in range(p.nwg):
                    w = p.w_grid[iw]
                    # eV
                    ltza[iw] = lorentzian(dE+Eql+w, p.eta)
                    ltzb[iw] = lorentzian(dE-Eql+w, p.eta)
                    # eV^-1
                # run over temperatures
                for iT in range(p.ntmp):
                    T = p.temperatures[iT]
                    # bose occ.
                    nph = bose_occup(Eql, T)
                    # eV/ps^2*eV*ps^2*eV^-1 = eV
                    self.acf_sp[:,iT] += wq[iq] * A_lq[iql] ** 2 * ((1.+nph)*ltza[:] + nph*ltzb[:]) * F_lq[iql] * F_lq[iql].conjugate()
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
        # dE (ps^-1)
        dE = self.dE / hbar
        nu = p.eta / hbar
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
                    exp_iwt[t] = cmath.exp(-1j*(wql+dE-1j*nu)*p.time2[t])
                    cc_exp_iwt[t] = cmath.exp(1j*(wql-dE+1j*nu)*p.time2[t])
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
                    ft[:] = (1.+nph) * (exp_iwt[:] - 1.)/(-1j*(wql+dE-1j*nu)) + nph * (cc_exp_iwt[:] - 1.)/(1j*(wql-dE+1j*nu))
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
        # dE (eV)
        dE = self.dE
        ltza = np.zeros(p.nwg)
        ltzb = np.zeros(p.nwg)
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
                    ltza[iw] = lorentzian(dE+Eql+w, p.eta)
                    ltzb[iw] = lorentzian(dE-Eql+w, p.eta)
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
                            self.acf_wql_sp[:,ii,iT] += wq[iq] * A_lq[iql] ** 2 * ((1.+nph)*ltza[:] + nph*ltzb[:]) * Fjax_lq[jax,iql] * Fjax_lq[jax,iql].conjugate()
                            if il in p.phm_list:
                                iph = p.phm_list.index(il)
                                self.acf_phr_sp[:,iph,iT] += wq[iq] * A_lq[iql] ** 2 * ((1.+nph)*ltza[:] + nph*ltzb[:]) * Fjax_lq[jax,iql] * Fjax_lq[jax,iql].conjugate()
                        # at. resolved
                        if p.at_resolved:
                            ia = atoms.index_to_ia_map[jax] - 1
                            self.acf_atr_sp[:,ia,iT] += wq[iq] * A_lq[iql] ** 2 * ((1.+nph)*ltza[:] + nph*ltzb[:]) * Fjax_lq[jax,iql] * Fjax_lq[jax,iql].conjugate()
            iql += 1
    # ---------------------------------------------------------------------
    # compute <Delta V^(2)(t) Delta V^(2)(t')>_c
    # ---------------------------------------------------------------------
    def compute_acf_V2_oft(self, wq, wu, iq, il, qlp_list, A_lq, A_lqp, F_lqlqp):
        # dE
        dE = self.dE / hbar
        nu = p.eta / hbar
        # update acf_sp data
        if wu[iq][il] > p.min_freq:
            Eql = wu[iq][il] * THz_to_ev
            wql = 2.*np.pi*wu[iq][il]
            # bose occupations
            nql_T = np.zeros(p.ntmp)
            for iT in range(p.ntmp):
                T = p.temperatures[iT]
                nql_T[iT] = bose_occup(Eql, T)
            # iterate over (q',l')
            iqlp = 0
            for iqp, ilp in qlp_list:
                # E in eV
                Eqlp = wu[iqp][ilp] * THz_to_ev
                wqlp = 2.*np.pi*wu[iqp][ilp]
                # e^{-i wt} -> w = dE - wql + wqlp
                exp_iwt = np.zeros(p.nt, dtype=np.complex128)
                for t in range(p.nt):
                    exp_iwt[t] = cmath.exp(-1j*(dE+wqlp-wql-1j*nu)*p.time[t])
                # set total force F_ll' = F_lq,l'q' + F_l-q,l'q' + F_lq,l'-q' + F_l-q,l'-q'
                F_llp = sum(F_lqlqp[:,iqlp])
                #
                # run over temperatures
                for iT in range(p.ntmp):
                    T = p.temperatures[iT]
                    nqlp_T = bose_occup(Eqlp, T)
                    #
                    # compute time fluctuations
                    ft = np.zeros(p.nt, dtype=np.complex128)
                    ft[:] = nql_T[iT] * (1. + nqlp_T) * exp_iwt[:]
                    # (eV^2) units
                    self.acf_sp[:,0,iT] += wq[iq] * wq[iqp] * A_lq ** 2 * A_lqp[iqlp] ** 2 * ft[:] * F_llp * F_llp.conjugate()
                    ft[:] = 0.
                    ft[:] = nql_T[iT]*(1. + nqlp_T)*(exp_iwt[:] - 1.)/(-1j*(dE+wqlp-wql-1j*nu))
                    self.acf_sp[:,1,iT] += wq[iq] * wq[iqp] * A_lq ** 2 * A_lqp[iqlp] ** 2 * ft[:] * F_llp * F_llp.conjugate()
                    # (eV^2 ps) units
                iqlp += 1
    # ---------------------------------------------------------------------
    # compute <Delta V^(2) Delta V^(2)>_c(w)
    # ---------------------------------------------------------------------
    def compute_acf_V2_ofw(self, wq, wu, iq, il, qlp_list, A_lq, A_lqp, F_lqlqp):
        # dE
        dE = self.dE / hbar
        # update acf_sp data
        if wu[iq][il] > p.min_freq:
            Eql = wu[iq][il] * THz_to_ev
            # bose occupations
            nql_T = np.zeros(p.ntmp)
            for iT in range(p.ntmp):
                T = p.temperatures[iT]
                nql_T[iT] = bose_occup(Eql, T)
            # iterate over (q',l')
            iqlp = 0
            for iqp, ilp in qlp_list:
                # E in eV
                Eqlp = wu[iqp][ilp] * THz_to_ev
                # set total force F_ll' = F_lq,l'q' + F_l-q,l'q' + F_lq,l'-q' + F_l-q,l'-q'     
                F_llp = sum(F_lqlqp[:,iqlp])
                # set lorentzian
                ltz = np.zeros(p.nwg)
                for iw in range(p.nwg):
                    w = p.w_grid[iw]
                    # eV
                    ltz[iw] = lorentzian(dE+Eql-Eqlp+w, p.eta)
                    # eV^-1
                # run over temperatures
                for iT in range(p.ntmp):
                    T = p.temperatures[iT]
                    nqlp_T = bose_occup(Eqlp, T)
                    #
                    # (eV) units
                    self.acf_sp[:,iT] += wq[iq] * wq[iqp] * A_lq ** 2 * A_lqp[iqlp] ** 2 * nql_T[iT] * (1.+nqlp_T) * ltz[:] * F_llp * F_llp.conjugate()
                iqlp += 1
    #
    # compute <Delta V^(2)(t) Delta V^(2)(t')> -> ph / at resolved
    def compute_acf_V2_atphr_oft(self, nat, wq, wu, iq, il, qlp_list, A_lq, A_lqp, Fjax_lqlqp):
        if not p.ph_resolved and not p.at_resolved:
            return
        # dE (ps^-1)
        dE = self.dE / hbar
        nu = p.eta / hbar
        # update acf_sp
        if wu[iq][il] > p.min_freq:
            Eql = wu[iq][il] * THz_to_ev
            wql = 2.*np.pi*wu[iq][il]
            # bose occupations
            nql_T = np.zeros(p.ntmp)
            for iT in range(p.ntmp):
                T = p.temperatures[iT]
                nql_T[iT] = bose_occup(Eql, T)
            # iterate -> (q',l')
            iqlp = 0
            for iqp, ilp in qlp_list:
                # E in (eV)
                Eqlp = wu[iqp][ilp] * THz_to_ev
                wqlp = 2.*np.pi*wu[iqp][ilp]
                # set e^{-iwt} -> w = dE - wql + wqlp
                exp_iwt = np.zeros(p.nt2, dtype=np.complex128)
                for t in range(p.nt2):
                    exp_iwt[t] = cmath.exp(-1j*(dE+wqlp-wql-1j*nu)*p.time2[t])
                # set total force F_ll' = F_lq,l'q' + F_l-q,l'q' + F_lq,l'-q' + F_l-q,l'-q'
                Fjax_llp = np.zeros(3*nat, dtype=np.complex128)
                for jax in range(3*nat):
                    Fjax_llp[jax] = sum(Fjax_lqlqp[:,jax,iqlp])
                # run over temperatures
                for iT in range(p.ntmp):
                    T = p.temperatures[iT]
                    # bose occupation
                    nqlp_T = bose_occup(Eqlp, T)
                    #
                    # time fluctuations
                    ft = np.zeros(p.nt2, dtype=np.complex128)
                    ft[:] = nql_T[iT] * (1. + nqlp_T) * exp_iwt[:]
                    for jax in range(3*nat):
                        # (eV^2) units
                        # ph. resolved
                        if p.ph_resolved:
                            ii = p.wql_grid_index[iq,il]
                            self.acf_wql_sp[:,0,ii,iT] += wq[iq] * wq[iqp] * A_lq ** 2 * A_lqp[iqlp] ** 2 * ft[:] * Fjax_llp[jax] * Fjax_llp[jax].conjugate()
                            if il in p.phm_list:
                                iph = p.phm_list.index(il)
                                self.acf_phr_sp[:,0,iph,iT] += wq[iq] * wq[iqp] * A_lq ** 2 * A_lqp[iqlp] ** 2 * ft[:] * Fjax_llp[jax] * Fjax_llp[jax].conjugate()
                        # at. resolved
                        if p.at_resolved:
                            ia = atoms.index_to_ia_map[jax] - 1
                            self.acf_atr_sp[:,0,ia,iT] += wq[iq] * wq[iqp] * A_lq ** 2 * A_lqp[iqlp] ** 2 * ft[:] * Fjax_llp[jax] * Fjax_llp[jax].conjugate()
                    # integral
                    # (eV^2 ps) units
                    ft[:] = 0.
                    ft[:] = nql_T[iT] * (1.+nqlp_T) * (exp_iwt[:] - 1.)/(-1j*(dE+wqlp-wql-1j*nu))
                    for jax in range(3*nat):
                        # (eV^2) units
                        # ph. resolved
                        if p.ph_resolved:
                            ii = p.wql_grid_index[iq,il]
                            self.acf_wql_sp[:,1,ii,iT] += wq[iq] * wq[iqp] * A_lq ** 2 * A_lqp[iqlp] ** 2 * ft[:] * Fjax_llp[jax] * Fjax_llp[jax].conjugate()
                            if il in p.phm_list:
                                iph = p.phm_list.index(il)
                                self.acf_phr_sp[:,1,iph,iT] += wq[iq] * wq[iqp] * A_lq ** 2 * A_lqp[iqlp] ** 2 * ft[:] * Fjax_llp[jax] * Fjax_llp[jax].conjugate()
                        # at. resolved
                        if p.at_resolved:
                            ia = atoms.index_to_ia_map[jax] - 1
                            self.acf_atr_sp[:,1,ia,iT] += wq[iq] * wq[iqp] * A_lq ** 2 * A_lqp[iqlp] ** 2 * ft[:] * Fjax_llp[jax] * Fjax_llp[jax].conjugate()
                iqlp += 1
    #
    # compute <Delta V(2) \Delta V(2)>(w) / at-ph res.
    def compute_acf_V2_atphr_ofw(self, nat, wq, wu, iq, il, qlp_list, A_lq, A_lqp, Fjax_lqlqp):
        if not p.ph_resolved and not p.at_resolved:
            return
        # dE (eV)
        dE = self.dE
        # update acf_sp data
        if wu[iq][il] > p.min_freq:
            Eql = wu[iq][il] * THz_to_ev
            # bose occup.
            nql_T = np.zeros(p.ntmp)
            for iT in range(p.ntmp):
                T = p.temperatures[iT]
                nql_T[iT] = bose_occup(Eql, T)
            # compute acf (2) -> (q',l') 
            iqlp = 0
            for iqp, ilp in qlp_list:
                # E in eV
                Eqlp = wu[iqp][ilp] * THz_to_ev
                # set total force F_ll' = F_lq,l'q' + F_l-q,l'q' + F_lq,l'-q' + F_l-q,l'-q'
                Fjax_llp = np.zeros(3*nat, dtype=np.complex128)
                for jax in range(3*nat):
                    Fjax_llp[jax] = sum(Fjax_lqlqp[:,jax,iqlp])
                # compute lorentz.
                ltz = np.zeros(p.nwg)
                for iw in range(p.nwg):
                    w = p.w_grid[iw]
                    # eV
                    ltz[iw] = lorentzian(dE+Eql-Eqlp+w, p.eta)
                    # eV^-1
                # run over temperatures
                for iT in range(p.ntmp):
                    T = p.temperatures[iT]
                    # bose occupation
                    nqlp_T = bose_occup(Eqlp, T)
                    # fw
                    fw = np.zeros(p.nwg, dtype=np.complex128)
                    fw[:] = nql_T[iT] * (1. + nqlp_T) * ltz[:]
                    # eV^-1
                    for jax in range(3*nat):
                        # eV/ps^2*eV*ps^2*eV^-1 = eV
                        # ph. resolved
                        if p.ph_resolved:
                            ii = p.wql_grid_index[iq,il]
                            self.acf_wql_sp[:,ii,iT] += wq[iq] * wq[iqp] * A_lq ** 2 * A_lqp[iqlp] ** 2 * fw[:] * Fjax_llp[jax] * Fjax_llp[jax].conjugate()
                            if il in p.phm_list:
                                iph = p.phm_list.index(il)
                                self.acf_phr_sp[:,iph,iT] += wq[iq] * wq[iqp] * A_lq ** 2 * A_lqp[iqlp] ** 2 * fw[:] * Fjax_llp[jax] * Fjax_llp[jax].conjugate()
                        # at. resolved
                        if p.at_resolved:
                            ia = atoms.index_to_ia_map[jax] - 1
                            self.acf_atr_sp[:,ia,iT] += wq[iq] * wq[iqp] * A_lq ** 2 * A_lqp[iqlp] ** 2 * fw[:] * Fjax_llp[jax] * Fjax_llp[jax].conjugate()
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
# -------------------------------------------------------------------------
#
#              GPU CLASS
#
# -------------------------------------------------------------------------
class GPU_acf_sp_ph(acf_sp_ph):
    def __init__(self):
        super(GPU_acf_sp_ph, self).__init__()
        # set up constants
        # gpu inputs are capitalized - no space
        self.THZTOEV = np.double(THz_to_ev)
        self.KB = np.double(kb)
        self.TOLER = np.double(eps)
        # MINFREQ
        self.MINFREQ = np.double(p.min_freq)
    #
    # compute <Delta V^(1)(t) Delta V^(1)(t')>
    def compute_acf_V1_oft(self, wq, wu, ql_list, A_lq, F_lq):
        # initialize acf_sp -> 0 acf -> 1 integral
        self.acf_sp = np.zeros((p.nt, 2, p.ntmp),dtype=np.complex128)
        '''
        read GPU code
        '''
        gpu_src = Path('./pydephasing/gpu_source/compute_acf_V1.cu').read_text()
        mod = SourceModule(gpu_src)
        compute_acf = mod.get_function("compute_acf_V1_oft")
        # split modes on grid
        QL_LIST, INIT, LGTH = gpu.split_data_on_grid(range(len(ql_list)))
        # dE (ps^-1)
        dE = self.dE / hbar
        DE = np.double(dE)
        nu = p.eta / hbar
        NU = np.double(nu)
        # ps^-1
        # build input arrays
        WQ = np.zeros(len(ql_list), dtype=np.double)
        WUQ= np.zeros(len(ql_list), dtype=np.double)
        F_LQ= np.zeros(len(ql_list), dtype=np.complex128)
        A_LQ= np.zeros(len(ql_list), dtype=np.double)
        # run over (q,l) modes
        iql = 0
        for iq, il in ql_list:
            F_LQ[iql] = F_lq[iql]
            A_LQ[iql] = A_lq[iql]
            WQ[iql] = wq[iq]
            WUQ[iql]= wu[iq][il]
            iql += 1
        # temperatures
        for iT in range(p.ntmp):
            T = np.double(p.temperatures[iT])
            # iterate over time intervals
            t0 = 0
            while (t0 < p.nt):
                t1 = t0 + gpu.BLOCK_SIZE[0]*gpu.BLOCK_SIZE[1]*gpu.BLOCK_SIZE[2]
                size = min(t1, p.nt) - t0
                SIZE = np.int32(size)
                # acf array
                ACF = np.zeros(gpu.gpu_size, dtype=np.complex128)
                ACF_INT = np.zeros(gpu.gpu_size, dtype=np.complex128)
                TIME= np.zeros(size, dtype=np.double)
                for t in range(t0, min(t1,p.nt)):
                    TIME[t-t0] = p.time[t]
                # call function
                compute_acf(cuda.In(INIT), cuda.In(LGTH), cuda.In(QL_LIST), SIZE, cuda.In(TIME),
                            cuda.In(WQ), cuda.In(WUQ), cuda.In(A_LQ), cuda.In(F_LQ), T, DE, NU, 
                            self.MINFREQ, self.THZTOEV, self.KB, self.TOLER, cuda.Out(ACF), cuda.Out(ACF_INT),
                            block=gpu.block, grid=gpu.grid)
                ACF = gpu.recover_data_from_grid(ACF)
                ACF_INT = gpu.recover_data_from_grid(ACF_INT)
                for t in range(t0, min(t1,p.nt)):
                    self.acf_sp[t,0,iT] += ACF[t-t0]
                    self.acf_sp[t,1,iT] += ACF_INT[t-t0]
                t0 = t1
        #plt.plot(p.time, self.acf_sp[:,0,0])
        plt.plot(p.time, self.acf_sp[:,1,0].real)
        plt.show()
    #
    # compute <Delta V(1) \Delta V(1)>(w)
    def compute_acf_V1_ofw(self, wq, wu, ql_list, A_lq, F_lq):
        # initialize acf_sp -> 0 acf -> 1 integral
        self.acf_sp = np.zeros((p.nwg, p.ntmp),dtype=np.complex128)
        '''
        read GPU code
        '''
        gpu_src = Path('./pydephasing/gpu_source/compute_acf_V1.cu').read_text()
        mod = SourceModule(gpu_src)
        compute_acf = mod.get_function("compute_acf_V1_ofw")
        # split modes on grid
        QL_LIST, INIT, LGTH = gpu.split_data_on_grid(range(len(ql_list)))
        DE = np.double(self.dE)
        ETA = np.double(p.eta)
        # eV
        # build input arrays
        WQ = np.zeros(len(ql_list), dtype=np.double)
        WUQ= np.zeros(len(ql_list), dtype=np.double)
        F_LQ= np.zeros(len(ql_list), dtype=np.complex128)
        A_LQ= np.zeros(len(ql_list), dtype=np.double)
        # run over (q,l) modes
        iql = 0
        for iq, il in ql_list:
            F_LQ[iql] = F_lq[iql]
            A_LQ[iql] = A_lq[iql]
            WQ[iql] = wq[iq]
            WUQ[iql]= wu[iq][il]
            iql += 1
        # temperatures
        for iT in range(p.ntmp):
            T = np.double(p.temperatures[iT])
            # iterate over freq. intervals
            iw0 = 0
            while (iw0 < p.nwg):
                iw1 = iw0 + gpu.BLOCK_SIZE[0]*gpu.BLOCK_SIZE[1]*gpu.BLOCK_SIZE[2]
                size = min(iw1, p.nwg) - iw0
                SIZE = np.int32(size)
                # ACFW array
                ACFW = np.zeros(gpu.gpu_size, dtype=np.complex128)
                # freq. array
                WG = np.zeros(size, dtype=np.double)
                for iw in range(iw0, min(iw1,p.nwg)):
                    WG[iw-iw0] = p.w_grid[iw]
                    # eV
                # call device function
                compute_acf(cuda.In(INIT), cuda.In(LGTH), cuda.In(QL_LIST), SIZE, cuda.In(WG),
                            cuda.In(WQ), cuda.In(WUQ), cuda.In(A_LQ), cuda.In(F_LQ), T, DE,
                            self.MINFREQ, self.THZTOEV, self.KB, self.TOLER, ETA, cuda.Out(ACFW),
                            block=gpu.block, grid=gpu.grid)
                ACFW = gpu.recover_data_from_grid(ACFW)
                for iw in range(iw0, min(iw1, p.nwg)):
                    self.acf_sp[iw,iT] = ACFW[iw-iw0]
                iw0 = iw1
        import matplotlib.pyplot as plt
        plt.plot(p.w_grid, self.acf_sp[:,0].real)
        plt.show()
    #
    # compute <Delta V^(1)(t) Delta V^(1)(t')> -> ph / at resolved
    def compute_acf_V1_atphr_oft(self, nat, wq, wu, ql_list, A_lq, Fjax_lq):
        # load files
        gpu_src = Path('./pydephasing/gpu_source/compute_acf_V1.cu').read_text()
        mod = SourceModule(gpu_src)
        # CHECK ph/at res.
        if p.ph_resolved:
            self.acf_phr_sp = np.zeros((p.nt2, 2, p.nphr, p.ntmp), dtype=np.complex128)
            self.acf_wql_sp = np.zeros((p.nt2, 2, p.nwbn, p.ntmp), dtype=np.complex128)
            # load files
            compute_acf_phr = mod.get_function("compute_acf_V1_phr_oft")
        if p.at_resolved:
            NMODES = np.int32(len(ql_list))
            NAT = np.int32(nat)
            self.acf_atr_sp = np.zeros((p.nt2, 2, nat, p.ntmp), dtype=np.complex128)
            # load files
            compute_acf_atr = mod.get_function("compute_acf_V1_atr_oft")
        if not p.ph_resolved and not p.at_resolved:
            return
        dE = self.dE / hbar
        DE = np.double(dE)
        nu = p.eta / hbar
        NU = np.double(nu)
        # ps^-1
        # run over (jax,q,l) modes index
        # effective force
        if p.at_resolved:
            FJAX_LQ = np.zeros(3*nat*len(ql_list), dtype=np.complex128)
            ii = 0
            for iql in range(len(ql_list)):
                for jax in range(3*nat):
                    FJAX_LQ[ii] = Fjax_lq[jax,iql]
                    ii += 1
        if p.ph_resolved:
            F_LQ = np.zeros(len(ql_list), dtype=np.complex128)
            for jax in range(3*nat):
                F_LQ[:] += Fjax_lq[jax,:]
        # set other arrays
        WQ = np.zeros(len(ql_list), dtype=np.double)
        WUQ= np.zeros(len(ql_list), dtype=np.double)
        A_LQ = np.zeros(len(ql_list), dtype=np.double)
        iql = 0
        for iq, il in ql_list:
            WQ[iql] = wq[iq]
            WUQ[iql]= wu[iq][il]
            A_LQ[iql] = A_lq[iql]
            iql += 1
        # run over temperatures
        for iT in range(p.ntmp):
            T = np.double(p.temperatures[iT])
            # iterate over it intervals
            t0 = 0
            while (t0 < p.nt2):
                t1 = t0 + gpu.BLOCK_SIZE[0]*gpu.BLOCK_SIZE[1]*gpu.BLOCK_SIZE[2]
                size = min(t1, p.nt2) - t0
                SIZE = np.int32(size)
                # TIME
                TIME = np.zeros(size, dtype=np.double)
                for t in range(t0, min(t1,p.nt2)):
                    TIME[t-t0] = p.time2[t]
                # ATOM RESOLVED
                #
                if p.at_resolved:
                    # run over atoms
                    ia0 = 0
                    while ia0 < nat:
                        ia1 = ia0 + gpu.GRID_SIZE[0]*gpu.GRID_SIZE[1]
                        na = min(ia1, nat) - ia0
                        AT_LIST = np.zeros(na, dtype=np.int32)
                        for a in range(ia0, min(ia1, nat)):
                            AT_LIST[a-ia0] = a
                        NA_SIZE = np.int32(na)
                        # ACF array
                        ACF = np.zeros(gpu.gpu_size, dtype=np.complex128)
                        ACF_INT = np.zeros(gpu.gpu_size, dtype=np.complex128)
                        # compute ACF
                        compute_acf_atr(cuda.In(AT_LIST), cuda.In(WQ), cuda.In(WUQ), cuda.In(TIME), DE, NU,
                                        cuda.In(FJAX_LQ), cuda.In(A_LQ), SIZE, NA_SIZE, NMODES, NAT,
                                        T, self.MINFREQ, self.THZTOEV, self.KB, self.TOLER, cuda.Out(ACF),
                                        cuda.Out(ACF_INT), block=gpu.block, grid=gpu.grid)
                        # (eV^2 ps) units
                        ACF = gpu.recover_data_from_grid_apr(ACF, na, size)
                        ACF_INT = gpu.recover_data_from_grid_apr(ACF_INT, na, size)
                        for t in range(t0, min(t1,p.nt2)):
                            for a in range(ia0, min(ia1,nat)):
                                self.acf_atr_sp[t,0,a,iT] += ACF[t-t0,a-ia0]
                                self.acf_atr_sp[t,1,a,iT] += ACF_INT[t-t0,a-ia0]
                        ia0 = ia1
                # PH. RESOLVED
                #
                if p.ph_resolved:
                    # run over ph.
                    iph0 = 0
                    while iph0 < len(ql_list):
                        iph1 = iph0 + gpu.GRID_SIZE[0]*gpu.GRID_SIZE[1]
                        nph = min(iph1,len(ql_list)) - iph0
                        PH_LST = np.zeros(nph, dtype=np.int32)
                        for iph in range(iph0,min(iph1,len(ql_list))):
                            PH_LST[iph-iph0] = iph
                        NPH = np.int32(nph)
                        # ACF ARRAYS
                        ACF = np.zeros(gpu.gpu_size, dtype=np.complex128)
                        ACF_INT = np.zeros(gpu.gpu_size, dtype=np.complex128)
                        # compute ACF
                        compute_acf_phr(NPH, cuda.In(PH_LST), SIZE, cuda.In(TIME), cuda.In(WQ), cuda.In(WUQ), 
                                    cuda.In(A_LQ), cuda.In(F_LQ), T, DE, NU, self.MINFREQ, self.THZTOEV, 
                                    self.KB, self.TOLER, cuda.Out(ACF), cuda.Out(ACF_INT), block=gpu.block, grid=gpu.grid)
                        ACF = gpu.recover_data_from_grid_apr(ACF, nph, size)
                        ACF_INT = gpu.recover_data_from_grid_apr(ACF_INT, nph, size)
                        for t in range(t0, min(t1,p.nt2)):
                            for iph in range(iph0, min(iph1,len(ql_list))):
                                iq, il = ql_list[iph]
                                # wql grid
                                iwql = p.wql_grid_index[iq,il]
                                self.acf_wql_sp[:,0,iwql,iT] += ACF[t-t0,iph-iph0]
                                self.acf_wql_sp[:,1,iwql,iT] += ACF_INT[t-t0,iph-iph0]
                                # phr
                                if il in p.phm_list:
                                    iphr = p.phm_list.index(il)
                                    self.acf_phr_sp[:,0,iphr,iT] += ACF[t-t0,iph-iph0]
                                    self.acf_phr_sp[:,1,iphr,iT] += ACF_INT[t-t0,iph-iph0]
                        iph0 = iph1
                t0 = t1
    # compute <Delta V^(1) Delta V^(1)>(w) -> ph / at resolved
    def compute_acf_V1_atphr_ofw(self, nat, wq, wu, ql_list, A_lq, Fjax_lq):
        # load files
        gpu_src = Path('./pydephasing/gpu_source/compute_acf_V1.cu').read_text()
        mod = SourceModule(gpu_src)
        # CHECK at/ph resolved
        if p.ph_resolved:
            self.acf_phr_sp = np.zeros((p.nwg, p.nphr, p.ntmp), dtype=np.complex128)
            self.acf_wql_sp = np.zeros((p.nwg, p.nwbn, p.ntmp), dtype=np.complex128)
            # load files
            compute_acf_phr = mod.get_function("compute_acf_V1_phr_ofw")
        if p.at_resolved:
            NMODES = np.int32(len(ql_list))
            NAT = np.int32(nat)
            self.acf_atr_sp = np.zeros((p.nwg, nat, p.ntmp), dtype=np.complex128)
            # load files
            compute_acf_atr = mod.get_function("compute_acf_V1_atr_ofw")
        if not p.ph_resolved and not p.at_resolved:
            return
        DE = np.double(self.dE)
        ETA= np.double(p.eta)
        # eV units
        # run over (jax,q,l) modes index
        # effective force
        if p.at_resolved:
            FJAX_LQ = np.zeros(3*nat*len(ql_list), dtype=np.complex128)
            ii = 0
            for iql in range(len(ql_list)):
                for jax in range(3*nat):
                    FJAX_LQ[ii] = Fjax_lq[jax,iql]
                    ii += 1
        if p.ph_resolved:
            F_LQ = np.zeros(len(ql_list), dtype=np.complex128)
            for jax in range(3*nat):
                F_LQ[:] += Fjax_lq[jax,:]
        # set remaining arrays
        WQ = np.zeros(len(ql_list), dtype=np.double)
        WUQ= np.zeros(len(ql_list), dtype=np.double)
        A_LQ= np.zeros(len(ql_list), dtype=np.double)
        iql = 0
        for iq, il in ql_list:
            WQ[iql] = wq[iq]
            WUQ[iql]= wu[iq][il]
            A_LQ[iql] = A_lq[iql]
            iql += 1
        # run over temperatures
        for iT in range(p.ntmp):
            T = np.double(p.temperatures[iT])
            # iterate over w
            w0 = 0
            while (w0 < p.nwg):
                w1 = w0 + gpu.BLOCK_SIZE[0]*gpu.BLOCK_SIZE[1]*gpu.BLOCK_SIZE[2]
                size = min(w1, p.nwg) - w0
                SIZE = np.int32(size)
                # FREQ
                WG = np.zeros(size, dtype=np.double)
                for w in range(w0, min(w1,p.nwg)):
                    WG[w-w0] = p.w_grid[w]
                # ATOM RESOLVED
                #
                if p.at_resolved:
                    # run over atoms
                    ia0 = 0
                    while ia0 < nat:
                        ia1 = ia0 + gpu.GRID_SIZE[0]*gpu.GRID_SIZE[1]
                        na = min(ia1,nat) - ia0
                        AT_LST = np.zeros(na, dtype=np.int32)
                        for a in range(ia0, min(ia1,nat)):
                            AT_LST[a-ia0] = a
                        NA_SIZE = np.int32(na)
                        # ACF array
                        ACFW = np.zeros(gpu.gpu_size, dtype=np.complex128)
                        # compute ACF
                        compute_acf_atr(NA_SIZE, cuda.In(AT_LST), SIZE, cuda.In(WG), cuda.In(WQ),
                                        cuda.In(WUQ), cuda.In(A_LQ), cuda.In(FJAX_LQ), T, DE, NMODES, NAT,
                                        self.MINFREQ, self.THZTOEV, self.KB, self.TOLER, ETA, cuda.Out(ACFW),
                                        block=gpu.block, grid=gpu.grid)
                        # eV units
                        ACFW = gpu.recover_data_from_grid_apr(ACFW, na, size)
                        for w in range(w0, min(w1,p.nwg)):
                            for a in range(ia0, min(ia1,nat)):
                                self.acf_atr_sp[w,a,iT] += ACFW[w-w0,a-ia0]
                        ia0 = ia1
                # PH. RESOLVED
                #
                if p.ph_resolved:
                    # run over ph. modes
                    iph0 = 0
                    while iph0 < len(ql_list):
                        iph1 = iph0 + gpu.GRID_SIZE[0]*gpu.GRID_SIZE[1]
                        nph = min(iph1,len(ql_list)) - iph0
                        PH_LST = np.zeros(nph, dtype=np.int32)
                        for iph in range(iph0,min(iph1,len(ql_list))):
                            PH_LST[iph-iph0] = iph
                        NPH = np.int32(nph)
                        # ACF array
                        ACFW = np.zeros(gpu.gpu_size, dtype=np.complex128)
                        # compute ACFW
                        compute_acf_phr(NPH, cuda.In(PH_LST), SIZE, cuda.In(WG), cuda.In(WQ),
                                        cuda.In(WUQ), cuda.In(A_LQ), cuda.In(F_LQ), T, DE, self.MINFREQ,
                                        self.THZTOEV, self.KB, self.TOLER, ETA, cuda.Out(ACFW), block=gpu.block, grid=gpu.grid)
                        ACFW = gpu.recover_data_from_grid_apr(ACFW, nph, size)
                        for w in range(w0, min(w1,p.nwg)):
                            for iph in range(iph0, min(iph1,len(ql_list))):
                                iq, il = ql_list[iph]
                                # wql grid
                                iwql = p.wql_grid_index[iq,il]
                                self.acf_wql_sp[:,iwql,iT] += ACFW[w-w0,iph-iph0]
                                # phr
                                if il in p.phm_list:
                                    iphr = p.phm_list.index(il)
                                    self.acf_phr_sp[:,iphr,iT] += ACFW[w-w0,iph-iph0]
                        iph0 = iph1
                w0 = w1
    #
    # GPU equivalent function (order 2)
    def compute_acf_V2_oft(self, wq, wu, iq, il, qlp_list, A_lq, A_lqp, F_lqlqp):
        '''
        compute ACF partial sum
        '''
        gpu_src = Path('./pydephasing/gpu_source/compute_acf_V2.cu').read_text()
        mod = SourceModule(gpu_src)
        compute_acf = mod.get_function("compute_acf_V2_oft")
        # split modes on grid
        QLP_LIST, INIT, LGTH = gpu.split_data_on_grid(range(len(qlp_list)))
        # dE (ps^-1)
        dE = self.dE / hbar
        DE = np.double(dE)
        nu = p.eta / hbar
        NU = np.double(nu)
        # ps^-1
        # check energy Eql
        if wu[iq][il] > p.min_freq:
            # set (q,l) variables
            WQ = np.double(wq[iq])
            WUQ= np.double(wu[iq][il])
            ALQ= np.double(A_lq)
            # build input arrays
            WQP= np.zeros(len(qlp_list), dtype=np.double)
            WUQP = np.zeros(len(qlp_list), dtype=np.double)
            ALQP = np.zeros(len(qlp_list), dtype=np.double)
            FLQLQP = np.zeros(len(qlp_list), dtype=np.complex128)
            iqlp = 0
            for iqp, ilp in qlp_list:
                WQP[iqlp] = wq[iqp]
                WUQP[iqlp]= wu[iqp][ilp]
                ALQP[iqlp]= A_lqp[iqlp]
                FLQLQP[iqlp]= F_lqlqp[iqlp]
                iqlp += 1
            # run over
            #  temperature
            for iT in range(p.ntmp):
                T = np.double(p.temperatures[iT])
                # iterate over time variable
                t0 = 0
                while (t0 < p.nt):
                    t1 = t0 + gpu.BLOCK_SIZE[0]*gpu.BLOCK_SIZE[1]*gpu.BLOCK_SIZE[2]
                    size = min(t1, p.nt) - t0
                    SIZE = np.int32(size)
                    # set ACF array
                    ACF = np.zeros(gpu.gpu_size, dtype=np.complex128)
                    ACF_INT = np.zeros(gpu.gpu_size, dtype=np.complex128)
                    TIME = np.zeros(size, dtype=np.double)
                    for t in range(t0, min(t1,p.nt)):
                        TIME[t-t0] = p.time[t]
                    # call function
                    compute_acf(cuda.In(INIT), cuda.In(LGTH), cuda.In(QLP_LIST), SIZE, cuda.In(TIME),
                        WQ, cuda.In(WQP), WUQ, cuda.In(WUQP), ALQ, cuda.In(ALQP), cuda.In(FLQLQP),
                        T, DE, NU, self.MINFREQ, self.THZTOEV, self.KB, self.TOLER, 
                        cuda.Out(ACF), cuda.Out(ACF_INT), block=gpu.block, grid=gpu.grid)
                    ACF = gpu.recover_data_from_grid(ACF)
                    ACF_INT = gpu.recover_data_from_grid(ACF_INT)
                    for t in range(t0, min(t1,p.nt)):
                        self.acf_sp[t,0,iT] += ACF[t-t0]
                        self.acf_sp[t,1,iT] += ACF_INT[t-t0]
                    t0 = t1
    #
    # acf (2) of w
    def compute_acf_V2_ofw(self, wq, wu, iq, il, qlp_list, A_lq, A_lqp, F_lqlqp):
        '''
        compute ACF partial sum
        '''
        gpu_src = Path('./pydephasing/gpu_source/compute_acf_V2.cu').read_text()
        mod = SourceModule(gpu_src)
        compute_acf = mod.get_function("compute_acf_V2_ofw")
        # split modes on grid
        QLP_LIST, INIT, LGTH = gpu.split_data_on_grid(range(len(qlp_list)))
        DE = np.double(self.dE)
        ETA= np.double(p.eta)
        # eV
        # check wql
        if wu[iq][il] > p.min_freq:
            # set (q,l) variables
            WQ = np.double(wq[iq])
            WUQ= np.double(wu[iq][il])
            ALQ= np.double(A_lq)
            # input arrays
            WQP = np.zeros(len(qlp_list), dtype=np.double)
            WUQP= np.zeros(len(qlp_list), dtype=np.double)
            ALQP= np.zeros(len(qlp_list), dtype=np.double)
            FLQLQP= np.zeros(len(qlp_list), dtype=np.complex128)
            iqlp = 0
            for iqp, ilp in qlp_list:
                WQP[iqlp] = wq[iqp]
                WUQP[iqlp]= wu[iqp][ilp]
                ALQP[iqlp]= A_lqp[iqlp]
                FLQLQP[iqlp]= F_lqlqp[iqlp]
                iqlp += 1
            # iterate temperature
            for iT in range(p.ntmp):
                T = np.double(p.temperatures[iT])
                # iterate over w variable
                iw0 = 0
                while (iw0 < p.nwg):
                    iw1 = iw0 + gpu.BLOCK_SIZE[0]*gpu.BLOCK_SIZE[1]*gpu.BLOCK_SIZE[2]
                    size = min(iw1, p.nwg) - iw0
                    SIZE = np.int32(size)
                    # ACFW array
                    ACFW = np.zeros(gpu.gpu_size, dtype=np.complex128)
                    # local freq. array
                    WG = np.zeros(size, dtype=np.double)
                    for iw in range(iw0, min(iw1,p.nwg)):
                        WG[iw-iw0] = p.w_grid[iw]
                        # eV
                    # call GPU function
                    compute_acf(cuda.In(INIT), cuda.In(LGTH), cuda.In(QLP_LIST), SIZE, cuda.In(WG),
                                WQ, cuda.In(WQP), WUQ, cuda.In(WUQP), ALQ, cuda.In(ALQP), cuda.In(FLQLQP),
                                T, DE, self.MINFREQ, self.THZTOEV, self.KB, self.TOLER, ETA, cuda.Out(ACFW),
                                block=gpu.block, grid=gpu.grid)
                    ACFW = gpu.recover_data_from_grid(ACFW)
                    for iw in range(iw0, min(iw1,p.nwg)):
                        self.acf_sp[iw,iT] += ACFW[iw-iw0]
                    iw0 = iw1
    #
    # compute <Delta V^(2)(t) Delta V^(2)(t')> -> ph / at resolved
    def compute_acf_V2_atphr_oft(self, nat, wq, wu, iq, il, qlp_list, A_lq, A_lqp, Fjax_lqlqp):
        # load files
        gpu_src = Path('./pydephasing/gpu_source/compute_acf_V2.cu').read_text()
        mod = SourceModule(gpu_src)
        # CALC. type
        if p.ph_resolved:
            compute_acf_phr = mod.get_function("compute_acf_V2_phr_oft")
            # split modes on grid
            QLP_LIST, INIT, LGTH = gpu.split_data_on_grid(range(len(qlp_list)))
            # ph. index
            iwql = p.wql_grid_index[iq,il]
            if il in p.phm_list:
                iphr = p.phm_list.index(il)
        if p.at_resolved:
            NMODES = np.int32(len(qlp_list))
            NAT = np.int32(nat)
            # load function
            compute_acf_atr = mod.get_function("compute_acf_V2_atr_oft")
        if not p.ph_resolved and not p.at_resolved:
            return
        dE = self.dE / hbar
        DE = np.double(dE)
        nu = p.eta / hbar
        NU = np.double(nu)
        # ps^-1
        # CHECK Eql
        if wu[iq][il] > p.min_freq:
            # set (q,l) variables
            WQ = np.double(wq[iq])
            WUQ= np.double(wu[iq][il])
            ALQ= np.double(A_lq)
            # extract eff. force
            if p.at_resolved:
                FJAX_LQLQP = np.zeros(3*nat*len(qlp_list), dtype=np.complex128)
                ii = 0
                for iqlp in range(len(qlp_list)):
                    for jax in range(3*nat):
                        FJAX_LQLQP[ii] = Fjax_lqlqp[jax,iqlp]
                        ii += 1
            if p.ph_resolved:
                F_LQLQP = np.zeros(len(qlp_list), dtype=np.complex128)
                for jax in range(3*nat):
                    F_LQLQP[:] += Fjax_lqlqp[jax,:]
            # input arrays
            WQP = np.zeros(len(qlp_list), dtype=np.double)
            WUQP= np.zeros(len(qlp_list), dtype=np.double)
            ALQP= np.zeros(len(qlp_list), dtype=np.double)
            iqlp = 0
            for iqp, ilp in qlp_list:
                WQP[iqlp] = wq[iqp]
                WUQP[iqlp]= wu[iqp][ilp]
                ALQP[iqlp]= A_lqp[iqlp]
                iqlp += 1
            # temperature cycle
            for iT in range(p.ntmp):
                T = np.double(p.temperatures[iT])
                # iterate over time
                t0 = 0
                while (t0 < p.nt2):
                    t1 = t0 + gpu.BLOCK_SIZE[0]*gpu.BLOCK_SIZE[1]*gpu.BLOCK_SIZE[2]
                    size = min(t1, p.nt2) - t0
                    SIZE = np.int32(size)
                    # TIME
                    TIME = np.zeros(size, dtype=np.double)
                    for t in range(t0, min(t1,p.nt2)):
                        TIME[t-t0] = p.time2[t]
                    #
                    # ATOM RES.
                    if p.at_resolved:
                        ia0 = 0
                        while ia0 < nat:
                            ia1 = ia0 + gpu.GRID_SIZE[0]*gpu.GRID_SIZE[1]
                            na = min(ia1,nat) - ia0
                            AT_LIST = np.zeros(na, dtype=np.int32)
                            for a in range(ia0, min(ia1,nat)):
                                AT_LIST[a-ia0] = a
                            NA_SIZE = np.int32(na)
                            # ACF array
                            ACF = np.zeros(gpu.gpu_size, dtype=np.complex128)
                            ACF_INT = np.zeros(gpu.gpu_size, dtype=np.complex128)
                            # compute ACF
                            compute_acf_atr(cuda.In(AT_LIST), NA_SIZE, cuda.In(TIME), SIZE, NMODES,
                                            NAT, DE, NU, WQ, WUQ, ALQ, cuda.In(WQP), cuda.In(WUQP),
                                            cuda.In(ALQP), cuda.In(FJAX_LQLQP), T, self.MINFREQ,
                                            self.THZTOEV, self.KB, self.TOLER, cuda.Out(ACF), cuda.Out(ACF_INT),
                                            block=gpu.block, grid=gpu.grid)
                            # (eV^2 ps) -> ACF_INT
                            ACF = gpu.recover_data_from_grid_apr(ACF, na, size)
                            ACF_INT = gpu.recover_data_from_grid_apr(ACF_INT, na, size)
                            for t in range(t0, min(t1,p.nt2)):
                                for a in range(ia0, min(ia1,nat)):
                                    self.acf_atr_sp[t,0,a,iT] += ACF[t-t0,a-ia0]
                                    self.acf_atr_sp[t,1,a,iT] += ACF_INT[t-t0,a-ia0]
                            ia0 = ia1
                    #
                    # PH. RES.
                    if p.ph_resolved:
                        # ACF ARRAYS
                        ACF = np.zeros(gpu.gpu_size, dtype=np.complex128)
                        ACF_INT = np.zeros(gpu.gpu_size, dtype=np.complex128)
                        # compute ACF
                        compute_acf_phr(cuda.In(INIT), cuda.In(LGTH), cuda.In(QLP_LIST), cuda.In(TIME),
                                        SIZE, DE, NU, WQ, WUQ, ALQ, cuda.In(WQP), cuda.In(WUQP), 
                                        cuda.In(ALQP), cuda.In(F_LQLQP), T, self.MINFREQ, self.THZTOEV, 
                                        self.KB, self.TOLER, cuda.Out(ACF), cuda.Out(ACF_INT), 
                                        block=gpu.block, grid=gpu.grid)
                        ACF = gpu.recover_data_from_grid(ACF)
                        ACF_INT = gpu.recover_data_from_grid(ACF_INT)
                        for t in range(t0, min(t1,p.nt2)):
                            # wql grid
                            self.acf_wql_sp[:,0,iwql,iT] += ACF[t-t0]
                            self.acf_wql_sp[:,1,iwql,iT] += ACF_INT[t-t0]
                            # phr
                            if il in p.phm_list:
                                self.acf_phr_sp[:,0,iphr,iT] += ACF[t-t0]
                                self.acf_phr_sp[:,1,iphr,iT] += ACF_INT[t-t0]
                    t0 = t1
    #
    # compute <Delta V^(2) Delta V^(2)>(w) -> ph / at resolved
    def compute_acf_V2_atphr_ofw(self, nat, wq, wu, iq, il, qlp_list, A_lq, A_lqp, Fjax_lqlqp):
        # load files
        gpu_src = Path('./pydephasing/gpu_source/compute_acf_V2.cu').read_text()
        mod = SourceModule(gpu_src)
        # CALC. type
        if p.ph_resolved:
            compute_acf_phr = mod.get_function("compute_acf_V2_phr_ofw")
            # split modes on grid
            QLP_LIST, INIT, LGTH = gpu.split_data_on_grid(range(len(qlp_list)))
            # ph. index
            iwql = p.wql_grid_index[iq,il]
            if il in p.phm_list:
                iphr = p.phm_list.index(il)
        if p.at_resolved:
            NMODES = np.int32(len(qlp_list))
            NAT = np.int32(nat)
            # load function
            compute_acf_atr = mod.get_function("compute_acf_V2_atr_ofw")
        if not p.ph_resolved and not p.at_resolved:
            return
        DE = np.double(self.dE)
        ETA= np.double(p.eta)
        # eV units
        # CHECK Eql
        if wu[iq][il] > p.min_freq:
            # set (q,l) variables
            WQ = np.double(wq[iq])
            WUQ= np.double(wu[iq][il])
            ALQ= np.double(A_lq)
            # extract eff. force
            if p.at_resolved:
                FJAX_LQLQP = np.zeros(3*nat*len(qlp_list), dtype=np.complex128)
                ii = 0
                for iqlp in range(len(qlp_list)):
                    for jax in range(3*nat):
                        FJAX_LQLQP[ii] = Fjax_lqlqp[jax,iqlp]
                        ii += 1
            if p.ph_resolved:
                F_LQLQP = np.zeros(len(qlp_list), dtype=np.complex128)
                for jax in range(3*nat):
                    F_LQLQP[:] += Fjax_lqlqp[jax,:]
            # input arrays
            WQP = np.zeros(len(qlp_list), dtype=np.double)
            WUQP= np.zeros(len(qlp_list), dtype=np.double)
            ALQP= np.zeros(len(qlp_list), dtype=np.double)
            iqlp = 0
            for iqp, ilp in qlp_list:
                WQP[iqlp] = wq[iqp]
                WUQP[iqlp]= wu[iqp][ilp]
                ALQP[iqlp]= A_lqp[iqlp]
                iqlp += 1
            # temperature cycle
            for iT in range(p.ntmp):
                T = np.double(p.temperatures[iT])
                # iterate over w
                w0 = 0
                while (w0 < p.nwg):
                    w1 = w0 + gpu.BLOCK_SIZE[0]*gpu.BLOCK_SIZE[1]*gpu.BLOCK_SIZE[2]
                    size = min(w1, p.nwg) - w0
                    SIZE = np.int32(size)
                    # FREQ.
                    WG = np.zeros(size, dtype=np.double)
                    for w in range(w0, min(w1,p.nwg)):
                        WG[w-w0] = p.w_grid[w]
                    # ATOM RESOLVED
                    #
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