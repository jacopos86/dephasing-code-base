#
# This module sets up the methods
# needed to compute the energy / spin fluctuations
# auto correlation function
#
import numpy as np
import logging
from pydephasing.phys_constants import THz_to_ev, eps
from pydephasing.T2_calc import T2_eval
from pydephasing.log import log
from pydephasing.mpi import mpi
from pydephasing.input_parameters import p
from pydephasing.utility_functions import bose_occup
from pydephasing.extract_ph_data import set_ql_list_red_qgrid, set_iqlp_list
from tqdm import tqdm
from pydephasing.ph_resolved_quant import compute_ph_amplitude_q, transf_1st_order_force_phr, phr_force_2nd_order
import matplotlib.pyplot as plt
from pydephasing.global_params import GPU_ACTIVE
from pydephasing.utility_functions import print_acf_dict
import sys
#
# spin-exc-ph dephasing class -> homogeneous
class acf_ph(object):
    def __init__(self):
        # output dir
        self.write_dir = p.write_dir
        # auto correlation functions
        self.acf     = None
        self.acf_phr = None
        self.acf_wql = None
        self.acf_atr = None
        # local proc. acf
        self.acf_sp     = None
        self.acf_phr_sp = None
        self.acf_wql_sp = None
        self.acf_atr_sp = None
        # Delta^2
        self.Delta_2 = None
        self.Delta_w0= None
    #
    #  instance CPU / GPU
    #           deph / relax
    def generate_instance(self):
        from pydephasing.auto_correlation_spph_mod import GPU_acf_sp_ph, CPU_acf_sp_ph
        if GPU_ACTIVE:
            return GPU_acf_sp_ph ()
        else:
            return CPU_acf_sp_ph ()
    #
    # driver for acf - order 1 autocorrelation
    # see Eq. (20) and Eq. (60) in notes
    def compute_acf_1_driver(self, nat, wq, wu, ql_list, A_lq, Fjax_lq):
        # compute F_lq matrix elements
        F_lq = np.zeros(len(ql_list), dtype=np.complex128)
        for jax in range(3*nat):
            F_lq[:] += Fjax_lq[jax,:]
        # compute <V(1)(t) V(1)(t')>
        if p.time_resolved:
            self.compute_acf_V1_oft(wq, wu, ql_list, A_lq, F_lq)
        if p.w_resolved:
            self.compute_acf_V1_ofw(wq, wu, ql_list, A_lq, F_lq)
        # ph. / atom resolved
        if p.ph_resolved or p.at_resolved:
            if p.time_resolved:
                self.compute_acf_V1_atphr_oft(nat, wq, wu, ql_list, A_lq, Fjax_lq)
            if p.w_resolved:
                self.compute_acf_V1_atphr_ofw(nat, wq, wu, ql_list, A_lq, Fjax_lq)
        #
        # check Delta^2 value
        if log.level <= logging.INFO:
            if p.time_resolved:
                self.Delta_2 = np.zeros(p.ntmp)
                self.Delta_2 = self.compute_acf_V1_t0(wq, wu, ql_list, A_lq, F_lq)
            if p.w_resolved:
                self.Delta_w0= np.zeros(p.ntmp)
                self.Delta_w0= self.compute_acf_V1_w0(wq, wu, ql_list, A_lq, F_lq)
    #
    # driver for acf - order 2 autocorrelation
    # see Eq. (34) and Eq. (73) in notes
    def compute_acf_2_driver(self, nat, wq, u, wu, ql_list, qlp_list_full, qmq_map, A_lq, eff_force_obj, H):
        iql = 0
        # run over external (q,l) pair list -> distributed over different
        # processors
        for iq, il in tqdm(ql_list):
            # set the list of (iqp,ilp)
            qlp_list = set_iqlp_list(il, iq, qlp_list_full, wu, H)
            print(len(qlp_list), len(qlp_list_full))
            # update A_lqp with only needed amplitudes
            A_lqp = compute_ph_amplitude_q(wu, nat, qlp_list)
            # compute eff. force
            Fjax_lqlqp = eff_force_obj.transf_2nd_order_force_phr(il, iq, wu, u, nat, qlp_list)
            nlqp = len(qlp_list)
            F_lqlqp = np.zeros((4,nlqp), dtype=np.complex128)
            for jax in range(3*nat):
                F_lqlqp[:,:] += Fjax_lqlqp[:,jax,:]
                # [ps^-2] units
            # ----------------------------------
            #    ACF calculation
            # ----------------------------------
            if p.time_resolved:
                pass
            if p.w_resolved:
                pass
            iql += 1
    #
    # acf(2,t=0)
    def compute_acf_order2_zero_time(self, wq, wu, iq, il, qlp_list, A_lq, A_lqp, F_lqlqp):
        # \sum_lq,lqp(q,qp>0) A_lq^2 A_lqp^2 [1+n_lq+3n_lqp+4n_lq n_lqp] {|F_lq_lqp|^2 + |F_l-q_lqp|^2}
        # eV^2 units
        Delta_2 = np.zeros(p.ntmp, dtype=np.complex128)
        # set wu[q]
        wuq = wu[iq]
        Eql = wuq[il] * THz_to_ev
        if wuq[il] > p.min_freq:
            # run over T
            for iT in range(p.ntmp):
                T = p.temperatures[iT]
                n_ql = bose_occup(Eql, T)
                # run over (qp,ilp)
                iqlp = 0
                for iqp, ilp in qlp_list:
                    # bose occup.
                    wuqp = wu[iqp]
                    Eqlp = wuqp[ilp] * THz_to_ev
                    if wuqp[ilp] > p.min_freq:
                        n_qlp = bose_occup(Eqlp, T)
                        # compute Delta^2
                        A_th = 1 + n_ql + 3 * n_qlp + 4. * n_ql * n_qlp
                        Delta_2[iT] += wq[iq] * wq[iqp] * A_lq ** 2 * A_lqp[iqlp] ** 2 * A_th * F_lqlqp[iqlp] * np.conjugate(F_lqlqp[iqlp])
                    iqlp += 1
        Delta_2r = np.zeros(p.ntmp)
        for iT in range(p.ntmp):
            Delta_2r[iT] = Delta_2[iT].real
        return Delta_2r
    #
    # compute acf parameters
    def compute_acf(self, wq, wu, u, qpts, nat, Fax, Faxby, H):
        # prepare calculation over (q,l) pts. -> first order
        nq = len(qpts)
        ql_list_1 = mpi.split_ph_modes(nq, 3*nat)
        # set dE (relax)
        self.set_dE(H)
        # compute ph. amplitude
        A_lq = compute_ph_amplitude_q(wu, nat, ql_list_1)
        # compute effective force (first order)
        Fjax_lq = transf_1st_order_force_phr(u, qpts, nat, Fax, ql_list_1)
        # call acf_1 driver
        self.compute_acf_1_driver(nat, wq, wu, ql_list_1, A_lq, Fjax_lq)
        # if 2nd order
        if p.order_2_correct:
            # set qlp list (only q, -q excluded)
            ql_list_2, qlp_list_2, qmq_map = set_ql_list_red_qgrid(qpts, nat, wu)
            # complete amplitudes
            A_lq  = compute_ph_amplitude_q(wu, nat, ql_list_2)
            # set 2nd order force object
            eff_force_obj = phr_force_2nd_order().generate_instance()
            eff_force_obj.set_up_2nd_order_force_phr(qpts, Fax, Faxby, H)
            # driver of ACF - order 2 calculation
            self.compute_acf_2_driver(nat, wq, u, wu, ql_list_2, qlp_list_2, qmq_map, A_lq, eff_force_obj, H)
            sys.exit()
            # compute effective force (second order)
            # run over q pts list
            iql = 0
            for iq, il in tqdm(ql_list):
                # effective force
                Fjax_lqlqp = transf_2nd_order_force_phr(il, iq, wu, u, qpts, nat, Fax, Faxby, qlp_list, H)
                F_lqlqp = np.zeros(len(qlp_list), dtype=np.complex128)
                for jax in range(3*nat):
                    F_lqlqp[:] += Fjax_lqlqp[jax,:]
                sys.exit()
                # update acf / ph. res.
                self.compute_acf_Vph2(wq, wu, iq, il, qlp_list, A_lq[iql], A_lqp, F_lqlqp)
                # at res.
                if p.at_resolved or p.ph_resolved:
                    self.compute_acf_Vph2_atphr(nat, wq, wu, iq, il, qlp_list, A_lq[iql], A_lqp, Fjax_lqlqp)
                if log.level <= logging.INFO:
                    self.Delta_2 += self.compute_acf_order2_zero_time(wq, wu, iq, il, qlp_list, A_lq[iql], A_lqp, F_lqlqp)
                #print(self.Delta_2[0], self.acf_sp[0,0], iq, il, np.max(np.abs(F_lqlqp)), A_lq[iql], np.max(A_lqp))
                # iterate
                iql += 1
    #
    # compute acf parameters -> dyndec calculation
    def compute_acf_dyndec(self, wq, wu, u, qpts, nat, Fax, Faxby, ql_list):
        # compute ph. amplitude
        A_lq = compute_ph_amplitude_q(wu, nat, ql_list)
        # compute effective force (first order)
        Fjax_lq = transf_1st_order_force_phr(u, qpts, nat, Fax, ql_list)
        F_lq = np.zeros(len(ql_list), dtype=np.complex128)
        for jax in range(3*nat):
            F_lq[:] += Fjax_lq[jax,:]
        # set w_k coefficients
        npl = len(p.n_pulses)
        # compute <V(1)(t) V(1)(t')>
        self.compute_dkt0_acf_Vph1(wq, wu, ql_list, A_lq, F_lq)
        #self.compute_acf_Vph1(wq, wu, ql_list, A_lq, F_lq)
        plt.xlim([0.,1.5])
        #plt.plot(p.time[:], self.acf_sp[:,0].real)
        plt.plot(p.time[:], self.acfdd_sp[:,0,0].real)
        plt.show()
        #
        #plt.plot(p.time[:], self.acfdd_sp[:,0].real)
        #if mpi.rank == mpi.root:
        #    plt.show()
        #
        # if 2nd order
        if p.order_2_correct:
            nq = len(qpts)
            qmq_list = set_q_to_mq_list(qpts, nq)
            # set qlp list (only q>0)
            qlp_list = []
            for iqpair in qmq_list:
                iq1 = iqpair[0]
                for il in range(3*nat):
                    qlp_list.append((iq1,il))
            # complete amplitudes
            A_lqp = compute_ph_amplitude_q(wu, nat, qlp_list)
            # compute effective force (second order)
            # run over q pts list
            iql = 0
            for iq, il in tqdm(ql_list):
                # effective force
                Fjax_lqlqp = transf_2nd_order_force_phr(il, iq, u, qpts, nat, Faxby, qlp_list)
                F_lqlqp = np.zeros(len(qlp_list), dtype=np.complex128)
                for jax in range(3*nat):
                    F_lqlqp[:] += Fjax_lqlqp[jax,:]
                # update acf - dyndec
                #self.compute_dkt0_acf_Vph2(wq, wu, iq, il, qlp_list, A_lq[iql], A_lqp, F_lqlqp, w_k)
                plt.xlim([0.,.3])
                plt.plot(p.time[:], self.acfdd_sp[:,0].real)
                if mpi.rank == mpi.root:
                    plt.show()
                #sys.exit()
                # iterate
                iql += 1
    # collect data
    def collect_acf_from_processes(self, nat):
        # collect data from processes
        if p.time_resolved:
            self.acf = np.zeros((p.nt,2,p.ntmp), dtype=type(self.acf_sp[0,0,0]))
            for iT in range(p.ntmp):
                self.acf[:,0,iT] = mpi.collect_time_freq_array(self.acf_sp[:,0,iT])
                self.acf[:,1,iT] = mpi.collect_time_freq_array(self.acf_sp[:,1,iT])
        if p.w_resolved:
            self.acf = np.zeros((p.nwg,p.ntmp), dtype=type(self.acf_sp[0,0]))
            for iT in range(p.ntmp):
                self.acf[:,iT] = mpi.collect_time_freq_array(self.acf_sp[:,iT])
        # ph / at resolved
        if p.ph_resolved:
            if p.time_resolved:
                self.acf_phr = np.zeros((p.nt2,2,p.nphr,p.ntmp), dtype=type(self.acf_phr_sp[0,0,0,0]))
                self.acf_wql = np.zeros((p.nt2,2,p.nwbn,p.ntmp), dtype=type(self.acf_wql_sp[0,0,0,0]))
                for iT in range(p.ntmp):
                    for iph in range(p.nphr):
                        self.acf_phr[:,0,iph,iT] = mpi.collect_time_freq_array(self.acf_phr_sp[:,0,iph,iT])
                        self.acf_phr[:,1,iph,iT] = mpi.collect_time_freq_array(self.acf_phr_sp[:,1,iph,iT])
                    for iwb in range(p.nwbn):
                        if p.wql_freq[iwb] > 0:
                            self.acf_wql[:,0,iwb,iT] = mpi.collect_time_freq_array(self.acf_wql_sp[:,0,iwb,iT]) / p.wql_freq[iwb]
                            self.acf_wql[:,1,iwb,iT] = mpi.collect_time_freq_array(self.acf_wql_sp[:,1,iwb,iT]) / p.wql_freq[iwb]
            if p.w_resolved:
                self.acf_phr = np.zeros((p.nwg,p.nphr,p.ntmp), dtype=type(self.acf_phr_sp[0,0,0]))
                self.acf_wql = np.zeros((p.nwg,p.nwbn,p.ntmp), dtype=type(self.acf_wql_sp[0,0,0]))
                for iT in range(p.ntmp):
                    for iph in range(p.nphr):
                        self.acf_phr[:,iph,iT] = mpi.collect_time_freq_array(self.acf_phr_sp[:,iph,iT])
                    for iwb in range(p.nwbn):
                        if p.wql_freq[iwb] > 0:
                            self.acf_wql[:,iwb,iT] = mpi.collect_time_freq_array(self.acf_wql_sp[:,iwb,iT])
        #
        # atom resolved
        if p.at_resolved:
            if p.time_resolved:
                self.acf_atr = np.zeros((p.nt2,2,nat,p.ntmp), dtype=type(self.acf_atr_sp[0,0,0,0]))
                for iT in range(p.ntmp):
                    for ia in range(nat):
                        self.acf_atr[:,0,ia,iT] = mpi.collect_time_freq_array(self.acf_atr_sp[:,0,ia,iT])
                        self.acf_atr[:,1,ia,iT] = mpi.collect_time_freq_array(self.acf_atr_sp[:,1,ia,iT])
            if p.w_resolved:
                self.acf_atr = np.zeros((p.nwg,nat,p.ntmp), dtype=type(self.acf_atr_sp[0,0,0]))
                for iT in range(p.ntmp):
                    for ia in range(nat):
                        self.acf_atr[:,ia,iT] = mpi.collect_time_freq_array(self.acf_atr_sp[:,ia,iT])
    def collect_acfdd_from_processes(self):
        # n. pulses
        npl = len(p.n_pulses)
        self.acfdd = np.zeros((p.nt,npl,p.ntmp), dtype=type(self.acfdd_sp[0,0,0]))
        # run over T
        for iT in range(p.ntmp):
            for ipl in range(npl):
                self.acfdd[:,ipl,iT] = mpi.collect_time_array(self.acfdd_sp[:,ipl,iT])
    # extract dephasing parameters
    # from acf
    def extract_dephas_data(self, T2_obj, Delt_obj, tauc_obj, iT, lw_obj=None):
        # Delta^2 -> (eV^2)
        D2 = self.acf[0,iT].real
        Ct = np.zeros(p.nt)
        if np.abs(D2) == 0.:
            pass
        else:
            for t in range(p.nt):
                Ct[t] = self.acf[t,iT].real / D2
        # extract T2 time
        T2 = T2_eval()
        tau_c, T2_inv, ft = T2.extract_T2(p.time, Ct, D2)
        # store data in objects
        if tau_c is not None and T2_inv is not None:
            T2_obj.set_T2(iT, T2_inv)
            tauc_obj.set_tauc(iT, tau_c)
            Delt_obj.set_Delt(iT, D2)
            if lw_obj is not None:
                lw_obj.set_lw(iT, T2_inv)
        return ft
    # from dd acf
    def extract_dephas_data_dyndec(self, T2_obj, Delt_obj, tauc_obj, iT, ipl):
        # Delta^2 -> (eV^2)
        D2 = self.acfdd[0,ipl,iT].real
        Ct = np.zeros(p.nt)
        if np.abs(D2) == 0.:
            pass
        else:
            for t in range(p.nt):
                Ct[t] = self.acfdd[t,ipl,iT].real / D2
        # extract T2 time
        T2 = T2_eval()
        tau_c, T2_inv, ft = T2.extract_T2(p.time, Ct, D2)
        # store data in objects
        if tau_c is not None and T2_inv is not None:
            T2_obj.set_T2(ipl, iT, T2_inv)
            tauc_obj.set_tauc(ipl, iT, tau_c)
            Delt_obj.set_Delt(ipl, iT, D2)
        return ft
    #
    def extract_dephas_data_phr(self, T2_obj, Delt_obj, tauc_obj, iph, iT, lw_obj=None):
        # Delta^2
        D2 = self.acf_phr[0,iph,iT].real
        Ct = np.zeros(p.nt2)
        if np.abs(D2) == 0.:
            pass
        else:
            for t in range(p.nt2):
                Ct[t] = self.acf_phr[t,iph,iT].real / D2
        # extract T2 time
        T2 = T2_eval()
        tau_c, T2_inv, ft = T2.extract_T2(p.time2, Ct, D2)
        # store data in objects
        if tau_c is not None and T2_inv is not None:
            T2_obj.set_T2_phr(iph, iT, T2_inv)
            tauc_obj.set_tauc_phr(iph, iT, tau_c)
            Delt_obj.set_Delt_phr(iph, iT, D2)
            if lw_obj is not None:
                lw_obj.set_lw_phr(iph, iT, T2_inv)
        return ft
    #
    def extract_dephas_data_wql(self, T2_obj, Delt_obj, tauc_obj, iwb, iT, lw_obj=None):
        # Delta^2
        D2 = self.acf_wql[0,iwb,iT].real
        Ct = np.zeros(p.nt2)
        if np.abs(D2) == 0.:
            pass
        else:
            for t in range(p.nt2):
                Ct[t] = self.acf_wql[t,iwb,iT].real / D2
        # extract T2 time
        T2 = T2_eval()
        tau_c, T2_inv, ft = T2.extract_T2(p.time2, Ct, D2)
        # store data into objects
        if tau_c is not None and T2_inv is not None:
            T2_obj.set_T2_wql(iwb, iT, T2_inv)
            tauc_obj.set_tauc_wql(iwb, iT, tau_c)
            Delt_obj.set_Delt_wql(iwb, iT, D2)
            if lw_obj is not None:
                lw_obj.set_lw_wql(iwb, iT, T2_inv)
        return ft
    def extract_dephas_data_atr(self, T2_obj, Delt_obj, tauc_obj, ia, iT, lw_obj=None):
        # Delta^2
        D2 = self.acf_atr[0,ia,iT].real
        Ct = np.zeros(p.nt2)
        if np.abs(D2) == 0.:
            pass
        else:
            for t in range(p.nt2):
                Ct[t] = self.acf_atr[t,ia,iT].real / D2
        # extract T2 time
        T2 = T2_eval()
        tau_c, T2_inv, ft = T2.extract_T2(p.time2, Ct, D2)
        # store data
        if tau_c is not None and T2_inv is not None:
            T2_obj.set_T2_atr(ia, iT, T2_inv)
            tauc_obj.set_tauc_atr(ia, iT, tau_c)
            Delt_obj.set_Delt_atr(ia, iT, D2)
            if lw_obj is not None:
                lw_obj.set_lw_atr(ia, iT, T2_inv)
        return ft
    # print acf data
    def print_autocorrel_data(self, ft, ft_atr, ft_wql, ft_phr, iT):
        # Delta^2
        D2 = self.acf[0,iT].real
        Ct = np.zeros(p.nt)
        if np.abs(D2) == 0.:
            pass
        else:
            for t in range(p.nt):
                Ct[t] = self.acf[t,iT].real / D2
        # write data on file
        if log.level <= logging.INFO:
            namef = self.write_dir + "/acf-data-iT" + str(iT+1) + ".yml"
            print_acf_dict(p.time, Ct, ft, namef)
        # at. resolved
        if ft_atr is not None and p.at_resolved:
            nat = self.acf_atr.shape[1]
            Ct = np.zeros((p.nt2,nat))
            # run over ia
            for ia in range(nat):
                D2 = self.acf_atr[0,ia,iT].real
                if np.abs(D2) == 0.:
                    pass
                else:
                    for t in range(p.nt2):
                        Ct[t,ia] = self.acf_atr[t,ia,iT].real / D2
            # write data on file
            if log.level <= logging.INFO:
                namef = self.write_dir + "/acf-data-atr-iT" + str(iT+1) + ".yml"
                print_acf_dict(p.time2, Ct, ft_atr, namef)
        # ph. resolved
        if ft_phr is not None and p.ph_resolved:
            Ct = np.zeros((p.nt2,p.nphr))
            # run over iph
            for iph in range(p.nphr):
                D2 = self.acf_phr[0,iph,iT].real
                if np.abs(D2) == 0.:
                    pass
                else:
                    for t in range(p.nt2):
                        Ct[t,iph] = self.acf_phr[t,iph,iT].real / D2
            # write data on file
            if log.level <= logging.INFO:
                namef = self.write_dir + "/acf-data-phr-iT" + str(iT+1) + ".yml"
                print_acf_dict(p.time2, Ct, ft_phr, namef)
        # wql data
        if ft_wql is not None and p.ph_resolved:
            Ct = np.zeros((p.nt2,p.nwbn))
            # run over iph
            for iwb in range(p.nwbn):
                D2 = self.acf_wql[0,iwb,iT].real
                if np.abs(D2) == 0.:
                    pass
                else:
                    for t in range(p.nt2):
                        Ct[t,iwb] = self.acf_wql[t,iwb,iT].real / D2
            # write data on file
            if log.level <= logging.INFO:
                namef = self.write_dir + "/acf-data-wql-iT" + str(iT+1) + ".yml"
                print_acf_dict(p.time2, Ct, ft_wql, namef)
    #
    def print_autocorrel_data_spinconf(self, ft, ft_atr, ft_wql, ft_phr, ic, iT):
        # Delta^2
        D2 = self.acf[0,iT].real
        Ct = np.zeros(p.nt)
        if np.abs(D2) == 0.:
            pass
        else:
            for t in range(p.nt):
                Ct[t] = self.acf[t,iT].real / D2
        # write data on file
        if log.level <= logging.INFO:
            namef = self.write_dir + "/acf-data-ic" + str(ic) + "-iT" + str(iT+1) + ".yml"
            print_acf_dict(p.time, Ct, ft, namef)
        # at. resolved
        if ft_atr is not None and p.at_resolved:
            nat = self.acf_atr.shape[1]
            Ct = np.zeros((p.nt2,nat))
            # run over ia
            for ia in range(nat):
                D2 = self.acf_atr[0,ia,iT].real
                if np.abs(D2) == 0.:
                    pass
                else:
                    for t in range(p.nt2):
                        Ct[t,ia] = self.acf_atr[t,ia,iT].real / D2
            # write data on file
            if log.level <= logging.INFO:
                namef = self.write_dir + "/acf-data-atr-ic" + str(ic) + "-iT" + str(iT+1) + ".yml"
                print_acf_dict(p.time2, Ct, ft_atr, namef)
        # ph. resolved
        if ft_phr is not None and p.ph_resolved:
            Ct = np.zeros((p.nt2,p.nphr))
            # run over iph
            for iph in range(p.nphr):
                D2 = self.acf_phr[0,iph,iT].real
                if np.abs(D2) == 0.:
                    pass
                else:
                    for t in range(p.nt2):
                        Ct[t,iph] = self.acf_phr[t,iph,iT].real / D2
            # write data on file
            if log.level <= logging.INFO:
                namef = self.write_dir + "/acf-data-phr-ic" + str(ic) + "-iT" + str(iT+1) + ".yml"
                print_acf_dict(p.time2, Ct, ft_phr, namef)
        # wql data
        if ft_wql is not None and p.ph_resolved:
            Ct = np.zeros((p.nt2,p.nwbn))
            # run over iph
            for iwb in range(p.nwbn):
                D2 = self.acf_wql[0,iwb,iT].real
                if np.abs(D2) == 0.:
                    pass
                else:
                    for t in range(p.nt2):
                        Ct[t,iwb] = self.acf_wql[t,iwb,iT].real / D2
            # write data on file
            if log.level <= logging.INFO:
                namef = self.write_dir + "/acf-data-wql-ic" + str(ic) + "-iT" + str(iT+1) + ".yml"
                print_acf_dict(p.time2, Ct, ft_wql, namef)
    #
    # print function dynamical decoupling
    def print_autocorrel_data_dyndec(self, ft, ipl, iT):
        # Delta^2
        D2 = self.acfdd[0,ipl,iT].real
        # Ct
        Ct = np.zeros(p.nt)
        if np.abs(D2) == 0.:
            pass
        else:
            for t in range(p.nt):
                Ct[t] = self.acfdd[t,ipl,iT].real / D2
        # write data on file
        if log.level <= logging.INFO:
            namef = self.write_dir + "/acf-data-ip" + str(ipl+1) + "-iT" + str(iT+1) + ".yml"
            print_acf_dict(p.time, Ct, ft, namef)
    #
    # auto correlation tests
    def auto_correl_test(self):
        if p.time_resolved:
            if mpi.rank == mpi.root:
                log.info("Delta^2 TEST")
            self.Delta_2 = mpi.collect_array(self.Delta_2)
            for iT in range(p.ntmp):
                assert np.fabs(self.Delta_2[iT]/self.acf[0,0,iT].real - 1.0) < eps
            if mpi.rank == mpi.root:
                log.info("Delta^2 TEST PASSED")
        if p.w_resolved:
            if mpi.rank == mpi.root:
                log.info("Delta(w=0) TEST")
            self.Delta_w0 = mpi.collect_array(self.Delta_w0)
            for iT in range(p.ntmp):
                assert np.fabs(self.Delta_w0[iT]/self.acf[0,iT].real - 1.0) < eps
            if mpi.rank == mpi.root:
                log.info("Delta(w=0) TEST PASSED")