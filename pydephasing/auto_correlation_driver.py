#
# This module sets up the methods
# needed to compute the energy / spin fluctuations
# auto correlation function
#
import numpy as np
import logging
from common.phys_constants import eps
from pydephasing.log import log
from pydephasing.mpi import mpi
from pydephasing.set_param_object import p
from pydephasing.extract_ph_data import set_ql_list_red_qgrid, set_iqlp_list
from tqdm import tqdm
import matplotlib.pyplot as plt
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
        if mpi.rank == mpi.root:
            print(self.acf_sp[0,:])
        #
        # check Delta^2 value
        if log.level <= logging.INFO:
            if p.time_resolved:
                self.Delta_2 = np.zeros(p.ntmp)
                self.Delta_2 = self.compute_acf_V1_t0(wq, wu, ql_list, A_lq, F_lq)
            if p.w_resolved:
                self.Delta_w0= np.zeros(p.ntmp)
                self.Delta_w0= self.compute_acf_V1_w0(wq, wu, ql_list, A_lq, F_lq)
        if mpi.rank == mpi.root:
            print(self.Delta_w0)
        #plt.plot(p.w_grid[:], self.acf_sp[:,0].real)
        #plt.show()
    #
    # driver for acf - order 2 autocorrelation
    # see Eq. (34) and Eq. (73) in notes
    def compute_acf_2_driver(self, nat, wq, u, wu, ql_list, qlp_list_full, qmq_map, A_lq, eff_force_obj, H):
        if mpi.rank == mpi.root:
            log.info("\n")
            log.info("\t " + p.sep)
            log.info("\t PROGRESS (q,l) ITERATION ...")
        # run over external (q,l) pair list -> distributed over different
        # processors
        iql = 0
        for iq, il in tqdm(ql_list):
            # set the list of (iqp,ilp)
            qlp_list = set_iqlp_list(il, iq, qlp_list_full, wu, H)
            print(len(qlp_list), len(qlp_list_full))
            # update A_lqp with only needed amplitudes
            A_lqp = compute_ph_amplitude_q(wu, nat, qlp_list)
            # compute eff. force
            Fjax_lqlqp = eff_force_obj.transf_2nd_order_force_phr(il, iq, wu, u, nat, qlp_list)
            # num. (q',l')
            nlqp = len(qlp_list)
            F_lqlqp = np.zeros((4,nlqp), dtype=np.complex128)
            for jax in range(3*nat):
                F_lqlqp[:,:] += Fjax_lqlqp[:,jax,:]
                # [ps^-2]
            # ----------------------------------
            #    ACF calculation
            # ----------------------------------
            if p.time_resolved:
                self.compute_acf_V2_oft(wq, wu, iq, il, qlp_list, A_lq[iql], A_lqp, F_lqlqp)
            if p.w_resolved:
                self.compute_acf_V2_ofw(wq, wu, iq, il, qlp_list, A_lq[iql], A_lqp, F_lqlqp)
            # -----------------------------------
            #    AT / PHR calculation
            # -----------------------------------
            if p.at_resolved or p.ph_resolved:
                if p.time_resolved:
                    self.compute_acf_V2_atphr_oft(nat, wq, wu, iq, il, qlp_list, A_lq[iql], A_lqp, Fjax_lqlqp)
                if p.w_resolved:
                    self.compute_acf_V2_atphr_ofw(nat, wq, wu, iq, il, qlp_list, A_lq[iql], A_lqp, Fjax_lqlqp)
            #
            # check Delta^2 value
            if mpi.rank == mpi.root:
                print(self.acf_sp[0,:])
            if log.level <= logging.INFO:
                if p.time_resolved:
                    self.Delta_2 += self.compute_acf_V2_t0(wq, wu, iq, il, qlp_list, A_lq[iql], A_lqp, F_lqlqp)
                if p.w_resolved:
                    self.Delta_w0 += self.compute_acf_V2_w0(wq, wu, iq, il, qlp_list, A_lq[iql], A_lqp, F_lqlqp)
            if mpi.rank == mpi.root:
                print(self.Delta_2)
            # iterate (q,l)
            iql += 1
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
            eff_force_obj.set_up_2nd_order_force_phr(nat, qpts, Fax, Faxby, H)
            # driver of ACF - order 2 calculation
            self.compute_acf_2_driver(nat, wq, u, wu, ql_list_2, qlp_list_2, qmq_map, A_lq, eff_force_obj, H)
    #
    # collect data
    def collect_acf_from_processes(self, nat):
        # collect data from processes
        if p.time_resolved:
            if p.ACF_INTEG:
                self.acf = np.zeros((p.nt,2,p.ntmp), dtype=type(self.acf_sp[0,0,0]))
                for iT in range(p.ntmp):
                    self.acf[:,0,iT] = mpi.collect_time_freq_array(self.acf_sp[:,0,iT])
                    self.acf[:,1,iT] = mpi.collect_time_freq_array(self.acf_sp[:,1,iT])
            elif p.ACF_FIT:
                self.acf = np.zeros((p.nt,p.ntmp), dtype=type(self.acf_sp[0,0]))
                for iT in range(p.ntmp):
                    self.acf[:,iT] = mpi.collect_time_freq_array(self.acf_sp[:,iT])
        if p.w_resolved:
            self.acf = np.zeros((p.nwg,p.ntmp), dtype=type(self.acf_sp[0,0]))
            for iT in range(p.ntmp):
                self.acf[:,iT] = mpi.collect_time_freq_array(self.acf_sp[:,iT])
        # ph / at resolved
        if p.ph_resolved:
            if p.time_resolved:
                if p.ACF_INTEG:
                    self.acf_wql = np.zeros((p.nt2,2,p.nwbn,p.ntmp), dtype=type(self.acf_wql_sp[0,0,0,0]))
                    for iT in range(p.ntmp):
                        for iwb in range(p.nwbn):
                            if p.wql_freq[iwb] > 0:
                                self.acf_wql[:,0,iwb,iT] = mpi.collect_time_freq_array(self.acf_wql_sp[:,0,iwb,iT]) / p.wql_freq[iwb]
                                self.acf_wql[:,1,iwb,iT] = mpi.collect_time_freq_array(self.acf_wql_sp[:,1,iwb,iT]) / p.wql_freq[iwb]
                    if p.nphr > 0:
                        self.acf_phr = np.zeros((p.nt2,2,p.nphr,p.ntmp), dtype=type(self.acf_phr_sp[0,0,0,0]))
                        for iT in range(p.ntmp):
                            for iph in range(p.nphr):
                                self.acf_phr[:,0,iph,iT] = mpi.collect_time_freq_array(self.acf_phr_sp[:,0,iph,iT])
                                self.acf_phr[:,1,iph,iT] = mpi.collect_time_freq_array(self.acf_phr_sp[:,1,iph,iT])
                elif p.ACF_FIT:
                    self.acf_wql = np.zeros((p.nt2,p.nwbn,p.ntmp), dtype=type(self.acf_wql_sp[0,0,0]))
                    for iT in range(p.ntmp):
                        for iwb in range(p.nwbn):
                            if p.wql_freq[iwb] > 0:
                                self.acf_wql[:,iwb,iT] = mpi.collect_time_freq_array(self.acf_wql_sp[:,iwb,iT]) / p.wql_freq[iwb]
                    if p.nphr > 0:
                        self.acf_phr = np.zeros((p.nt2,p.nphr,p.ntmp), dtype=type(self.acf_phr_sp[0,0,0]))
                        for iT in range(p.ntmp):
                            for iph in range(p.nphr):
                                self.acf_phr[:,iph,iT] = mpi.collect_time_freq_array(self.acf_phr_sp[:,iph,iT])
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
                if p.ACF_INTEG:
                    self.acf_atr = np.zeros((p.nt2,2,nat,p.ntmp), dtype=type(self.acf_atr_sp[0,0,0,0]))
                    for iT in range(p.ntmp):
                        for ia in range(nat):
                            self.acf_atr[:,0,ia,iT] = mpi.collect_time_freq_array(self.acf_atr_sp[:,0,ia,iT])
                            self.acf_atr[:,1,ia,iT] = mpi.collect_time_freq_array(self.acf_atr_sp[:,1,ia,iT])
                elif p.ACF_FIT:
                    self.acf_atr = np.zeros((p.nt2,nat,p.ntmp), dtype=type(self.acf_atr_sp[0,0,0]))
                    for iT in range(p.ntmp):
                        for ia in range(nat):
                            self.acf_atr[:,ia,iT] = mpi.collect_time_freq_array(self.acf_atr_sp[:,ia,iT])
            if p.w_resolved:
                self.acf_atr = np.zeros((p.nwg,nat,p.ntmp), dtype=type(self.acf_atr_sp[0,0,0]))
                for iT in range(p.ntmp):
                    for ia in range(nat):
                        self.acf_atr[:,ia,iT] = mpi.collect_time_freq_array(self.acf_atr_sp[:,ia,iT])
    #
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
    #
    # auto correlation tests
    def auto_correl_test(self):
        if p.time_resolved:
            if mpi.rank == mpi.root:
                log.info("\n")
                log.info("\t " + p.sep)
                log.info("\t Delta^2 : TEST -> START")
            self.Delta_2 = mpi.collect_array(self.Delta_2)
            # ACF integ
            if p.ACF_INTEG:
                for iT in range(p.ntmp):
                    assert np.fabs(self.Delta_2[iT]/self.acf[0,0,iT].real - 1.0) < eps
            elif p.ACF_FIT:
                for iT in range(p.ntmp):
                    assert np.fabs(self.Delta_2[iT]/self.acf[0,iT].real - 1.0) < eps
            if mpi.rank == mpi.root:
                log.info("\t Delta^2 : TEST PASSED")
                log.info("\t " + p.sep)
                log.info("\n")
        if p.w_resolved:
            if mpi.rank == mpi.root:
                log.info("\n")
                log.info("\t " + p.sep)
                log.info("\t Delta(w=0) TEST -> START")
            self.Delta_w0 = mpi.collect_array(self.Delta_w0)
            for iT in range(p.ntmp):
                assert np.fabs(self.Delta_w0[iT]/self.acf[0,iT].real - 1.0) < eps
            if mpi.rank == mpi.root:
                log.info("\t Delta(w=0) : TEST PASSED")
                log.info("\t " + p.sep)
                log.info("\n")