#
# This module sets up the method
# needed to compute T2 / T2*
# given the energy fluct. auto correlation function
#
import numpy as np
import scipy
from scipy import integrate
import yaml
import logging
from pydephasing.T2_classes import T2i_class, Delta_class, tauc_class, lw_class, \
    T2i_inhom_stat_dyndec, lw_inhom_stat_dyndec, T2i_inhom_stat, lw_inhom_stat
from pydephasing.phys_constants import hbar
from pydephasing.log import log
from pydephasing.input_parameters import p
from pydephasing.mpi import mpi
from pydephasing.extract_ph_data import extract_ph_data
from abc import ABC, abstractmethod
import warnings
warnings.filterwarnings("ignore")
#
# functions : 
# 1)  Exp 
# 2)  ExpSin 
# 3)  Explg
#
def Exp(x, c):
    return np.exp(-c * x)
def ExpSin(x, c, w, phi):
    return np.sin(w * x + phi) * np.exp(-c * x)
# fit gaussian+lorentzian decay
def Explg(x, a, b, c, sig):
    return a * np.exp(-c * x) + b * np.exp(-x**2 / 2 / sig**2)
#
#   class T2_eval_class -> freq. resolved
class T2_eval_class_freq_res:
    def __init__(self):
        self.T2_obj = None
        self.lw_obj = None
    def set_up_param_objects_from_scratch(self, nat, nconf=None):
        self.T2_obj = T2i_class().generate_instance(nat, nconf)
        self.lw_obj = lw_class().generate_instance(nat, nconf)
    def set_up_param_objects(self, T2_obj, lw_obj):
        self.T2_obj = T2_obj
        self.lw_obj = lw_obj
    # compute T2_inv
    def evaluate_T2(self, acf_w):
        acf_w0 = acf_w[0]
        # eV units
        T2_inv = acf_w0 / hbar
        # ps^-1
        return T2_inv
    # print ACF data
    def print_autocorrel_data(self, namef, wg, acf_w):
        # write data on file
        if log.level <= logging.INFO:
            # acf dictionary
            acf_dict = {'wg' : 0, 'acf' : 0}
            acf_dict['wg'] = wg
            acf_dict['acf'] = acf_w
            # save dict on file
            with open(namef, 'w') as out_file:
                yaml.dump(acf_dict, out_file)
    # print T2 times
    def print_decoherence_times(self):
        self.print_T2_times_data()
        # at. resolved
        if p.at_resolved:
            self.print_T2_atr_data()
        # ph. resolved
        if p.ph_resolved:
            self.print_T2_phr_data()
    def print_T2_times_data(self):
        T2_dict = {'T2_sec' : None, 'lw_eV' : None, 'T_K' : None}
        T2_dict['T2_sec'] = self.T2_obj.get_T2_sec()
        T2_dict['lw_eV']  = self.lw_obj.get_lw()
        T2_dict['T_K'] = p.temperatures
        # write yaml file
        namef = p.write_dir + "/T2-data.yml"
        with open(namef, 'w') as out_file:
            yaml.dump(T2_dict, out_file)
    def print_T2_atr_data(self):
        T2_dict = {'T2_sec' : None, 'lw_eV' : None, 'T_K' : None}
        T2_dict['T2_sec'] = self.T2_obj.get_T2_atr_sec()
        T2_dict['lw_eV']  = self.lw_obj.get_lw_atr()
        T2_dict['T_K'] = p.temperatures
        # write yaml file
        namef = p.write_dir + "/T2-atr-data.yml"
        with open(namef, 'w') as out_file:
            yaml.dump(T2_dict, out_file)
    def print_T2_phr_data(self):
        T2_dict = {'T2_sec' : None, 'lw_eV' : None, 'T_K' : None, 'wql' : None}
        T2_dict['T2_sec'] = self.T2_obj.get_T2_wql_sec()
        T2_dict['lw_eV']  = self.lw_obj.get_lw_wql()
        T2_dict['T_K'] = p.temperatures
        T2_dict['wql'] = p.wql_grid
        # write yaml file
        namef = p.write_dir + "/T2-wql-data.yml"
        with open(namef, 'w') as out_file:
            yaml.dump(T2_dict, out_file)
        # nphr > 0
        if p.nphr > 0:
            T2_dict = {'T2_sec' : None, 'lw_eV' : None, 'T_K' : None, 'wql' : None}
            # extract ph. mode energies
            u, wu, nq, qpts, wq, mesh = extract_ph_data()
            assert mesh[0]*mesh[1]*mesh[2] == nq
            assert len(qpts) == nq
            assert len(u) == nq
            wql = np.zeros(len(wu[0]))
            for iq in range(nq):
                wuq = wu[iq]
                wql[:] += wq[iq] * wuq[:]
            w_ph = np.zeros(p.nphr)
            for iph in range(p.nphr):
                ilq = p.phm_list[iph]
                w_ph= wql[ilq]
            T2_dict['wql'] = w_ph
            T2_dict['T2_sec'] = self.T2_obj.get_T2_phr_sec()
            T2_dict['lw_eV']  = self.lw_obj.get_lw_phr()
            T2_dict['T_K'] = p.temperatures
            # write yaml file
            namef = p.write_dir + "/T2-phr-data.yml"
            with open(namef, 'w') as out_file:
                yaml.dump(T2_dict, out_file)
# ----------------------------------------------------
# subclass of the frequency calculation model
# to be used for homogeneous calculations
# ----------------------------------------------------
class T2_eval_freq_homo_class(T2_eval_class_freq_res):
    def __init__(self):
        super(T2_eval_freq_homo_class, self).__init__()
    # extract phys. quant.
    def extract_physical_quantities(self, acf_obj, nat):
        for iT in range(p.ntmp):
            # compute Cw
            Cw = self.parameter_eval_driver(acf_obj, iT)
            # write data on file
            namef = p.write_dir + "/acf-data-iT" + str(iT) + ".yml"
            self.print_autocorrel_data(namef, p.w_grid, Cw)
            #
            # atom resolved
            if p.at_resolved:
                # local atom list
                atr_list = mpi.split_list(range(nat))
                Cw_atr = np.zeros((p.nwg,nat))
                # run over atoms
                for ia in atr_list:
                    # compute T2 times
                    Ca_w = self.atr_parameter_eval_driver(acf_obj, ia, iT)
                    if Ca_w is not None:
                        Cw_atr[:,ia] = Ca_w[:]
                Cw_atr = mpi.collect_array(Cw_atr)
                # collect to single proc.
                self.T2_obj.collect_atr_from_other_proc(iT)
                self.lw_obj.collect_atr_from_other_proc(iT)
                # write data on file
                namef = p.write_dir + "/acf-data-atr-iT" + str(iT) + ".yml"
                self.print_autocorrel_data(namef, p.w_grid, Cw_atr)
            #
            # ph. resolved
            if p.ph_resolved:
                # local wql grid list
                local_wql_lst = mpi.split_list(np.arange(0, p.nwbn, 1))
                Cw_wql = np.zeros((p.nwg,p.nwbn))
                # run over modes
                for iwb in local_wql_lst:
                    # compute T2 times
                    Cw_w = self.wql_parameter_eval_driver(acf_obj, iwb, iT)
                    if Cw_w is not None:
                        Cw_wql[:,iwb] = Cw_w[:]
                Cw_wql = mpi.collect_array(Cw_wql)
                # write data on file
                namef = p.write_dir + "/acf-data-wql-iT" + str(iT) + ".yml"
                self.print_autocorrel_data(namef, p.w_grid, Cw_wql)
                # check nphr > 0
                if p.nphr > 0:
                    # local list of modes
                    local_ph_lst = mpi.split_list(p.phm_list)
                    # run over modes
                    Cw_phr = np.zeros((p.nwg,p.nphr))
                    for im in local_ph_lst:
                        iph = p.phm_list.index(im)
                        # compute T2 times
                        Cp_w = self.phr_parameter_eval_driver(acf_obj, iph, iT)
                        if Cp_w is not None:
                            Cw_phr[:,iph] = Cp_w[:]
                    Cw_phr = mpi.collect_array(Cw_phr)
                    # write data on file
                    namef = p.write_dir + "/acf-data-phr-iT" + str(iT) + ".yml"
                    self.print_autocorrel_data(namef, p.w_grid, Cw_phr)
                # collect into single proc.
                self.T2_obj.collect_phr_from_other_proc(iT)
                self.lw_obj.collect_phr_from_other_proc(iT)
    # compute parameters
    def parameter_eval_driver(self, acf_obj, iT):
        acf_ofw = np.zeros(p.nwg)
        # store acf_w
        acf_ofw[:] = np.real(acf_obj.acf[:,iT])
        # compute T2_inv
        T2_inv = self.evaluate_T2(acf_ofw)
        self.T2_obj.set_T2_sec(iT, T2_inv)
        self.lw_obj.set_lw(iT, T2_inv)
        return acf_ofw
    def atr_parameter_eval_driver(self, acf_obj, ia, iT):
        acf_ofw = np.zeros(p.nwg)
        # store acf_w
        acf_ofw[:] = np.real(acf_obj.acf_atr[:,ia,iT])
        # compute T2_inv
        T2_inv = self.evaluate_T2(acf_ofw)
        self.T2_obj.set_T2_atr(ia, iT, T2_inv)
        self.lw_obj.set_lw_atr(ia, iT, T2_inv)
        return acf_ofw
    # ph. res. version
    def phr_parameter_eval_driver(self, acf_obj, iph, iT):
        acf_ofw = np.zeros(p.nwg)
        # store acf_ofw
        acf_ofw[:] = np.real(acf_obj.acf_phr[:,iph,iT])
        # compute T2_inv
        T2_inv = self.evaluate_T2(acf_ofw)
        # store data
        self.T2_obj.set_T2_phr(iph, iT, T2_inv)
        self.lw_obj.set_lw_phr(iph, iT, T2_inv)
        return acf_ofw
    def wql_parameter_eval_driver(self, acf_obj, iwql, iT):
        acf_ofw = np.zeros(p.nwg)
        # store acf_ofw
        acf_ofw[:] = np.real(acf_obj.acf_wql[:,iwql,iT])
        # compute T2_inv
        T2_inv = self.evaluate_T2(acf_ofw)
        # store data
        self.T2_obj.set_T2_wql(iwql, iT, T2_inv)
        self.lw_obj.set_lw_wql(iwql, iT, T2_inv)
        return acf_ofw
# ----------------------------------------------------
# subclass of the frequency calculation model
# to be used for inhomogeneous calculations
# ----------------------------------------------------
class T2_eval_freq_inhom_class(T2_eval_class_freq_res):
    def __init__(self):
        super(T2_eval_freq_inhom_class, self).__init__()
    # extract phys. quant.
    def extract_physical_quantities(self, acf_obj, ic, nat):
        # run over temperatures
        for iT in range(p.ntmp):
            #    Cw
            Cw = self.parameter_eval_driver(acf_obj, ic, iT)
            # write data on file
            namef = p.write_dir + "/acf-data-ic" + str(ic) + "-iT" + str(iT) + ".yml"
            self.print_autocorrel_data(namef, p.w_grid, Cw)
            #
            # at. resolved
            if p.at_resolved:
                # local atom list
                atr_list = mpi.split_list(range(nat))
                Cw_atr = np.zeros((p.nwg,nat))
                # run over atoms
                for ia in atr_list:
                    # compute T2 times
                    Ca_w = self.atr_parameter_eval_driver(acf_obj, ia, ic, iT)
                    if Ca_w is not None:
                        Cw_atr[:,ia] = Ca_w[:]
                Cw_atr = mpi.collect_array(Cw_atr)
                # collect into single proc.
                self.T2_obj.collect_atr_from_other_proc(ic, iT)
                self.lw_obj.collect_atr_from_other_proc(ic, iT)
                # write data on file
                namef = p.write_dir + "/acf-data-atr-ic" + str(ic) + "-iT" + str(iT) + ".yml"
                self.print_autocorrel_data(namef, p.w_grid, Cw_atr)
            #
            # ph. resolved
            if p.ph_resolved:
                # local ph. list
                local_wql_lst = mpi.split_list(np.arange(0, p.nwbn, 1))
                Cw_wql = np.zeros((p.nwg,p.nwbn))
                # run over modes
                for iwb in local_wql_lst:
                    # compute T2 times
                    Cw_w = self.wql_parameter_eval_driver(acf_obj, iwb, ic, iT)
                    if Cw_w is not None:
                        Cw_wql[:,iwb] = Cw_w[:]
                Cw_wql = mpi.collect_array(Cw_wql)
                # write data on file
                namef = p.write_dir + "/acf-data-wql-ic" + str(ic) + "-iT" + str(iT) + ".yml"
                self.print_autocorrel_data(namef, p.w_grid, Cw_wql)
                # check nphr > 0
                if p.nphr > 0:
                    # local list
                    local_ph_lst = mpi.split_list(p.phm_list)
                    # run over modes
                    Cw_phr = np.zeros((p.nwg,p.nphr))
                    for im in local_ph_lst:
                        iph = p.phm_list.index(im)
                        # compute T2 times
                        Cp_w = self.phr_parameter_eval_driver(acf_obj, iph, ic, iT)
                        if Cp_w is not None:
                            Cw_phr[:,iph] = Cp_w[:]
                    Cw_phr = mpi.collect_array(Cw_phr)
                    # write data on file
                    namef = p.write_dir + "/acf-data-phr-ic" + str(ic) + "-iT" + str(iT) + ".yml"
                    self.print_autocorrel_data(namef, p.w_grid, Cw_phr)
                # collect to single proc.
                self.T2_obj.collect_phr_from_other_proc(ic, iT)
                self.lw_obj.collect_phr_from_other_proc(ic, iT)
    # extract avg. phys. quant.
    def extract_avg_physical_quantities(self, acf_obj, nat):
        # run over temperatures
        for iT in range(p.ntmp):
            # compute Cw
            Cw = self.avg_parameter_eval_driver(acf_obj, iT)
            # write data on file
            namef = p.write_dir + "/acf-avg-data-iT" + str(iT) + ".yml"
            self.print_autocorrel_data(namef, p.w_grid, Cw)
            #
            # at. resolved
            if p.at_resolved:
                # local atom list
                atr_list = mpi.split_list(range(nat))
                Cw_atr = np.zeros((p.nwg,nat))
                # run over atoms
                for ia in atr_list:
                    # compute T2 times
                    Ca_w = self.avg_atr_parameter_eval_driver(acf_obj, ia, iT)
                    if Ca_w is not None:
                        Cw_atr[:,ia] = Ca_w[:]
                Cw_atr = mpi.collect_array(Cw_atr)
                # collect to single proc.
                self.T2_obj.collect_avg_atr_from_other_proc(iT)
                self.lw_obj.collect_avg_atr_from_other_proc(iT)
                # write data on file
                namef = p.write_dir + "/acf-avg-data-atr-iT" + str(iT) + ".yml"
                self.print_autocorrel_data(namef, p.w_grid, Cw_atr)
            #
            # ph. resolved
            if p.ph_resolved:
                # local ph. list
                local_wql_lst = mpi.split_list(np.arange(0, p.nwbn, 1))
                Cw_wql = np.zeros((p.nwg,p.nwbn))
                # run over modes
                for iwb in local_wql_lst:
                    # compute T2 times
                    Cw_w = self.avg_wql_parameter_eval_driver(acf_obj, iwb, iT)
                    if Cw_w is not None:
                        Cw_wql[:,iwb] = Cw_w[:]
                Cw_wql = mpi.collect_array(Cw_wql)
                # write data on file
                namef = p.write_dir + "/acf-avg-data-wql-iT" + str(iT) + ".yml"
                self.print_autocorrel_data(namef, p.w_grid, Cw_wql)
                # check nphr > 0
                if p.nphr > 0:
                    # local list
                    local_ph_lst = mpi.split_list(p.phm_list)
                    # run over modes
                    Cw_phr = np.zeros((p.nwg,p.nphr))
                    for im in local_ph_lst:
                        iph = p.phm_list.index(im)
                        # compute T2 times
                        Cp_w = self.avg_phr_parameter_eval_driver(acf_obj, iph, iT)
                        if Cp_w is not None:
                            Cw_phr[:,iph] = Cp_w[:]
                    Cw_phr = mpi.collect_array(Cw_phr)
                    # write data on file
                    namef = p.write_dir + "/acf-avg-data-phr-iT" + str(iT) + ".yml"
                    self.print_autocorrel_data(namef, p.w_grid, Cw_phr)
                # collect to single proc.
                self.T2_obj.collect_avg_phr_from_other_proc(iT)
                self.lw_obj.collect_avg_phr_from_other_proc(iT)
    # compute parameters
    def parameter_eval_driver(self, acf_obj, ic, iT):
        acf_ofw = np.zeros(p.nwg)
        # store acf_ofw data
        acf_ofw[:] = np.real(acf_obj.acf[:,iT])
        # compute T2_inv
        T2_inv = self.evaluate_T2(acf_ofw)
        # store data
        self.T2_obj.set_T2_sec(ic, iT, T2_inv)
        self.lw_obj.set_lw(ic, iT, T2_inv)
        return acf_ofw
    def avg_parameter_eval_driver(self, acf_obj, iT):
        acf_ofw = np.zeros(p.nwg)
        # store acf(w)
        acf_ofw[:] = np.real(acf_obj.acf_avg[:,iT])
        # compute T2_inv
        T2_inv = self.evaluate_T2(acf_ofw)
        # store data
        self.T2_obj.set_T2_avg(iT, T2_inv)
        self.lw_obj.set_lw_avg(iT, T2_inv)
        return acf_ofw
    # atom resolved version
    def atr_parameter_eval_driver(self, acf_obj, ia, ic, iT):
        acf_ofw = np.zeros(p.nwg)
        # store acf_w data
        acf_ofw[:] = np.real(acf_obj.acf_atr[:,ia,iT])
        # compute T2_inv
        T2_inv = self.evaluate_T2(acf_ofw)
        # store data
        self.T2_obj.set_T2_atr(ia, ic, iT, T2_inv)
        self.lw_obj.set_lw_atr(ia, ic, iT, T2_inv)
        return acf_ofw
    def avg_atr_parameter_eval_driver(self, acf_obj, ia, iT):
        acf_ofw = np.zeros(p.nwg)
        # store acf(w) data
        acf_ofw[:] = np.real(acf_obj.acf_atr_avg[:,ia,iT])
        # compute T2_inv
        T2_inv = self.evaluate_T2(acf_ofw)
        # store T2 data
        self.T2_obj.set_T2_atr_avg(ia, iT, T2_inv)
        self.lw_obj.set_lw_atr_avg(ia, iT, T2_inv)
        return acf_ofw
    # ph. resolved
    def phr_parameter_eval_driver(self, acf_obj, iph, ic, iT):
        acf_ofw = np.zeros(p.nwg)
        # store acf_w data
        acf_ofw[:] = np.real(acf_obj.acf_phr[:,iph,iT])
        # compute T2_inv
        T2_inv = self.evaluate_T2(acf_ofw)
        # store data
        self.T2_obj.set_T2_phr(iph, ic, iT, T2_inv)
        self.lw_obj.set_lw_phr(iph, ic, iT, T2_inv)
        return acf_ofw
    def avg_phr_parameter_eval_driver(self, acf_obj, iph, iT):
        acf_ofw = np.zeros(p.nwg)
        # store acf(w) data
        acf_ofw[:] = np.real(acf_obj.acf_phr_avg[:,iph,iT])
        # compute T2_inv
        T2_inv = self.evaluate_T2(acf_ofw)
        # store data
        self.T2_obj.set_T2_phr_avg(iph, iT, T2_inv)
        self.lw_obj.set_lw_phr_avg(iph, iT, T2_inv)
        return acf_ofw
    def wql_parameter_eval_driver(self, acf_obj, iwql, ic, iT):
        acf_ofw = np.zeros(p.nwg)
        # store acf_w data
        acf_ofw[:] = np.real(acf_obj.acf_wql[:,iwql,iT])
        # compute T2_inv
        T2_inv = self.evaluate_T2(acf_ofw)
        # store data
        self.T2_obj.set_T2_wql(iwql, ic, iT, T2_inv)
        self.lw_obj.set_lw_wql(iwql, ic, iT, T2_inv)
        return acf_ofw
    def avg_wql_parameter_eval_driver(self, acf_obj, iwql, iT):
        acf_ofw = np.zeros(p.nwg)
        # store acf(w)
        acf_ofw[:] = np.real(acf_obj.acf_wql_avg[:,iwql,iT])
        # compute T2_inv
        T2_inv = self.evaluate_T2(acf_ofw)
        # store data
        self.T2_obj.set_T2_wql_avg(iwql, iT, T2_inv)
        self.lw_obj.set_lw_wql_avg(iwql, iT, T2_inv)
        return acf_ofw
# ----------------------------------------------------
#   abstract T2_eval_class -> time resolved
# ----------------------------------------------------
class T2_eval_class_time_res(ABC):
    def __init__(self):
        self.T2_obj = None
        self.lw_obj = None
        self.tauc_obj = None
        self.Delt_obj = None
    def set_up_param_objects_from_scratch(self, nat, nconf=None):
        self.T2_obj   = T2i_class().generate_instance(nat, nconf)
        self.Delt_obj = Delta_class().generate_instance(nat, nconf)
        self.tauc_obj = tauc_class().generate_instance(nat, nconf)
        self.lw_obj   = lw_class().generate_instance(nat, nconf)
    def set_up_param_objects(self, T2_obj, Delt_obj, tauc_obj, lw_obj):
        self.T2_obj = T2_obj
        self.Delt_obj = Delt_obj
        self.tauc_obj = tauc_obj
        self.lw_obj = lw_obj
    def parametrize_acf(self, t, acf_oft):
        # units of tau_c depends on units of t
        # if it is called by static calculation -> mu sec
        # otherwise p sec
        D2 = acf_oft[0]
        if D2 == 0.:
            Ct = acf_oft
        else:
            Ct = acf_oft / D2
        # check non Nan
        if not np.isfinite(Ct).all():
            return None, None, None, None
        # set parametrization
        # e^-t/tau * sin(wt) parametrization
        # fit over exp. function
        if p.FIT_MODEL == "ExS":
            p0 = [1, 1, 1]    # start with values near those we expect
            res = scipy.optimize.curve_fit(ExpSin, t, Ct, p0, maxfev=self.maxiter)
            param = res[0]
            # fitting function
            ft = ExpSin(t, param[0], param[1], param[2])
        elif p.FIT_MODEL == "Ex":
            p0 = 1    # start with values near those we expect
            res = scipy.optimize.curve_fit(Exp, t, Ct, p0, maxfev=self.maxiter)
            param = res[0]
            # fitting function
            ft = Exp(t, param[0])
        # p = 1/tau_c (ps^-1/musec^-1)
        tau_c = 1./param[0]
        if D2 == 0.:
            tau_c = 0.
        return D2, tau_c, Ct, ft
    # print data
    def print_decoherence_times(self):
        self.print_T2_times_data()
        # at. resolved
        if p.at_resolved:
            self.print_T2_atr_data()
        # ph. resolved
        if p.ph_resolved:
            self.print_T2_phr_data()
    def print_T2_times_data(self):
        T2_dict = {'T2_sec' : None, 'tauc_ps' : None, 'Delt_eV' : None, 'lw_eV' : None, 'T_K' : None}
        T2_dict['T2_sec'] = self.T2_obj.get_T2_sec()
        T2_dict['tauc_ps']= self.tauc_obj.get_tauc()
        T2_dict['Delt_eV']= self.Delt_obj.get_Delt()
        T2_dict['lw_eV'] = self.lw_obj.get_lw()
        T2_dict['T_K'] = p.temperatures
        # write yaml file
        namef = p.write_dir + "/T2-data.yml"
        with open(namef, 'w') as out_file:
            yaml.dump(T2_dict, out_file)
    def print_T2_atr_data(self):
        T2_dict = {'T2_sec' : None, 'tauc_ps' : None, 'Delt_eV' : None, 'lw_eV' : None, 'T_K' : None}
        T2_dict['T2_sec'] = self.T2_obj.get_T2_atr_sec()
        T2_dict['tauc_ps']= self.tauc_obj.get_tauc_atr()
        T2_dict['Delt_eV']= self.Delt_obj.get_Delt_atr()
        T2_dict['lw_eV'] = self.lw_obj.get_lw_atr()
        T2_dict['T_K'] = p.temperatures
        # write yaml file
        namef = p.write_dir + "/T2-atr-data.yml"
        with open(namef, 'w') as out_file:
            yaml.dump(T2_dict, out_file)
    def print_T2_phr_data(self):
        T2_dict = {'T2_sec' : None, 'tauc_ps' : None, 'Delt_eV' : None, 'lw_eV' : None, 'T_K' : None, 'wql' : None}
        T2_dict['T2_sec'] = self.T2_obj.get_T2_wql_sec()
        T2_dict['tauc_ps']= self.tauc_obj.get_tauc_wql()
        T2_dict['Delt_eV']= self.Delt_obj.get_Delt_wql()
        T2_dict['lw_eV']  = self.lw_obj.get_lw_wql()
        T2_dict['T_K'] = p.temperatures
        T2_dict['wql'] = p.wql_grid
        # write yaml file
        namef = p.write_dir + "/T2-wql-data.yml"
        with open(namef, 'w') as out_file:
            yaml.dump(T2_dict, out_file)
        # nphr > 0
        if p.nphr > 0:
            T2_dict = {'T2_sec' : None, 'tauc_ps' : None, 'Delt_eV' : None, 'lw_eV' : None, 'T_K' : None, 'wql' : None}
            # extract ph. mode energies
            u, wu, nq, qpts, wq, mesh = extract_ph_data()
            assert mesh[0]*mesh[1]*mesh[2] == nq
            assert len(qpts) == nq
            assert len(u) == nq
            wql = np.zeros(len(wu[0]))
            for iq in range(nq):
                wuq = wu[iq]
                wql[:] += wq[iq] * wuq[:]
            w_ph = np.zeros(p.nphr)
            for iph in range(p.nphr):
                ilq = p.phm_list[iph]
                w_ph= wql[ilq]
            T2_dict['wql'] = w_ph
            T2_dict['T2_sec'] = self.T2_obj.get_T2_phr_sec()
            T2_dict['tauc_ps']= self.tauc_obj.get_tauc_phr()
            T2_dict['Delt_eV']= self.Delt_obj.get_Delt_phr()
            T2_dict['lw_eV'] = self.lw_obj.get_lw_phr()
            T2_dict['T_K'] = p.temperatures
            # write yaml data
            namef = p.write_dir + "/T2-phr-data.yml"
            with open(namef, 'w') as out_file:
                yaml.dump(T2_dict, out_file)
'''
# --------------------------------------------------------------
#  time resolved calculation -> concrete class implementation
#  fit the autocorrelation over 
#  (1) e^-t or sin(wt) (2) e^-t model
#  -> depending on the model -> different g(t)
#  depending on Delta tau_c value determine T2 / linwidth
# --------------------------------------------------------------
class T2_eval_fit_model_class(T2_eval_class_time_res):
    def __init__(self):
        super().__init__()
        # fft sample points
        self.N = p.N_df
        # sample spacing -> ps
        self.T = p.T_df
        # max iteration curve fitting
        self.maxiter = p.maxiter
    def get_T2_data(self):
        super().get_T2_data()
    def generate_instance(self):
        if not p.deph and not p.relax:
            return T2_eval_fit_model_stat_class()
        else:
            return T2_eval_fit_model_dyn_class()
    #
    # e^-g(t) -> g(t)=D2*tau_c^2[e^(-t/tau_c)+t/tau_c-1]
    # D2 -> eV^2
    # tau_c -> ps
    def exp_gt(self, x, D2, tau_c):
        r = np.exp(-self.gt(x, D2, tau_c))
        return r
    #
    def gt(self, x, D2, tau_c):
        r = np.zeros(len(x))
        r[:] = D2 / hbar ** 2 * tau_c ** 2 * (np.exp(-x[:]/tau_c) + x[:]/tau_c - 1)
        return r
    #
    # compute T2*
    # input : t, Ct, D2
    # output: tauc, T2_inv, [expsin, fit]
    def evaluate_T2(self, D2, tau_c):
        # check non Nan
        if not np.isfinite(Ct).all():
            return [None, None, None]
        # perform the fit
        p0 = [1., 1., 1.]
        res = scipy.optimize.curve_fit(ExpSin, t, Ct, p0, maxfev=self.maxiter)
        p = res[0]
        # p = 1/tau_c (mus^-1)
        tau_c = 1./p[2]
        # tau_c (mu sec)
        r = np.sqrt(D2) / hbar * tau_c * 1.E+6
        #
        # check limit r conditions
        if r < 1.E-4:
            T2_inv = D2 / hbar ** 2 * tau_c * 1.E+6
            # ps^-1
        elif r > 1.E+4:
            T2_inv = np.sqrt(D2) / hbar * 1.E+6
            # ps^-1
        else:
            # -> implement here
            tauc_ps = tau_c * 1.E+6
            T2_inv = self.T2inv_interp_eval(D2, tauc_ps)
            
        return tau_c, T2_inv, ExpSin(t, p[0], p[1], p[2])
'''
# -------------------------------------------------------------
# this class is unique for dynamical calculations
# relax / dephas calculations
# extract T2 from integrated auto correlation function directly
# -------------------------------------------------------------
class T2_eval_from_integ_class(T2_eval_class_time_res):
    def __init__(self):
        super().__init__()
        # max. number iter. curve fitting
        self.maxiter = p.maxiter
    def get_T2_data(self):
        super().get_T2_data()
    def evaluate_T2(self, acf_int_oft):
        # extract T2i as limt t->inf of acf_int_oft
        nt = acf_int_oft.shape[0]
        ft = acf_int_oft[int(7/8*nt):]
        T2i = sum(ft)/len(ft)
        T2i = T2i / hbar ** 2
        # eV^2 ps / eV^2 ps^2 = ps^-1
        return T2i
    # print ACF data
    #
    def print_autocorrel_data(self, namef, time, ft, Ct, integ_oft):
        # write data on file
        if log.level <= logging.INFO:
            # acf dictionary
            acf_dict = {'time' : 0, 'acf' : 0, 'ft' : 0, 'acf_integ' : 0}
            acf_dict['time'] = time
            acf_dict['acf'] = Ct
            acf_dict['ft'] = ft
            acf_dict['acf_integ'] = integ_oft
            # save dict on file
            with open(namef, 'w') as out_file:
                yaml.dump(acf_dict, out_file)
# -------------------------------------------------------------
# subclass of the integral model
# to be used for dynamical homogeneous calculations
# -------------------------------------------------------------
class T2_eval_from_integ_homo_class(T2_eval_from_integ_class):
    def __init__(self):
        super(T2_eval_from_integ_homo_class, self).__init__()
    def extract_physical_quantities(self, acf_obj, nat):
        # run over temperatures
        for iT in range(p.ntmp):
            # Ct, ft, acf_integ_oft
            Ct, ft, acf_integ_oft = self.parameter_eval_driver(acf_obj, iT)
            # write data on file
            namef = p.write_dir + "/acf-data-iT" + str(iT) + ".yml"
            self.print_autocorrel_data(namef, p.time, ft, Ct, acf_integ_oft)
            #
            # atom resolved
            if p.at_resolved:
                # local atom list
                atr_list = mpi.split_list(range(nat))
                ft_atr = np.zeros((p.nt2,nat))
                Ct_atr = np.zeros((p.nt2,nat))
                integ_atr = np.zeros((p.nt2,nat))
                # run over atoms
                for ia in atr_list:
                    # compute T2 times
                    Ca_t, fa_t, acf_integ_a_oft = self.atr_parameter_eval_driver(acf_obj, ia, iT)
                    if fa_t is not None:
                        ft_atr[:,ia] = fa_t[:]
                    if Ca_t is not None:
                        Ct_atr[:,ia] = Ca_t[:]
                    if acf_integ_a_oft is not None:
                        integ_atr[:,ia] = acf_integ_a_oft[:]
                ft_atr = mpi.collect_array(ft_atr)
                Ct_atr = mpi.collect_array(Ct_atr)
                integ_atr = mpi.collect_array(integ_atr)
                # collect into single processor
                self.T2_obj.collect_atr_from_other_proc(iT)
                self.Delt_obj.collect_atr_from_other_proc(iT)
                self.tauc_obj.collect_atr_from_other_proc(iT)
                self.lw_obj.collect_atr_from_other_proc(iT)
                # write data on file
                namef = p.write_dir + "/acf-data-atr-iT" + str(iT) + ".yml"
                self.print_autocorrel_data(namef, p.time2, ft_atr, Ct_atr, integ_atr)
            #
            # ph. resolved
            if p.ph_resolved:
                # local wql grid list
                local_wql_lst = mpi.split_list(np.arange(0, p.nwbn, 1))
                ft_wql = np.zeros((p.nt2,p.nwbn))
                Ct_wql = np.zeros((p.nt2,p.nwbn))
                integ_wql = np.zeros((p.nt2,p.nwbn))
                # run over modes
                for iwb in local_wql_lst:
                    # compute T2 times + print data
                    Cw_t, fw_t, acf_integ_w_oft = self.wql_parameter_eval_driver(acf_obj, iwb, iT)
                    if fw_t is not None:
                        ft_wql[:,iwb] = fw_t[:]
                    if Cw_t is not None:
                        Ct_wql[:,iwb] = Cw_t[:]
                    if acf_integ_w_oft is not None:
                        integ_wql[:,iwb] = acf_integ_w_oft[:]
                ft_wql = mpi.collect_array(ft_wql)
                Ct_wql = mpi.collect_array(Ct_wql)
                integ_wql = mpi.collect_array(integ_wql)
                # write data on file
                namef = p.write_dir + "/acf-data-wql-iT" + str(iT) + ".yml"
                self.print_autocorrel_data(namef, p.time2, ft_wql, Ct_wql, integ_wql)
                # check if nphr > 0
                if p.nphr > 0:
                    # local list of modes
                    local_ph_lst = mpi.split_list(p.phm_list)
                    # run over modes
                    ft_phr = np.zeros((p.nt2,p.nphr))
                    Ct_phr = np.zeros((p.nt2,p.nphr))
                    integ_phr = np.zeros((p.nt2,p.nphr))
                    for im in local_ph_lst:
                        iph = p.phm_list.index(im)
                        # compute T2 times + print data
                        Cp_t, fp_t, acf_integ_p_oft = self.phr_parameter_eval_driver(acf_obj, iph, iT)
                        if fp_t is not None:
                            ft_phr[:,iph] = fp_t[:]
                        if Cp_t is not None:
                            Ct_phr[:,iph] = Cp_t[:]
                        if acf_integ_p_oft is not None:
                            integ_phr[:,iph] = acf_integ_p_oft[:]
                    ft_phr = mpi.collect_array(ft_phr)
                    Ct_phr = mpi.collect_array(Ct_phr)
                    integ_phr = mpi.collect_array(integ_phr)
                    # write data on file
                    namef = p.write_dir + "/acf-data-phr-iT" + str(iT) + ".yml"
                    self.print_autocorrel_data(namef, p.time2, ft_phr, Ct_phr, integ_phr)
                # collect into single processor
                self.T2_obj.collect_phr_from_other_proc(iT)
                self.Delt_obj.collect_phr_from_other_proc(iT)
                self.tauc_obj.collect_phr_from_other_proc(iT)
                self.lw_obj.collect_phr_from_other_proc(iT)
    #
    # parameters evaluation
    def parameter_eval_driver(self, acf_obj, iT):
        acf_oft = np.zeros(p.nt)
        acf_integ_oft = np.zeros(p.nt)
        # storing acf_oft
        acf_oft[:] = np.real(acf_obj.acf[:,0,iT])
        # parametrize acf_oft
        D2, tauc_ps, Ct, ft = self.parametrize_acf(p.time, acf_oft)
        self.Delt_obj.set_Delt(iT, D2)
        self.tauc_obj.set_tauc(iT, tauc_ps)
        # compute T2_inv
        acf_integ_oft[:] = np.real(acf_obj.acf[:,1,iT])
        T2_inv = self.evaluate_T2(acf_integ_oft)
        self.T2_obj.set_T2_sec(iT, T2_inv)
        self.lw_obj.set_lw(iT, T2_inv)
        return Ct, ft, acf_integ_oft
    # atom resolved version
    def atr_parameter_eval_driver(self, acf_obj, ia, iT):
        acf_oft = np.zeros(p.nt2)
        acf_integ_oft = np.zeros(p.nt2)
        # storing acf_oft
        acf_oft[:] = np.real(acf_obj.acf_atr[:,0,ia,iT])
        # parametrize acf_oft
        D2, tauc_ps, Ct, ft = self.parametrize_acf(p.time2, acf_oft)
        self.Delt_obj.set_Delt_atr(ia, iT, D2)
        self.tauc_obj.set_tauc_atr(ia, iT, tauc_ps)
        # compute T2_inv
        acf_integ_oft[:] = np.real(acf_obj.acf_atr[:,1,ia,iT])
        T2_inv = self.evaluate_T2(acf_integ_oft)
        self.T2_obj.set_T2_atr(ia, iT, T2_inv)
        self.lw_obj.set_lw_atr(ia, iT, T2_inv)
        return Ct, ft, acf_integ_oft
    # ph. res. version
    def phr_parameter_eval_driver(self, acf_obj, iph, iT):
        acf_oft = np.zeros(p.nt2)
        acf_integ_oft = np.zeros(p.nt2)
        # storing acf_oft
        acf_oft[:] = np.real(acf_obj.acf_phr[:,0,iph,iT])
        # parametrize acf_oft
        D2, tauc_ps, Ct, ft = self.parametrize_acf(p.time2, acf_oft)
        self.Delt_obj.set_Delt_phr(iph, iT, D2)
        self.tauc_obj.set_tauc_phr(iph, iT, tauc_ps)
        # compute T2_inv
        acf_integ_oft[:] = np.real(acf_obj.acf_phr[:,1,iph,iT])
        T2_inv = self.evaluate_T2(acf_integ_oft)
        self.T2_obj.set_T2_phr(iph, iT, T2_inv)
        self.lw_obj.set_lw_phr(iph, iT, T2_inv)
        return Ct, ft, acf_integ_oft
    # wql resolved
    def wql_parameter_eval_driver(self, acf_obj, iwql, iT):
        acf_oft = np.zeros(p.nt2)
        acf_integ_oft = np.zeros(p.nt2)
        # storing acf_oft
        acf_oft[:] = np.real(acf_obj.acf_wql[:,0,iwql,iT])
        # parametrize acf_oft
        D2, tauc_ps, Ct, ft = self.parametrize_acf(p.time2, acf_oft)
        self.Delt_obj.set_Delt_wql(iwql, iT, D2)
        self.tauc_obj.set_tauc_wql(iwql, iT, tauc_ps)
        # compute T2_inv
        acf_integ_oft[:] = np.real(acf_obj.acf_wql[:,1,iwql,iT])
        T2_inv = self.evaluate_T2(acf_integ_oft)
        self.T2_obj.set_T2_wql(iwql, iT, T2_inv)
        self.lw_obj.set_lw_wql(iwql, iT, T2_inv)
        return Ct, ft, acf_integ_oft
# -------------------------------------------------------------
# subclass of the integral model
# to be used for dynamical inhomogeneous calculations
# -------------------------------------------------------------
class T2_eval_from_integ_inhom_class(T2_eval_from_integ_class):
    def __init__(self):
        super(T2_eval_from_integ_inhom_class, self).__init__()
    def extract_physical_quantities(self, acf_obj, ic, nat):
        # run over temperatures
        for iT in range(p.ntmp):
            # Ct, ft, acf_integ_oft
            Ct, ft, acf_integ_oft = self.parameter_eval_driver(acf_obj, ic, iT)
            # write data on file
            namef = p.write_dir + "/acf-data-ic" + str(ic) + "-iT" + str(iT) + ".yml"
            self.print_autocorrel_data(namef, p.time, ft, Ct, acf_integ_oft)
            #
            # atom resolved
            if p.at_resolved:
                # local atom list
                atr_list = mpi.split_list(range(nat))
                ft_atr = np.zeros((p.nt2,nat))
                Ct_atr = np.zeros((p.nt2,nat))
                integ_atr = np.zeros((p.nt2,nat))
                # run over atoms
                for ia in atr_list:
                    # compute T2 times
                    Ca_t, fa_t, acf_integ_a_oft = self.atr_parameter_eval_driver(acf_obj, ia, ic, iT)
                    if fa_t is not None:
                        ft_atr[:,ia] = fa_t[:]
                    if Ca_t is not None:
                        Ct_atr[:,ia] = Ca_t[:]
                    if acf_integ_a_oft is not None:
                        integ_atr[:,ia] = acf_integ_a_oft[:]
                ft_atr = mpi.collect_array(ft_atr)
                Ct_atr = mpi.collect_array(Ct_atr)
                integ_atr = mpi.collect_array(integ_atr)
                # collect into single processor
                self.T2_obj.collect_atr_from_other_proc(ic, iT)
                self.Delt_obj.collect_atr_from_other_proc(ic, iT)
                self.tauc_obj.collect_atr_from_other_proc(ic, iT)
                self.lw_obj.collect_atr_from_other_proc(ic, iT)
                # write data on file
                namef = p.write_dir + "/acf-data-atr-ic" + str(ic) + "-iT" + str(iT) + ".yml"
                self.print_autocorrel_data(namef, p.time2, ft_atr, Ct_atr, integ_atr)
            #
            # ph. resolved
            if p.ph_resolved:
                # local wql grid list
                local_wql_lst = mpi.split_list(np.arange(0, p.nwbn, 1))
                ft_wql = np.zeros((p.nt2,p.nwbn))
                Ct_wql = np.zeros((p.nt2,p.nwbn))
                integ_wql = np.zeros((p.nt2,p.nwbn))
                # run over modes
                for iwb in local_wql_lst:
                    # compute T2 times + print data
                    Cw_t, fw_t, acf_integ_w_oft = self.wql_parameter_eval_driver(acf_obj, iwb, ic, iT)
                    if fw_t is not None:
                        ft_wql[:,iwb] = fw_t[:]
                    if Cw_t is not None:
                        Ct_wql[:,iwb] = Cw_t[:]
                    if acf_integ_w_oft is not None:
                        integ_wql[:,iwb] = acf_integ_w_oft[:]
                ft_wql = mpi.collect_array(ft_wql)
                Ct_wql = mpi.collect_array(Ct_wql)
                integ_wql = mpi.collect_array(integ_wql)
                # write data on file
                namef = p.write_dir + "/acf-data-wql-ic" + str(ic) + "-iT" + str(iT) + ".yml"
                self.print_autocorrel_data(namef, p.time2, ft_wql, Ct_wql, integ_wql)
                # check if nphr > 0
                if p.nphr > 0:
                    # local list of modes
                    local_ph_lst = mpi.split_list(p.phm_list)
                    # run over modes
                    ft_phr = np.zeros((p.nt2,p.nphr))
                    Ct_phr = np.zeros((p.nt2,p.nphr))
                    integ_phr = np.zeros((p.nt2,p.nphr))
                    for im in local_ph_lst:
                        iph = p.phm_list.index(im)
                        # compute T2 times + print data
                        Cp_t, fp_t, acf_integ_p_oft = self.phr_parameter_eval_driver(acf_obj, iph, ic, iT)
                        if fp_t is not None:
                            ft_phr[:,iph] = fp_t[:]
                        if Cp_t is not None:
                            Ct_phr[:,iph] = Cp_t[:]
                        if acf_integ_p_oft is not None:
                            integ_phr[:,iph] = acf_integ_p_oft[:]
                    ft_phr = mpi.collect_array(ft_phr)
                    Ct_phr = mpi.collect_array(Ct_phr)
                    integ_phr = mpi.collect_array(integ_phr)
                    # write data on file
                    namef = p.write_dir + "/acf-data-phr-ic" + str(ic) + "-iT" + str(iT) + ".yml"
                    self.print_autocorrel_data(namef, p.time2, ft_phr, Ct_phr, integ_phr)
                # collect into single processor
                self.T2_obj.collect_phr_from_other_proc(ic, iT)
                self.Delt_obj.collect_phr_from_other_proc(ic, iT)
                self.tauc_obj.collect_phr_from_other_proc(ic, iT)
                self.lw_obj.collect_phr_from_other_proc(ic, iT)
    # extract avg. phys. quant.
    def extract_avg_physical_quantities(self, acf_obj, nat):
        # run over temperatures
        for iT in range(p.ntmp):
            # Ct, ft, acf_integ_oft
            Ct, ft, acf_integ_oft = self.avg_parameter_eval_driver(acf_obj, iT)
            # write data on file
            namef = p.write_dir + "/acf-avg-data-iT" + str(iT) + ".yml"
            self.print_autocorrel_data(namef, p.time, ft, Ct, acf_integ_oft)
            #
            # at. resolved
            if p.at_resolved:
                # local atom list
                atr_list = mpi.split_list(range(nat))
                ft_atr = np.zeros((p.nt2,nat))
                Ct_atr = np.zeros((p.nt2,nat))
                integ_atr = np.zeros((p.nt2,nat))
                # run over atoms
                for ia in atr_list:
                    # compute T2_times
                    Ca_t, fa_t, acf_integ_a_oft = self.avg_atr_parameter_eval_driver(acf_obj, ia, iT)
                    if fa_t is not None:
                        ft_atr[:,ia] = fa_t[:]
                    if Ca_t is not None:
                        Ct_atr[:,ia] = Ca_t[:]
                    if acf_integ_a_oft is not None:
                        integ_atr[:,ia] = acf_integ_a_oft[:]
                ft_atr = mpi.collect_array(ft_atr)
                Ct_atr = mpi.collect_array(Ct_atr)
                integ_atr = mpi.collect_array(integ_atr)
                # collect to single proc.
                self.T2_obj.collect_avg_atr_from_other_proc(iT)
                self.Delt_obj.collect_avg_atr_from_other_proc(iT)
                self.tauc_obj.collect_avg_atr_from_other_proc(iT)
                self.lw_obj.collect_avg_atr_from_other_proc(iT)
                # write data on file
                namef = p.write_dir + "/acf-avg-data-atr-iT" + str(iT) + ".yml"
                self.print_autocorrel_data(namef, p.time2, ft_atr, Ct_atr, integ_atr)
            #
            # ph. resolved
            if p.ph_resolved:
                # local wql grid list
                local_wql_lst = mpi.split_list(np.arange(0, p.nwbn, 1))
                ft_wql = np.zeros((p.nt2,p.nwbn))
                Ct_wql = np.zeros((p.nt2,p.nwbn))
                integ_wql = np.zeros((p.nt2,p.nwbn))
                # run over modes
                for iwb in local_wql_lst:
                    # compute T2 times
                    Cw_t, fw_t, acf_integ_w_oft = self.avg_wql_parameter_eval_driver(acf_obj, iwb, iT)
                    if fw_t is not None:
                        ft_wql[:,iwb] = fw_t[:]
                    if Cw_t is not None:
                        Ct_wql[:,iwb] = Cw_t[:]
                    if acf_integ_w_oft is not None:
                        integ_wql[:,iwb] = acf_integ_w_oft[:]
                ft_wql = mpi.collect_array(ft_wql)
                Ct_wql = mpi.collect_array(Ct_wql)
                integ_wql = mpi.collect_array(integ_wql)
                # write data on file
                namef = p.write_dir + "/acf-avg-data-wql-iT" + str(iT) + ".yml"
                self.print_autocorrel_data(namef, p.time2, ft_wql, Ct_wql, integ_wql)
                # check if nphr > 0
                if p.nphr > 0:
                    # local modes list
                    local_ph_lst = mpi.split_list(p.phm_list)
                    # run modes
                    ft_phr = np.zeros((p.nt2,p.nphr))
                    Ct_phr = np.zeros((p.nt2,p.nphr))
                    integ_phr = np.zeros((p.nt2,p.nphr))
                    for im in local_ph_lst:
                        iph = p.phm_list.index(im)
                        # compute T2 times
                        Cp_t, fp_t, acf_integ_p_oft = self.avg_phr_parameter_eval_driver(acf_obj, iph, iT)
                        if fp_t is not None:
                            ft_phr[:,iph] = fp_t[:]
                        if Cp_t is not None:
                            Ct_phr[:,iph] = Cp_t[:]
                        if acf_integ_p_oft is not None:
                            integ_phr[:,iph] = acf_integ_p_oft[:]
                    ft_phr = mpi.collect_array(ft_phr)
                    Ct_phr = mpi.collect_array(Ct_phr)
                    integ_phr = mpi.collect_array(integ_phr)
                    # write data on file
                    namef = p.write_dir + "/acf-avg-data-phr-iT" + str(iT) + ".yml"
                    self.print_autocorrel_data(namef, p.time2, ft_phr, Ct_phr, integ_phr)
                # collect data to single proc.
                self.T2_obj.collect_avg_phr_from_other_proc(iT)
                self.Delt_obj.collect_avg_phr_from_other_proc(iT)
                self.tauc_obj.collect_avg_phr_from_other_proc(iT)
                self.lw_obj.collect_avg_phr_from_other_proc(iT)
    #
    # parameters evaluation
    def parameter_eval_driver(self, acf_obj, ic, iT):
        acf_oft = np.zeros(p.nt)
        acf_integ_oft = np.zeros(p.nt)
        # storing acf_oft
        acf_oft[:] = np.real(acf_obj.acf[:,0,iT])
        # parametrize acf_oft
        D2, tauc_ps, Ct, ft = self.parametrize_acf(p.time, acf_oft)
        self.Delt_obj.set_Delt(ic, iT, D2)
        self.tauc_obj.set_tauc(ic, iT, tauc_ps)
        # compute T2_inv
        acf_integ_oft[:] = np.real(acf_obj.acf[:,1,iT])
        T2_inv = self.evaluate_T2(acf_integ_oft)
        self.T2_obj.set_T2_sec(ic, iT, T2_inv)
        self.lw_obj.set_lw(ic, iT, T2_inv)
        return Ct, ft, acf_integ_oft
    def avg_parameter_eval_driver(self, acf_obj, iT):
        acf_oft = np.zeros(p.nt)
        acf_integ_oft = np.zeros(p.nt)
        # store acf_oft
        acf_oft[:] = np.real(acf_obj.acf_avg[:,0,iT])
        # parametrize acf(t)
        D2, tauc_ps, Ct, ft = self.parametrize_acf(p.time, acf_oft)
        self.Delt_obj.set_Delt_avg(iT, D2)
        self.tauc_obj.set_tauc_avg(iT, tauc_ps)
        # compute T2_inv
        acf_integ_oft[:] = np.real(acf_obj.acf_avg[:,1,iT])
        T2_inv = self.evaluate_T2(acf_integ_oft)
        self.T2_obj.set_T2_avg(iT, T2_inv)
        self.lw_obj.set_lw_avg(iT, T2_inv)
        return Ct, ft, acf_integ_oft
    # atom resolved version
    def atr_parameter_eval_driver(self, acf_obj, ia, ic, iT):
        acf_oft = np.zeros(p.nt2)
        acf_integ_oft = np.zeros(p.nt2)
        # store acf_oft
        acf_oft[:] = np.real(acf_obj.acf_atr[:,0,ia,iT])
        # parametrize acf_oft
        D2, tauc_ps, Ct, ft = self.parametrize_acf(p.time2, acf_oft)
        self.Delt_obj.set_Delt_atr(ia, ic, iT, D2)
        self.tauc_obj.set_tauc_atr(ia, ic, iT, tauc_ps)
        # compute T2_inv
        acf_integ_oft[:] = np.real(acf_obj.acf_atr[:,1,ia,iT])
        T2_inv = self.evaluate_T2(acf_integ_oft)
        self.T2_obj.set_T2_atr(ia, ic, iT, T2_inv)
        self.lw_obj.set_lw_atr(ia, ic, iT, T2_inv)
        return Ct, ft, acf_integ_oft
    def avg_atr_parameter_eval_driver(self, acf_obj, ia, iT):
        acf_oft = np.zeros(p.nt2)
        acf_integ_oft = np.zeros(p.nt2)
        # store acf(t)
        acf_oft[:] = np.real(acf_obj.acf_atr_avg[:,0,ia,iT])
        # parametrize acf(t)
        D2, tauc_ps, Ct, ft = self.parametrize_acf(p.time2, acf_oft)
        self.Delt_obj.set_Delt_atr_avg(ia, iT, D2)
        self.tauc_obj.set_tauc_atr_avg(ia, iT, tauc_ps)
        # compute T2_inv
        acf_integ_oft[:] = np.real(acf_obj.acf_atr_avg[:,1,ia,iT])
        T2_inv = self.evaluate_T2(acf_integ_oft)
        self.T2_obj.set_T2_atr_avg(ia, iT, T2_inv)
        self.lw_obj.set_lw_atr_avg(ia, iT, T2_inv)
        return Ct, ft, acf_integ_oft
    # ph. resolved version
    def phr_parameter_eval_driver(self, acf_obj, iph, ic, iT):
        acf_oft = np.zeros(p.nt2)
        acf_integ_oft = np.zeros(p.nt2)
        # store acf_oft
        acf_oft[:] = np.real(acf_obj.acf_phr[:,0,iph,iT])
        # parametrize acf_oft
        D2, tauc_ps, Ct, ft = self.parametrize_acf(p.time2, acf_oft)
        self.Delt_obj.set_Delt_phr(iph, ic, iT, D2)
        self.tauc_obj.set_tauc_phr(iph, ic, iT, tauc_ps)
        # compute T2_inv
        acf_integ_oft[:] = np.real(acf_obj.acf_phr[:,1,iph,iT])
        T2_inv = self.evaluate_T2(acf_integ_oft)
        self.T2_obj.set_T2_phr(iph, ic, iT, T2_inv)
        self.lw_obj.set_lw_phr(iph, ic, iT, T2_inv)
        return Ct, ft, acf_integ_oft
    def avg_phr_parameter_eval_driver(self, acf_obj, iph, iT):
        acf_oft = np.zeros(p.nt2)
        acf_integ_oft = np.zeros(p.nt2)
        # store acf(t)
        acf_oft[:] = np.real(acf_obj.acf_phr_avg[:,0,iph,iT])
        # parametrize acf(t)
        D2, tauc_ps, Ct, ft = self.parametrize_acf(p.time2, acf_oft)
        self.Delt_obj.set_Delt_phr_avg(iph, iT, D2)
        self.tauc_obj.set_tauc_phr_avg(iph, iT, tauc_ps)
        # compute T2_inv
        acf_integ_oft[:] = np.real(acf_obj.acf_phr_avg[:,1,iph,iT])
        T2_inv = self.evaluate_T2(acf_integ_oft)
        self.T2_obj.set_T2_phr_avg(iph, iT, T2_inv)
        self.lw_obj.set_lw_phr_avg(iph, iT, T2_inv)
        return Ct, ft, acf_integ_oft
    # wql resolved version
    def wql_parameter_eval_driver(self, acf_obj, iwql, ic, iT):
        acf_oft = np.zeros(p.nt2)
        acf_integ_oft = np.zeros(p.nt2)
        # store acf_oft
        acf_oft[:] = np.real(acf_obj.acf_wql[:,0,iwql,iT])
        # parametrize acf_oft
        D2, tauc_ps, Ct, ft = self.parametrize_acf(p.time2, acf_oft)
        self.Delt_obj.set_Delt_wql(iwql, ic, iT, D2)
        self.tauc_obj.set_tauc_wql(iwql, ic, iT, tauc_ps)
        # compute T2_inv
        acf_integ_oft[:] = np.real(acf_obj.acf_wql[:,1,iwql,iT])
        T2_inv = self.evaluate_T2(acf_integ_oft)
        self.T2_obj.set_T2_wql(iwql, ic, iT, T2_inv)
        self.lw_obj.set_lw_wql(iwql, ic, iT, T2_inv)
        return Ct, ft, acf_integ_oft
    def avg_wql_parameter_eval_driver(self, acf_obj, iwql, iT):
        acf_oft = np.zeros(p.nt2)
        acf_integ_oft = np.zeros(p.nt2)
        # store acf(t)
        acf_oft[:] = np.real(acf_obj.acf_wql_avg[:,0,iwql,iT])
        # parametrize acf_oft
        D2, tauc_ps, Ct, ft = self.parametrize_acf(p.time2, acf_oft)
        self.Delt_obj.set_Delt_wql_avg(iwql, iT, D2)
        self.tauc_obj.set_tauc_wql_avg(iwql, iT, tauc_ps)
        # compute T2_inv
        acf_integ_oft[:] = np.real(acf_obj.acf_wql_avg[:,1,iwql,iT])
        T2_inv = self.evaluate_T2(acf_integ_oft)
        self.T2_obj.set_T2_wql_avg(iwql, iT, T2_inv)
        self.lw_obj.set_lw_wql_avg(iwql, iT, T2_inv)
        return Ct, ft, acf_integ_oft
# -------------------------------------------------------------
# subclass -> template for pure fitting calculation
# this is also abstract -> real class we must specifiy
# if static / dyn. homogeneous / dyn. inhomogeneous 
# -------------------------------------------------------------
class T2_eval_fit_model_class(T2_eval_class_time_res):
    def __init__(self):
        super().__init__()
        self.maxiter = p.maxiter
    @abstractmethod
    def evaluate_T2(self, args):
        '''method to implement'''
        return
    def T2inv_interp_eval(self, D2, tauc_ps):
        # fft sample points
        N = self.N
        T = self.T
        x = np.linspace(0.0, N*T, N, endpoint=False)
        x_ps = x * 1.E+6
        y = self.exp_gt(x_ps, D2, tauc_ps)
        try:
            c0 = D2 / hbar ** 2 * tauc_ps * 1.E+6   # mu s^-1
            s0 = hbar / np.sqrt(D2) * 1.E-6         # mu sec
            p0 = [0.5, 0.5, c0, s0]                 # start with values close to those expected
            res = scipy.optimize.curve_fit(Explg, x, y, p0, maxfev=self.maxiter)
            p1 = res[0]
            # gauss vs lorentzian
            if p1[0] > p1[1]:
                # T2 -> lorentzian (mu sec)
                T2_inv = p1[2]
                # ps^-1
                T2_inv = T2_inv * 1.E-6
            else:
                T2_inv = 1./p1[3]
                # mu s^-1 units
                T2_inv = T2_inv * 1.E-6
                # ps^-1 units
        except RuntimeError:
            T2_inv = None
        return T2_inv
    #
    # print ACF data
    def print_autocorrel_data(self, namef, time, ft, Ct):
        # write data on file
        if log.level <= logging.INFO:
            # acf dictionary
            acf_dict = {'time' : 0, 'acf' : 0, 'ft' : 0}
            acf_dict['time'] = time
            acf_dict['acf'] = Ct
            acf_dict['ft'] = ft
            # save dict on file
            with open(namef, 'w') as out_file:
                yaml.dump(acf_dict, out_file)
# -------------------------------------------------------------
# abstract subclass of the fitting model
# template for homo/inhomo dynamical calculations
# -------------------------------------------------------------
class T2_eval_fit_model_dyn_class(T2_eval_fit_model_class):
    def __init__(self):
        super(T2_eval_fit_model_dyn_class, self).__init__()
    def evaluate_T2(self, args):
        if p.FIT_MODEL == "Ex":
            [D2, tauc_ps] = args
            return self.evaluate_T2_exp_fit(D2, tauc_ps)
        elif p.FIT_MODEL == "ExS":
            [D2, t, ft] = args
            return self.evaluate_T2_expsin_fit(D2, t, ft)
    #
    #  T2 implementation    
    def evaluate_T2_exp_fit(self, D2, tauc_ps):
        # compute r = Delta tau_c
        # dynamical calculations -> r << 1
        r = np.sqrt(D2) / hbar * tauc_ps
        if r < 1.E-4:
            # lorentzian decay
            T2_inv = D2 / hbar ** 2 * tauc_ps
            # ps^-1
        else:
            log.info("\n")
            log.info("\t " + p.sep)
            log.warning("\t dynamical calc: r >~ 1 : " + str(r))
            T2_inv = None
        return T2_inv
    def evaluate_T2_expsin_fit(self, D2, t, ft):
        # T2_inv = D2 * int_0^T dt ft -> eV^2 ps
        # T2_inv /= hbar^2 -> ps^-1
        if ft is not None:
            T2_inv = integrate.simpson(ft, t)
            T2_inv = T2_inv * D2 / hbar ** 2
            # ps^-1
        else:
            T2_inv = None
        return T2_inv
# -------------------------------------------------------------
# subclass of the fitting model
# to be used for dynamical calculation -> different fitting
# time domain wrt to static calculations
# -------------------------------------------------------------
class T2_eval_fit_model_dyn_homo_class(T2_eval_fit_model_dyn_class):
    def __init__(self):
        super(T2_eval_fit_model_dyn_homo_class, self).__init__()
    def extract_physical_quantities(self, acf_obj, nat):
        # run over temperatures
        for iT in range(p.ntmp):
            # Ct, ft
            Ct, ft = self.parameter_eval_driver(acf_obj, iT)
            # write data on file
            namef = p.write_dir + "/acf-data-iT" + str(iT) + ".yml"
            self.print_autocorrel_data(namef, p.time, ft, Ct)
            #
            # atom resolved
            if p.at_resolved:
                # local atom list
                atr_list = mpi.split_list(range(nat))
                ft_atr = np.zeros((p.nt2,nat))
                Ct_atr = np.zeros((p.nt2,nat))
                # run over atoms
                for ia in atr_list:
                    # compute T2 times
                    Ca_t, fa_t = self.atr_parameter_eval_driver(acf_obj, ia, iT)
                    if fa_t is not None:
                        ft_atr[:,ia] = fa_t[:]
                    if Ca_t is not None:
                        Ct_atr[:,ia] = Ca_t[:]
                ft_atr = mpi.collect_array(ft_atr)
                Ct_atr = mpi.collect_array(Ct_atr)
                # collect into single proc.
                self.T2_obj.collect_atr_from_other_proc(iT)
                self.Delt_obj.collect_atr_from_other_proc(iT)
                self.tauc_obj.collect_atr_from_other_proc(iT)
                self.lw_obj.collect_atr_from_other_proc(iT)
                # write data on file
                namef = p.write_dir + "/acf-data-atr-iT" + str(iT) + ".yml"
                self.print_autocorrel_data(namef, p.time2, ft_atr, Ct_atr)
            #
            # ph. resolved
            if p.ph_resolved:
                # local wql list
                local_wql_lst = mpi.split_list(np.arange(0, p.nwbn, 1))
                ft_wql = np.zeros((p.nt2,p.nwbn))
                Ct_wql = np.zeros((p.nt2,p.nwbn))
                # run over modes
                for iwb in local_wql_lst:
                    # compute T2 times + print data
                    Cw_t, fw_t = self.wql_parameter_eval_driver(acf_obj, iwb, iT)
                    if fw_t is not None:
                        ft_wql[:,iwb] = fw_t[:]
                    if Cw_t is not None:
                        Ct_wql[:,iwb] = Cw_t[:]
                ft_wql = mpi.collect_array(ft_wql)
                Ct_wql = mpi.collect_array(Ct_wql)
                # write data on file
                namef = p.write_dir + "/acf-data-wql-iT" + str(iT) + ".yml"
                self.print_autocorrel_data(namef, p.time2, ft_wql, Ct_wql)
                # check if nphr > 0
                if p.nphr > 0:
                    # local list modes
                    local_ph_lst = mpi.split_list(p.phm_list)
                    # run over modes
                    ft_phr = np.zeros((p.nt2,p.nphr))
                    Ct_phr = np.zeros((p.nt2,p.nphr))
                    for im in local_ph_lst:
                        iph = p.phm_list.index(im)
                        # compute T2 times
                        Cp_t, fp_t = self.phr_parameter_eval_driver(acf_obj, iph, iT)
                        if fp_t is not None:
                            ft_phr[:,iph] = fp_t[:]
                        if Cp_t is not None:
                            Ct_phr[:,iph] = Cp_t[:]
                    ft_phr = mpi.collect_array(ft_phr)
                    Ct_phr = mpi.collect_array(Ct_phr)
                    # write data file
                    namef = p.write_dir + "/acf-data-phr-iT" + str(iT) + ".yml"
                    self.print_autocorrel_data(namef, p.time2, ft_phr, Ct_phr)
                # collect data
                self.T2_obj.collect_phr_from_other_proc(iT)
                self.Delt_obj.collect_phr_from_other_proc(iT)
                self.tauc_obj.collect_phr_from_other_proc(iT)
                self.lw_obj.collect_phr_from_other_proc(iT)
    #
    #  parameters calculation
    def parameter_eval_driver(self, acf_obj, iT):
        acf_oft = np.zeros(p.nt)
        # store acf_oft
        acf_oft[:] = np.real(acf_obj.acf[:,iT])
        # parametrize acf_oft
        D2, tauc_ps, Ct, ft = self.parametrize_acf(p.time, acf_oft)
        self.Delt_obj.set_Delt(iT, D2)
        self.tauc_obj.set_tauc(iT, tauc_ps)
        # compute T2_inv
        if p.FIT_MODEL == "Ex":
            T2_inv = self.evaluate_T2([D2, tauc_ps])
        elif p.FIT_MODEL == "ExS":
            T2_inv = self.evaluate_T2([D2, p.time, ft])
        self.T2_obj.set_T2_sec(iT, T2_inv)
        # lw obj
        self.lw_obj.set_lw(iT, T2_inv)
        return Ct, ft
    # atom resolved version
    def atr_parameter_eval_driver(self, acf_obj, ia, iT):
        acf_oft = np.zeros(p.nt2)
        # store acf_oft
        acf_oft[:] = np.real(acf_obj.acf_atr[:,ia,iT])
        # parametrize acf_oft
        D2, tauc_ps, Ct, ft = self.parametrize_acf(p.time2, acf_oft)
        self.Delt_obj.set_Delt_atr(ia, iT, D2)
        self.tauc_obj.set_tauc_atr(ia, iT, tauc_ps)
        # compute T2_inv
        if p.FIT_MODEL == "Ex":
            T2_inv = self.evaluate_T2([D2, tauc_ps])
        elif p.FIT_MODEL == "ExS":
            T2_inv = self.evaluate_T2([D2, p.time2, ft])
        self.T2_obj.set_T2_atr(ia, iT, T2_inv)
        # lw obj.
        self.lw_obj.set_lw_atr(ia, iT, T2_inv)
        return Ct, ft
    # ph. resolved version
    def phr_parameter_eval_driver(self, acf_obj, iph, iT):
        acf_oft = np.zeros(p.nt2)
        # store acf_oft
        acf_oft[:] = np.real(acf_obj.acf_phr[:,iph,iT])
        # parametrize acf
        D2, tauc_ps, Ct, ft = self.parametrize_acf(p.time2, acf_oft)
        self.Delt_obj.set_Delt_phr(iph, iT, D2)
        self.tauc_obj.set_tauc_phr(iph, iT, tauc_ps)
        # compute T2_inv
        if p.FIT_MODEL == "Ex":
            T2_inv = self.evaluate_T2([D2, tauc_ps])
        elif p.FIT_MODEL == "ExS":
            T2_inv = self.evaluate_T2([D2, p.time2, ft])
        self.T2_obj.set_T2_phr(iph, iT, T2_inv)
        # lw obj.
        self.lw_obj.set_lw_phr(iph, iT, T2_inv)
        return Ct, ft
    # wql resolved version
    def wql_parameter_eval_driver(self, acf_obj, iwql, iT):
        acf_oft = np.zeros(p.nt2)
        # store acf_oft
        acf_oft[:] = np.real(acf_obj.acf_wql[:,iwql,iT])
        # parametrize acf_oft
        D2, tauc_ps, Ct, ft = self.parametrize_acf(p.time2, acf_oft)
        self.Delt_obj.set_Delt_wql(iwql, iT, D2)
        self.tauc_obj.set_tauc_wql(iwql, iT, tauc_ps)
        # compute T2_inv
        if p.FIT_MODEL == "Ex":
            T2_inv = self.evaluate_T2([D2, tauc_ps])
        elif p.FIT_MODEL == "ExS":
            T2_inv = self.evaluate_T2([D2, p.time2, ft])
        self.T2_obj.set_T2_wql(iwql, iT, T2_inv)
        # store lw_obj
        self.lw_obj.set_lw_wql(iwql, iT, T2_inv)
        return Ct, ft
# -------------------------------------------------------------
# subclass of the fitting model
# to be used for dynamical inhomo calculation -> different fitting
# time domain wrt to static calculations
# -------------------------------------------------------------
class T2_eval_fit_model_dyn_inhom_class(T2_eval_fit_model_dyn_class):
    def __init__(self):
        super(T2_eval_fit_model_dyn_inhom_class, self).__init__()
    def extract_physical_quantities(self, acf_obj, ic, nat):
        # run over temperatures
        for iT in range(p.ntmp):
            # Ct, ft
            Ct, ft = self.parameter_eval_driver(acf_obj, ic, iT)
            # write data on file
            namef = p.write_dir + "/acf-data-ic" + str(ic) + "-iT" + str(iT) + ".yml"
            self.print_autocorrel_data(namef, p.time, ft, Ct)
            #
            # atom resolved
            if p.at_resolved:
                # local atom list
                atr_list = mpi.split_list(range(nat))
                ft_atr = np.zeros((p.nt2,nat))
                Ct_atr = np.zeros((p.nt2,nat))
                # run over atoms
                for ia in atr_list:
                    # compute T2 times
                    Ca_t, fa_t = self.atr_parameter_eval_driver(acf_obj, ia, ic, iT)
                    if fa_t is not None:
                        ft_atr[:,ia] = fa_t[:]
                    if Ca_t is not None:
                        Ct_atr[:,ia] = Ca_t[:]
                ft_atr = mpi.collect_array(ft_atr)
                Ct_atr = mpi.collect_array(Ct_atr)
                # into single proc.
                self.T2_obj.collect_atr_from_other_proc(ic, iT)
                self.Delt_obj.collect_atr_from_other_proc(ic, iT)
                self.tauc_obj.collect_atr_from_other_proc(ic, iT)
                self.lw_obj.collect_atr_from_other_proc(ic, iT)
                # write data on file
                namef = p.write_dir + "/acf-data-atr-ic" + str(ic) + "-iT" + str(iT) + ".yml"
                self.print_autocorrel_data(namef, p.time2, ft_atr, Ct_atr)
            #
            # ph. resolved
            if p.ph_resolved:
                # local wql list
                local_wql_lst = mpi.split_list(np.arange(0, p.nwbn, 1))
                ft_wql = np.zeros((p.nt2,p.nwbn))
                Ct_wql = np.zeros((p.nt2,p.nwbn))
                # run over modes
                for iwb in local_wql_lst:
                    # compute T2 times + print data
                    Cw_t, fw_t = self.wql_parameter_eval_driver(acf_obj, iwb, ic, iT)
                    if fw_t is not None:
                        ft_wql[:,iwb] = fw_t[:]
                    if Cw_t is not None:
                        Ct_wql[:,iwb] = Cw_t[:]
                ft_wql = mpi.collect_array(ft_wql)
                Ct_wql = mpi.collect_array(Ct_wql)
                # write data on file
                namef = p.write_dir + "/acf-data-wql-ic" + str(ic) + "-iT" + str(iT) + ".yml"
                self.print_autocorrel_data(namef, p.time2, ft_wql, Ct_wql)
                # check if nphr > 0
                if p.nphr > 0:
                    # local list of modes
                    local_ph_lst = mpi.split_list(p.phm_list)
                    # run over modes
                    ft_phr = np.zeros((p.nt2,p.nphr))
                    Ct_phr = np.zeros((p.nt2,p.nphr))
                    for im in local_ph_lst:
                        iph = p.phm_list.index(im)
                        # compute T2 times + print data
                        Cp_t, fp_t = self.phr_parameter_eval_driver(acf_obj, iph, ic, iT)
                        if fp_t is not None:
                            ft_phr[:,iph] = fp_t[:]
                        if Cp_t is not None:
                            Ct_phr[:,iph] = Cp_t[:]
                    ft_phr = mpi.collect_array(ft_phr)
                    Ct_phr = mpi.collect_array(Ct_phr)
                    # write data on file
                    namef = p.write_dir + "/acf-data-phr-ic" + str(ic) + "-iT" + str(iT) + ".yml"
                    self.print_autocorrel_data(namef, p.time2, ft_phr, Ct_phr)
                # collect to single proc.
                self.T2_obj.collect_phr_from_other_proc(ic, iT)
                self.Delt_obj.collect_phr_from_other_proc(ic, iT)
                self.tauc_obj.collect_phr_from_other_proc(ic, iT)
                self.lw_obj.collect_phr_from_other_proc(ic, iT)
    # extract avg. phys. quant.
    def extract_avg_physical_quantities(self, acf_obj, nat):
        # run temperatures
        for iT in range(p.ntmp):
            # Ct, ft
            Ct, ft = self.avg_parameter_eval_driver(acf_obj, iT)
            # write data on file
            namef = p.write_dir + "/acf-avg-data-iT" + str(iT) + ".yml"
            self.print_autocorrel_data(namef, p.time, ft, Ct)
            #
            # at. resolved
            if p.at_resolved:
                # local atom list
                atr_list = mpi.split_list(range(nat))
                ft_atr = np.zeros((p.nt2,nat))
                Ct_atr = np.zeros((p.nt2,nat))
                # run over atoms
                for ia in atr_list:
                    # compute T2 times
                    Ca_t, fa_t = self.avg_atr_parameter_eval_driver(acf_obj, ia, iT)
                    if fa_t is not None:
                        ft_atr[:,ia] = fa_t[:]
                    if Ca_t is not None:
                        Ct_atr[:,ia] = Ca_t[:]
                ft_atr = mpi.collect_array(ft_atr)
                Ct_atr = mpi.collect_array(Ct_atr)
                # collect to single proc.
                self.T2_obj.collect_avg_atr_from_other_proc(iT)
                self.Delt_obj.collect_avg_atr_from_other_proc(iT)
                self.tauc_obj.collect_avg_atr_from_other_proc(iT)
                self.lw_obj.collect_avg_atr_from_other_proc(iT)
                # write data on file
                namef = p.write_dir + "/acf-avg-data-atr-iT" + str(iT) + ".yml"
                self.print_autocorrel_data(namef, p.time2, ft_atr, Ct_atr)
            #
            # ph. resolved
            if p.ph_resolved:
                # local wql list
                local_wql_lst = mpi.split_list(np.arange(0, p.nwbn, 1))
                ft_wql = np.zeros((p.nt2,p.nwbn))
                Ct_wql = np.zeros((p.nt2,p.nwbn))
                # run over modes
                for iwb in local_wql_lst:
                    # compute T2 times
                    Cw_t, fw_t = self.avg_wql_parameter_eval_driver(acf_obj, iwb, iT)
                    if fw_t is not None:
                        ft_wql[:,iwb] = fw_t[:]
                    if Cw_t is not None:
                        Ct_wql[:,iwb] = Cw_t[:]
                ft_wql = mpi.collect_array(ft_wql)
                Ct_wql = mpi.collect_array(Ct_wql)
                # write data on file
                namef = p.write_dir + "/acf-avg-data-wql-iT" + str(iT) + ".yml"
                self.print_autocorrel_data(namef, p.time2, ft_wql, Ct_wql)
                # check if nphr > 0
                if p.nphr > 0:
                    # local list modes
                    local_ph_lst = mpi.split_list(p.phm_list)
                    # run over modes
                    ft_phr = np.zeros((p.nt2,p.nphr))
                    Ct_phr = np.zeros((p.nt2,p.nphr))
                    for im in local_ph_lst:
                        iph = p.phm_list.index(im)
                        # compute T2 times
                        Cp_t, fp_t = self.avg_phr_parameter_eval_driver(acf_obj, iph, iT)
                        if fp_t is not None:
                            ft_phr[:,iph] = fp_t[:]
                        if Cp_t is not None:
                            Ct_phr[:,iph] = Cp_t[:]
                    ft_phr = mpi.collect_array(ft_phr)
                    Ct_phr = mpi.collect_array(Ct_phr)
                    # write data on file
                    namef = p.write_dir + "/acf-avg-data-phr-iT" + str(iT) + ".yml"
                    self.print_autocorrel_data(namef, p.time2, ft_phr, Ct_phr)
                # collect data -> root
                self.T2_obj.collect_avg_phr_from_other_proc(iT)
                self.Delt_obj.collect_avg_phr_from_other_proc(iT)
                self.tauc_obj.collect_avg_phr_from_other_proc(iT)
                self.lw_obj.collect_avg_phr_from_other_proc(iT)
    #
    #  parameters evaluation
    def parameter_eval_driver(self, acf_obj, ic, iT):
        acf_oft = np.zeros(p.nt)
        # store acf_oft
        acf_oft[:] = np.real(acf_obj.acf[:,iT])
        # parametrize acf_oft
        D2, tauc_ps, Ct, ft = self.parametrize_acf(p.time, acf_oft)
        self.Delt_obj.set_Delt(ic, iT, D2)
        self.tauc_obj.set_tauc(ic, iT, tauc_ps)
        # compute T2_inv
        if p.FIT_MODEL == "Ex":
            T2_inv = self.evaluate_T2([D2, tauc_ps])
        elif p.FIT_MODEL == "ExS":
            T2_inv = self.evaluate_T2([D2, p.time, ft])
        # store results in objects
        self.T2_obj.set_T2_sec(ic, iT, T2_inv)
        # lw object
        self.lw_obj.set_lw(ic, iT, T2_inv)
        return Ct, ft
    def avg_parameter_eval_driver(self, acf_obj, iT):
        acf_oft = np.zeros(p.nt)
        # store acf(t)
        acf_oft[:] = np.real(acf_obj.acf_avg[:,iT])
        # parametrize acf(t)
        D2, tauc_ps, Ct, ft = self.parametrize_acf(p.time, acf_oft)
        self.Delt_obj.set_Delt_avg(iT, D2)
        self.tauc_obj.set_tauc_avg(iT, tauc_ps)
        # compute T2_inv
        if p.FIT_MODEL == "Ex":
            T2_inv = self.evaluate_T2([D2, tauc_ps])
        elif p.FIT_MODEL == "ExS":
            T2_inv = self.evaluate_T2([D2, p.time, ft])
        # store data in objects
        self.T2_obj.set_T2_avg(iT, T2_inv)
        self.lw_obj.set_lw_avg(iT, T2_inv)
        return Ct, ft
    # atom resolved version
    def atr_parameter_eval_driver(self, acf_obj, ia, ic, iT):
        acf_oft = np.zeros(p.nt2)
        # store acf_oft
        acf_oft[:] = np.real(acf_obj.acf_atr[:,ia,iT])
        # parametrize acf_oft
        D2, tauc_ps, Ct, ft = self.parametrize_acf(p.time2, acf_oft)
        self.Delt_obj.set_Delt_atr(ia, ic, iT, D2)
        self.tauc_obj.set_tauc_atr(ia, ic, iT, tauc_ps)
        # compute T2_inv
        if p.FIT_MODEL == "Ex":
            T2_inv = self.evaluate_T2([D2, tauc_ps])
        elif p.FIT_MODEL == "ExS":
            T2_inv = self.evaluate_T2([D2, p.time2, ft])
        # store results
        self.T2_obj.set_T2_atr(ia, ic, iT, T2_inv)
        # lw obj.
        self.lw_obj.set_lw_atr(ia, ic, iT, T2_inv)
        return Ct, ft
    def avg_atr_parameter_eval_driver(self, acf_obj, ia, iT):
        acf_oft = np.zeros(p.nt2)
        # store acf_oft
        acf_oft[:] = np.real(acf_obj.acf_atr_avg[:,ia,iT])
        # parametrize acf_oft
        D2, tauc_ps, Ct, ft = self.parametrize_acf(p.time2, acf_oft)
        self.Delt_obj.set_Delt_atr_avg(ia, iT, D2)
        self.tauc_obj.set_tauc_atr_avg(ia, iT, tauc_ps)
        # compute T2_inv
        if p.FIT_MODEL == "Ex":
            T2_inv = self.evaluate_T2([D2, tauc_ps])
        elif p.FIT_MODEL == "ExS":
            T2_inv = self.evaluate_T2([D2, p.time2, ft])
        # store results
        self.T2_obj.set_T2_atr_avg(ia, iT, T2_inv)
        self.lw_obj.set_lw_atr_avg(ia, iT, T2_inv)
    # ph. resolved version
    def phr_parameter_eval_driver(self, acf_obj, iph, ic, iT):
        acf_oft = np.zeros(p.nt2)
        # store acf_oft
        acf_oft[:] = np.real(acf_obj.acf_phr[:,iph,iT])
        # parametrize acf_oft
        D2, tauc_ps, Ct, ft = self.parametrize_acf(p.time2, acf_oft)
        self.Delt_obj.set_Delt_phr(iph, ic, iT, D2)
        self.tauc_obj.set_tauc_phr(iph, ic, iT, tauc_ps)
        # compute T2_inv
        if p.FIT_MODEL == "Ex":
            T2_inv = self.evaluate_T2([D2, tauc_ps])
        elif p.FIT_MODEL == "ExS":
            T2_inv = self.evaluate_T2([D2, p.time2, ft])
        # store data
        self.T2_obj.set_T2_phr(iph, ic, iT, T2_inv)
        self.lw_obj.set_lw_phr(iph, ic, iT, T2_inv)
        return Ct, ft
    def avg_phr_parameter_eval_driver(self, acf_obj, iph, iT):
        acf_oft = np.zeros(p.nt2)
        # store acf_oft
        acf_oft[:] = np.real(acf_obj.acf_phr_avg[:,iph,iT])
        # parametrize acf
        D2, tauc_ps, Ct, ft = self.parametrize_acf(p.time2, acf_oft)
        self.Delt_obj.set_Delt_phr_avg(iph, iT, D2)
        self.tauc_obj.set_tauc_phr_avg(iph, iT, tauc_ps)
        # compute T2_inv
        if p.FIT_MODEL == "Ex":
            T2_inv = self.evaluate_T2([D2, tauc_ps])
        elif p.FIT_MODEL == "ExS":
            T2_inv = self.evaluate_T2([D2, p.time2, ft])
        self.T2_obj.set_T2_phr_avg(iph, iT, T2_inv)
        self.lw_obj.set_lw_phr_avg(iph, iT, T2_inv)
        return Ct, ft
    # wql resolved version
    def wql_parameter_eval_driver(self, acf_obj, iwql, ic, iT):
        acf_oft = np.zeros(p.nt2)
        # store acf_oft
        acf_oft[:] = np.real(acf_obj.acf_wql[:,iwql,iT])
        # parametrize acf_oft
        D2, tauc_ps, Ct, ft = self.parametrize_acf(p.time2, acf_oft)
        self.Delt_obj.set_Delt_wql(iwql, ic, iT, D2)
        self.tauc_obj.set_tauc_wql(iwql, ic, iT, tauc_ps)
        # compute T2_inv
        if p.FIT_MODEL == "Ex":
            T2_inv = self.evaluate_T2([D2, tauc_ps])
        elif p.FIT_MODEL == "ExS":
            T2_inv = self.evaluate_T2([D2, p.time2, ft])
        # store data
        self.T2_obj.set_T2_wql(iwql, ic, iT, T2_inv)
        self.lw_obj.set_lw_wql(iwql, ic, iT, T2_inv)
        return Ct, ft
    def avg_wql_parameter_eval_driver(self, acf_obj, iwql, iT):
        acf_oft = np.zeros(p.nt2)
        # store data
        acf_oft[:] = np.real(acf_obj.acf_wql_avg[:,iwql,iT])
        # parametrize acf_oft
        D2, tauc_ps, Ct, ft = self.parametrize_acf(p.time2, acf_oft)
        self.Delt_obj.set_Delt_wql_avg(iwql, iT, D2)
        self.tauc_obj.set_tauc_wql_avg(iwql, iT, tauc_ps)
        # compute T2_inv
        if p.FIT_MODEL == "Ex":
            T2_inv = self.evaluate_T2([D2, tauc_ps])
        elif p.FIT_MODEL == "ExS":
            T2_inv = self.evaluate_T2([D2, p.time2, ft])
        self.T2_obj.set_T2_wql_avg(iwql, iT, T2_inv)
        self.lw_obj.set_lw_wql_avg(iwql, iT, T2_inv)
        return Ct, ft
# -------------------------------------------------------------
# subclass of the static model
# -------------------------------------------------------------
class T2_eval_static_class():
    def __init__(self):
        self.T2_obj = None
        self.lw_obj = None
    def set_up_param_objects_from_scratch(self, nconf):
        self.T2_obj = T2i_inhom_stat(nconf)
        self.lw_obj = lw_inhom_stat(nconf)
    def print_T2_times_data(self):
        T2_dict = {'T2_musec' : None, 'lw_eV' : None}
        T2_dict['T2_musec'] = self.T2_obj.get_T2_musec()
        T2_dict['lw_eV'] = self.lw_obj.get_lw()
        # write yaml file
        namef = p.write_dir + "/T2-data.yml"
        with open(namef, 'w') as out_file:
            yaml.dump(T2_dict, out_file)
# -------------------------------------------------------------
# subclass of the dynamical dec. model
# -------------------------------------------------------------
class T2_eval_dyndec_class():
    def __init__(self):
        self.T2_obj = None
        self.lw_obj = None
#
# generate initial parameters function
def generate_initial_params(r, D2, tau_c):
    #
    p0 = []
    if r < 1.:
        a = 1.
        b = 0.
    else:
        a = 0.
        b = 1.
    p0.append(a)
    p0.append(b)
    c0 = D2 / hbar**2 * tau_c    # ps^-1
    p0.append(c0)
    s0 = hbar / np.sqrt(D2)      # ps
    p0.append(s0)
    #
    return p0
#
# class T2_eval definition
#
class T2_eval:
    # initialization calculation
    # parameters
    def __init__(self):
        # fft sample points
        self.N = p.N_df
        # sample spacing -> ps
        self.T = p.T_df
        # max iteration curve fitting
        self.maxiter = p.maxiter
    # extract T2 fulle
    def extract_T2(self, t, Ct, D2):
        # check Ct finite
        if not np.isfinite(Ct).all():
            return [None, None, None]
        # compute exp
        tau_c1, T2_inv1, ft1 = self.extract_T2_Exp(t, Ct, D2)
        # compute exp sin
        tau_c2, T2_inv2, ft2 = self.extract_T2_ExpSin(t, Ct, D2)
        # final object
        tau_c = np.array([tau_c1, tau_c2], dtype=object)
        T2_inv = np.array([T2_inv1, T2_inv2], dtype=object)
        ft = [ft1, ft2]
        return tau_c, T2_inv, ft
    #
    # T2 calculation methods
    #
    # 1)  compute T2
    # input : t, Ct, D2
    # output: tauc, T2_inv, [exp. fit]
    def extract_T2_Exp(self, t, Ct, D2):
        # t : time array
        # Ct : acf
        # D2 : Delta^2
        #
        # fit over exp. function
        p0 = 1    # start with values near those we expect
        res = scipy.optimize.curve_fit(Exp, t, Ct, p0, maxfev=self.maxiter)
        p = res[0]
        # p = 1/tau_c (ps^-1)
        tau_c = 1./p[0]
        # ps units
        r = np.sqrt(D2) / hbar * tau_c
        # check r size
        # see Mukamel book
        if r < 1.E-4:
            T2_inv = D2 / hbar ** 2 * tau_c
            # ps^-1
        elif r > 1.E4:
            T2_inv = np.sqrt(D2) / hbar
            # ps^-1
        else:
            N = self.N
            T = self.T
            x = np.linspace(0.0, N*T, N, endpoint=False)
            y = exp_gt(x, D2, tau_c)
            try:
                init_param = generate_initial_params(r, D2, tau_c)
                res = scipy.optimize.curve_fit(Explg, x, y, init_param, maxfev=self.maxiter, xtol=1.E-3, ftol=1.E-3)
                p3 = res[0]
                # gauss vs lorentzian
                if p3[0] > p3[1]:
                    # T2 -> lorentzian (psec)
                    T2_inv = p3[2]
                else:
                    T2_inv = 1./p3[3]
                    # ps^-1 units
            except RuntimeError:
                # set T2_inv1 to None
                log.warning("T2_inv is None")
                T2_inv = None
        #
        return tau_c, T2_inv, Exp(t, p)
    #
    # function 2 -> compute T2
    #
    # input : t, Ct, D2
    # output: tauc, T2_inv, [expsin, fit]
    def extract_T2_ExpSin(self, t, Ct, D2):
        # t : time array
        # Ct : acf
        # D2 : Delta^2
        #
        # fit over exp. function
        p0 = [1., 1., 1.]      # start with values near those we expect
        res = scipy.optimize.curve_fit(ExpSin, t, Ct, p0, maxfev=self.maxiter)
        p = res[0]
        # p = 1/tau_c (ps^-1)
        tau_c = 1. / p[2]      # ps
        # r -> Mukamel book
        r = np.sqrt(D2) / hbar * tau_c
        #
        if r < 1.E-4:
            T2_inv = D2 / hbar ** 2 * tau_c
            # ps^-1
        elif r > 1.E4:
            T2_inv = np.sqrt(D2) / hbar
        else:
            N = self.N
            T = self.T
            x = np.linspace(0.0, N*T, N, endpoint=False)
            y = exp_gt(x, D2, tau_c)
            try:
                # perform fit
                init_param = generate_initial_params(r, D2, tau_c)
                res = scipy.optimize.curve_fit(Explg, x, y, init_param, maxfev=self.maxiter, xtol=1.E-3, ftol=1.E-3)
                # gauss vs lorentzian
                p3 = res[0]
                if p3[0] > p3[1]:
                    # T2 -> lorentzian (psec)
                    T2_inv = p3[2]
                else:
                    T2_inv = 1./p3[3]
                    # ps^-1 units
            except RuntimeError:
                # set T2_inv to None
                log.warning("\t T2_inv is None")
                T2_inv = None
        #
        return tau_c, T2_inv, ExpSin(t, p[0], p[1], p[2])