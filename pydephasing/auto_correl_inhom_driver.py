from pydephasing.auto_correlation_spph_mod import GPU_acf_sp_ph, CPU_acf_sp_ph
from pydephasing.global_params import GPU_ACTIVE
from pydephasing.set_param_object import p
from pydephasing.restart import restart_calculation, save_data
from pydephasing.T2_calc_handler import set_T2_calc_handler
import os
import numpy as np
#
#  GPU calculation class -> inhomogeneous
#              calculation
class acf_sp_ph_inhom(GPU_acf_sp_ph if GPU_ACTIVE else CPU_acf_sp_ph):
    def __init__(self):
        super(acf_sp_ph_inhom, self).__init__()
        self.acf_avg = None
        self.acf_atr_avg = None
        self.acf_wql_avg = None
        self.acf_phr_avg = None
    # restart calculation
    def restart_calculation(self, nat, restart_file):
        # set calc. handler
        T2_calc_handler = set_T2_calc_handler()
        isExist = os.path.exists(restart_file)
        if not isExist:
            ic0 = 0
            if p.time_resolved:
                if p.ACF_INTEG:
                    self.acf_avg = np.zeros((p.nt,2,p.ntmp), dtype=np.complex128)
                elif p.ACF_FIT:
                    self.acf_avg = np.zeros((p.nt,p.ntmp), dtype=np.complex128)
            elif p.w_resolved:
                self.acf_avg = np.zeros((p.nwg,p.ntmp), dtype=np.complex128)
            # atom resolved initialization
            if p.at_resolved:
                if p.time_resolved:
                    if p.ACF_INTEG:
                        self.acf_atr_avg = np.zeros((p.nt2,2,nat,p.ntmp), dtype=np.complex128)
                    elif p.ACF_FIT:
                        self.acf_atr_avg = np.zeros((p.nt2,nat,p.ntmp), dtype=np.complex128)
                elif p.w_resolved:
                    self.acf_atr_avg = np.zeros((p.nwg,nat,p.ntmp), dtype=np.complex128)
            # ph. resolved initialization
            if p.ph_resolved:
                if p.time_resolved:
                    if p.ACF_INTEG:
                        self.acf_wql_avg = np.zeros((p.nt2,2,p.nwbn,p.ntmp), dtype=np.complex128)
                        if p.nphr > 0:
                            self.acf_phr_avg = np.zeros((p.nt2,2,p.nphr,p.ntmp), dtype=np.complex128)
                    elif p.ACF_FIT:
                        self.acf_wql_avg = np.zeros((p.nt2,p.nwbn,p.ntmp), dtype=np.complex128)
                        if p.nphr > 0:
                            self.acf_phr_avg = np.zeros((p.nt2,p.nphr,p.ntmp), dtype=np.complex128)
                elif p.w_resolved:
                    self.acf_wql_avg = np.zeros((p.nwg,p.nwbn,p.ntmp), dtype=np.complex128)
                    if p.nphr > 0:
                        self.acf_phr_avg = np.zeros((p.nwg,p.nphr,p.ntmp), dtype=np.complex128)
            # initialize T2_calc objects
            T2_calc_handler.set_up_param_objects_from_scratch(nat, p.nconf)
        else:
            restart_data = restart_calculation(restart_file)
            if p.time_resolved:
                [ic0, T2i_obj, Delt_obj, tauc_obj, lw_obj, acf_data] = restart_data
            elif p.w_resolved:
                [ic0, T2i_obj, lw_obj, acf_data] = restart_data
            # restart ACF arrays
            self.acf_avg = acf_data[0]
            # check atom resolved
            if p.at_resolved:
                self.acf_atr_avg = acf_data[1]
                if p.ph_resolved and len(acf_data) == 3:
                    self.acf_wql_avg = acf_data[2]
                elif p.ph_resolved and len(acf_data) == 4:
                    self.acf_wql_avg = acf_data[2]
                    self.acf_phr_avg = acf_data[3]
            else:
                if p.ph_resolved and len(acf_data) == 2:
                    self.acf_wql_avg = acf_data[1]
                elif p.ph_resolved and len(acf_data) == 3:
                    self.acf_wql_avg = acf_data[1]
                    self.acf_phr_avg = acf_data[2]
            # restart T2_calc
            if p.time_resolved:
                T2_calc_handler.set_up_param_objects(T2i_obj, Delt_obj, tauc_obj, lw_obj)
            elif p.w_resolved:
                T2_calc_handler.set_up_param_objects(T2i_obj, lw_obj)
        return ic0, T2_calc_handler
    #
    # save data
    def save_data(self, ic, T2_calc_handler):
        acf_data = [self.acf_avg]
        if p.at_resolved:
            acf_data.append(self.acf_atr_avg)
        if p.ph_resolved:
            acf_data.append(self.acf_wql_avg)
            if p.nphr > 0:
                acf_data.append(self.acf_phr_avg)
        save_data(ic, T2_calc_handler, acf_data)
    #
    # method : update acf_avg
    def update_avg_acf(self):
        self.acf_avg += self.acf
        # at. resolved
        if p.at_resolved:
            self.acf_atr_avg += self.acf_atr
        # ph. resolved
        if p.ph_resolved:
            self.acf_wql_avg += self.acf_wql
            if p.nphr > 0:
                self.acf_phr_avg += self.acf_phr
    def compute_avg_acf(self):
        self.acf_avg = self.acf_avg / p.nconf
        # at. resolved
        if p.at_resolved:
            self.acf_atr_avg = self.acf_atr_avg / p.nconf
        # ph. resolved
        if p.ph_resolved:
            self.acf_wql_avg = self.acf_wql_avg / p.nconf
            if p.nphr > 0:
                self.acf_phr_avg = self.acf_phr_avg / p.nconf