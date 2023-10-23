from pydephasing.input_parameters import p
from pydephasing.T2_calc import T2_eval
from pydephasing.log import log
import logging
from statsmodels import sm
from pydephasing.utility_functions import print_acf_dict
#
# auto-correlation HFI static
# class
#
class autocorrel_func_hfi_stat:
    # initialization
    def __init__(self, E_fluct):
        # output dir
        self.write_dir = p.write_dir
        # array variables
        self.nt = int(p.T_mus/p.dt_mus)
        self.nlags = p.nlags
        # time (mu sec)
        self.time = p.time2
        # arrays
        self.dE_oft = E_fluct.deltaE_oft
    # compute auto correlation
    # function
    def compute_acf(self):
        # acf
        Ct = sm.tsa.acf(self.dE_oft, nlags=self.nlags, fft=True)
        #
        # compute C(t) = acf(t) * <DeltaE^2>_T
        # eV^2
        #
        D2 = sum(self.dE_oft[:] * self.dE_oft[:]) / self.nt
        return D2, Ct
    # extract dephasing parameters
    # from acf
    def extract_dephas_data(self, D2, Ct, T2_obj, Delt_obj, tauc_obj, ic):
        nct = len(Ct)
        # extract T2 time
        T2 = T2_eval()
        tau_c, T2_inv, ft = T2.extract_T2_star(self.time[:nct], Ct, D2)
        # tau_c (mu sec)
        # T2_inv (ps^-1)
        if tau_c is not None and T2_inv is not None:
            T2_obj.set_T2_psec(ic, T2_inv)
            Delt_obj.set_Delt(ic, D2)
            tauc_obj.set_tauc(ic, tau_c)
        # write data on file
        if log.level <= logging.INFO:
            namef = self.write_dir + "/acf-data-ic" + str(ic+1) + ".yml"
            print_acf_dict(self.time, Ct, ft, namef)