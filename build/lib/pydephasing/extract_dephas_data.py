from pydephasing.input_parameters import p
from pydephasing.log import log
#
#  This module extract the dephasing data
#  from acf array
class decoherence_data(object):
    def __init__(self):
        # T2 object declarations
        self.T2_obj = None
        # tau_c object
        self.tauc_obj = None
        # Delta object
        self.Delt_obj = None
        # lw object
        self.lw_obj = None
    # instantiate
    # time / freq. calculation
    def generate_instance(self):
        if p.time_resolved:
            return deco_data_time_res ()
        elif p.w_resolved:
            return deco_data_freq_res ()
        else:
            log.error("neither time nor freq. resolved...")
    # extract data from acf
    # driver function
    def extract_data_from_acf(self, acf):
        # run over T
        for iT in range(p.ntmp):
            # extract dephas. data
            self.extract_dephas_data(iT, acf)
            # at. resolved
            if p.at_resolved:
                self.extract_dephas_data_atr(iT, acf)
            # ph. resolved
            if p.ph_resolved and p.nphr > 0:
                self.extract_dephas_data_phr(iT, acf)
            if p.ph_resolved:
                self.extract_dephas_data_wql(iT, acf)
#
#  decoherence data class
#  -> time resolved calculation
class deco_data_time_res(decoherence_data):
    def __init__(self, nat):
        super(deco_data_time_res, self).__init__()
        self.T2_obj = T2i_ofT(nat)
        self.Delt_obj = Delta_ofT(nat)
        self.tauc_obj = tauc_ofT(nat)
        self.lw_obj = lw_ofT(nat)
#
#  decoherence data class
#  -> freq. resolved calculation
class deco_data_freq_res(decoherence_data):
    def __init__(self, nat):
        super(deco_data_freq_res, self).__init__()
        self.T2_obj = T2i_ofT(nat)
        self.lw_obj = lw_ofT(nat)