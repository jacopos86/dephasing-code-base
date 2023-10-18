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
            print(iT)
#
#  decoherence data class
#  -> time resolved calculation
class deco_data_time_res(decoherence_data):
    def __init__(self):
        super(deco_data_time, self).__init__()

#
#  decoherence data class
#  -> freq. resolved calculation
class deco_data_freq_res(decoherence_data):
    def __init__(self):
        super(deco_data_time, self).__init__()