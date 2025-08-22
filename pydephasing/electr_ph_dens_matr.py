from pydephasing.global_params import GPU_ACTIVE
from pydephasing.mpi import mpi
#
#  check GPU status
if GPU_ACTIVE:
    from pathlib import Path
    from pydephasing.global_params import gpu
    from pycuda.compiler import SourceModule
    import pycuda.driver as cuda

#
#  electron - phonon density matrix fluct.
#  delta rho_{a,b}^q
class elec_ph_dmatr(object):
    # initialization
    def __init__(self):
        # total number of DM modes
        self.nm = None
    def set_num_modes(self, n_modes):
        self.nm = n_modes
    #
    #  instance CPU / GPU
    #
    def generate_instance(self):
        if GPU_ACTIVE:
            return GPU_elec_ph_dmatr ()
        else:
            return CPU_elec_ph_dmatr ()
        
# ------------------------------------------------------
#    CPU class
# ------------------------------------------------------
class CPU_elec_ph_dmatr(elec_ph_dmatr):
    def __init__(self):
        super(CPU_elec_ph_dmatr, self).__init__()
    # set list of modes (local)
    def set_modes_list(self, qgr):
        qp_lst = qgr.set_q2mq_list()
        ql_list = []
        for iqp in range(len(qp_lst)):
            iq = qp_lst[iqp][0]
            for il in range(self.nm):
                ql_list.append([iq,il])
        self.qlst_loc = mpi.split_list(ql_list)
    # set ph. occup. for each mode
    def set_ph_occup(self, ph, T):
        self.ph_occup = []
        for iql in range(len(self.qlst_loc)):
            [iq, il] = self.qlst_loc[iql]
            self.ph_occup[iql] = ph.ph_occup(E, T)
    # compute d/dt delta rho_q
    def compute_time_deriv(self, rho):
        pass

# ------------------------------------------------------
#    GPU class
# ------------------------------------------------------
class GPU_elec_ph_dmatr(elec_ph_dmatr):
    def __init__(self):
        super(GPU_elec_ph_dmatr, self).__init__()