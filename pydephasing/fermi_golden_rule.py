import numpy as np
from abc import ABC
from pydephasing.mpi import mpi
from pydephasing.global_params import GPU_ACTIVE
from common.phys_constants import THz_to_ev
from common.GPU_arrays_handler import GPU_ARRAY
from pydephasing.grids import set_w_grid, set_time_grid_B, set_time_grid_A
from pydephasing.set_param_object import p

#
#   This module implements the
#   generalized Fermi golden rule
#   calculation
#

if GPU_ACTIVE:
    from pydephasing.global_params import gpu
    from pycuda.compiler import SourceModule
    import pycuda.driver as cuda

#
#   Abstract class : instantiate CPU / GPU version
#

class GeneralizedFermiGoldenRule(object):
    def __init__(self):
        pass
    #
    #   instance CPU / GPU
    #
    def generate_instance(self, REAL_TIME, FREQ_DOMAIN):
        if GPU_ACTIVE:
            return GeneralizedFermiGoldenRuleGPU (REAL_TIME, FREQ_DOMAIN)
        else:
            return GeneralizedFermiGoldenRuleCPU (REAL_TIME, FREQ_DOMAIN)
        
class GeneralizedFermiGoldenRuleBase(ABC):
    def __init__(self):
        # compute real time correlation function
        self.REAL_TIME = None
        self.tgr = None
        # compute freq. space
        self.FREQ_DOMAIN = None
        self.wgr = None
    def set_grids(self):
        if self.FREQ_DOMAIN:
            self.wgr = set_w_grid(p.w_max, p.nwg)
        if self.REAL_TIME:
            if p.nt == None:
                self.tgr = set_time_grid_B(p.T, p.dt)
            elif p.dt == None:
                self.tgr = set_time_grid_A(p.T, p.nt)
    # compute linewidth
    # -> one phonon term
    def compute_relax_time_one_ph(self, H, inter_model, ph, T):
        # build (q,im) list
        ql_list = inter_model.ql_list
        gq = inter_model.g_ql
        print('OK')
        print(ql_list.shape)
        print(gq.shape)
        print(T)
        # energy eigenvalues
        n = len(H.basis_vectors)
        eig = np.zeros(n)
        for a in range(n):
            eig[a] = H.qs[a]['eig']
        # ph. frequencies
        wql = np.zeros(ql_list.shape[0])
        for iql in range(ql_list.shape[0]):
            iq, il = ql_list[iql]
            wql[iql] = ph.uql[iq][il]*THz_to_ev
        # T1 times
        inv_T1 = self.compute_T1_oneph(ql_list, gq, eig, wql, T, p.eta)
    # compute T2 times
    def compute_decoher_times_one_ph(self, H, inter_model, ph):
        # quantum states
        n = len(H.basis_vectors)
        # T2 times
        inv_T2 = self.compute_T2_oneph(ql_list, gq, eig, ph)
        
# --------------------------------------------------------------------------
#
#    CPU Fermi Golden Rule implementation
#
# --------------------------------------------------------------------------

class GeneralizedFermiGoldenRuleCPU(GeneralizedFermiGoldenRuleBase):
    def __init__(self, REAL_TIME, FREQ_DOMAIN):
        super(GeneralizedFermiGoldenRuleCPU, self).__init__()
        self.REAL_TIME = REAL_TIME
        self.FREQ_DOMAIN = FREQ_DOMAIN
    def compute_T1_oneph(self, ql_list, gq, eig, wql, temp):
        if self.REAL_TIME:
            self.gt, self.int_gt = self.compute_T1_oneph_tres(ql_list, gq, eig, wql, temp)
        if self.FREQ_DOMAIN:
            self.gw = self.compute_T1_oneph_wres(ql_list, gq, eig, wql, temp)
    def compute_T1_oneph_tres(self, ql_list, gq, eig, wql, temp, tgr):
        pass
    def compute_T1_oneph_wres(ql_list, gq, eig, wql, temp, wgr):
        pass





















# ----------------------------------------------------------------------
#
#    GPU Fermi Golden Rule implementation
#
# ----------------------------------------------------------------------

class GeneralizedFermiGoldenRuleGPU(GeneralizedFermiGoldenRuleBase):
    def __init__(self, REAL_TIME, FREQ_DOMAIN):
        super(GeneralizedFermiGoldenRuleGPU, self).__init__()
        self.REAL_TIME = REAL_TIME
        self.FREQ_DOMAIN = FREQ_DOMAIN
    def compute_T1_oneph(self, ql_list, gq, eig, wql, temp, eta):
        # n. states
        n = len(eig)
        # REAL TIME
        if self.REAL_TIME:
            TIME = GPU_ARRAY(self.tgr, np.double)
            NT = TIME.length()
            GOFT = GPU_ARRAY(np.zeros((n,NT)), np.double)
            INTGOFT = GPU_ARRAY(np.zeros((n,NT)), np.double)
        if self.FREQ_DOMAIN:
            WGR = GPU_ARRAY(self.wgr, np.double)
            NW = WGR.length()
            GOFW = GPU_ARRAY(np.zeros((n,NW)), np.double)
        # state energies
        EIG = GPU_ARRAY(eig, np.double)
        NST = EIG.length()
        # linewidth
        ETA = np.double(eta)
        print(ETA)
        # ph. freq.
        WQL = GPU_ARRAY(wql, np.double)
        GQL = GPU_ARRAY(gq, np.complex128)
        # distribute data on grid
        INIT_INDEX, SIZE_LIST = gpu.distribute_data_on_grid(ql_list)