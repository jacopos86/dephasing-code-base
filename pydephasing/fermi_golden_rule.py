import numpy as np
import random
import pytest
from abc import ABC
from pathlib import Path
from pydephasing.mpi import mpi
from pydephasing.log import log
from pydephasing.global_params import GPU_ACTIVE, CUDA_SOURCE_DIR
from pydephasing.GPU_arrays_handler import GPU_ARRAY
from common.phys_constants import THz_to_ev, kb, hbar
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
            # w_max in THz -> convert to ev
            wgr = set_w_grid(p.w_max, p.nwg)
            self.wgr = wgr * THz_to_ev
        if self.REAL_TIME:
            if p.nt == None:
                self.tgr = set_time_grid_B(p.T, p.dt)
            elif p.dt == None:
                self.tgr = set_time_grid_A(p.T, p.nt)
    # compute linewidth
    # -> one phonon term
    def compute_relax_time_one_ph(self, H, inter_model, ph, qgr, T):
        # build (q,im) list
        ql_list = inter_model.ql_list
        gq = inter_model.g_ql
        print(ql_list.shape)
        print(gq.shape)
        print(T)
        exit()
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
        # weights
        wq = np.zeros(ql_list.shape[0])
        for iql in range(ql_list.shape[0]):
            iq, il = ql_list[iql]
            wq[iql] = qgr.wq[iq]
        # T1 times
        inv_T1 = self.compute_T1_oneph(ql_list, wq, gq, eig, wql, T, p.eta)
    # compute relax time
    # two phonons
    def compute_relax_time_two_ph(self, nat, H, inter_model, interact_dict, ph, qgr, T):
        n = len(H.basis_vectors)
        # read gqqp for (q,qp) pair
        qqp_list = qgr.build_irred_qqp_pairs()
        # parallelize over (q,q')
        qqp_list = mpi.split_list(qqp_list)
        Fax, Faxby = inter_model.compute_forces(nat, H, interact_dict)
        for iq, iqp in qqp_list:
            file_name = 'G-iq-' + str(iq) + '-iqp-' + str(iqp) + '.npz'
            file_path = p.write_dir + '/restart/' + file_name
            file_path = "{}".format(file_path)
            fil = Path(file_path)
            if not fil.exists():
                log.error("file : " + file_path + " NOT FOUND")
                exit(1)
            else:
                gqqp = inter_model.read_gqqp_from_file(file_path)
            assert(gqqp.shape[0] == gqqp.shape[1] == n)
            assert(gqqp.shape[2] == gqqp.shape[3] == ph.nmodes)
            for i in range(2):
                il1 = random.randint(0, ph.nmodes-1)
                il2 = random.randint(0, ph.nmodes-1)
                # compute single gqqp coeff.
                g12 = inter_model.compute_gqqp_l12(nat, iq, iqp, il1, il2, qgr, ph, H, Fax, Faxby)
                print(mpi.rank, i, iq, iqp, il1, il2)
                for k1 in range(n):
                    for k2 in range(n):
                        assert(g12[k1,k2] == pytest.approx(gqqp[k1,k2,il1,il2]))
                #print(iq, iqp, np.max(gqqp.real), np.max(gqqp.imag))
            exit()
        exit()
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
    def compute_T1_oneph(self, ql_list, wq, gq, eig, wql, temp, eta):
        if self.REAL_TIME:
            self.gt, self.int_gt = self.compute_T1_oneph_tres(ql_list, wq, gq, eig, wql, temp, eta)
        if self.FREQ_DOMAIN:
            self.gw = self.compute_T1_oneph_wres(ql_list, wq, gq, eig, wql, temp, eta)
    def compute_T1_oneph_tres(self, wq, ql_list, gq, eig, wql, temp, eta):
        # n. states
        n = len(eig)
        nt = len(self.tgr)
        # g(t)
        g_oft = np.zeros((n,nt))
        intg_oft = np.zeros((n,nt))
        for t in range(nt):
            print(t, self.tgr[t])
    def compute_T1_oneph_wres(self, wq, ql_list, gq, eig, wql, temp, eta):
        # n. states
        n = len(eig)
        nw = len(self.wgr)
        # g(w)
        g_ofw = np.zeros((n,nw))
        # iterate over w
        for iw in range(nw):
            print(iw, self.wgr[iw])





















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
    def compute_T1_oneph(self, ql_list, wq, gq, eig, wql, temp, eta):
        # load file
        gpu_src = Path(CUDA_SOURCE_DIR+'compute_oneph_decoher.cu').read_text()
        gpu_mod = SourceModule(gpu_src, options=["-I"+CUDA_SOURCE_DIR])
        # state energies
        EIG = GPU_ARRAY(eig, np.double)
        NST = EIG.length()
        # REAL TIME
        if self.REAL_TIME:
            TIME = GPU_ARRAY(self.tgr, np.double)
            NT = TIME.length()
            GOFT = GPU_ARRAY(np.zeros((NST,NT)), np.double)
            INTGOFT = GPU_ARRAY(np.zeros((NST,NT)), np.double)
            # load function
            compute_T1 = gpu_mod.get_function("compute_T1_oneph_time_resolved")
        if self.FREQ_DOMAIN:
            WGR = GPU_ARRAY(self.wgr, np.double)
            NW = WGR.length()
            GOFW = GPU_ARRAY(np.zeros((NST,NW)), np.double)
            # load function
            compute_T1 = gpu_mod.get_function("compute_T1_oneph_w_resolved")
        # weights
        WQ = GPU_ARRAY(wq, np.double)
        # ph. freq.
        WQL = GPU_ARRAY(wql, np.double)
        GQL = GPU_ARRAY(gq, np.complex128)
        NQL = WQL.length()
        print(mpi.rank, gq[1,1,1])
        # distribute data on grid
        INIT_INDEX, SIZE_LIST = gpu.distribute_data_on_grid(ql_list)
        if mpi.rank == mpi.root:
            log.info("\n")
            log.info("\t " + p.sep)
            INIT_INDEX.print_array()
            log.info("\t " + p.sep)
            log.info("\n")
        # call function at different temperatures
        for it in range(1):
            KT = np.double(kb*temp[it])
            if self.REAL_TIME:
                # \bar{t}
                TBAR = np.double(hbar / eta)
                if mpi.rank == mpi.root:
                    log.info("\t " + p.sep)
                    log.info("\t tbar coefficient : " + str(TBAR) + " ps")
                    log.info("\t " + p.sep)
                compute_T1(NST, NQL, NT, KT, TBAR, cuda.In(INIT_INDEX.to_gpu()), cuda.In(SIZE_LIST.to_gpu()), 
                        cuda.In(TIME.to_gpu()), cuda.In(WQ.to_gpu()), cuda.In(WQL.to_gpu()), cuda.In(EIG.to_gpu()), 
                        cuda.In(GQL.to_gpu()), cuda.Out(GOFT.to_gpu()), cuda.Out(INTGOFT.to_gpu()), 
                        block=gpu.block, grid=gpu.grid)
            if self.FREQ_DOMAIN:
                # linewidth
                ETA = np.double(eta)
                compute_T1(NST, NW, KT, ETA, cuda.In(INIT_INDEX.to_gpu()), cuda.In(SIZE_LIST.to_gpu()),
                        cuda.In(WGR.to_gpu()), cuda.In(WQ.to_gpu()), cuda.In(WQL.to_gpu()), cuda.In(EIG.to_gpu()),
                        block=gpu.block, grid=gpu.grid)