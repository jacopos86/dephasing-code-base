import numpy as np
from itertools import product
from pydephasing.real_time_solver_base import RealTimeSolver
from pydephasing.log import log
from pydephasing.mpi import mpi
from pydephasing.set_param_object import p
from pydephasing.global_params import GPU_ACTIVE
from pathlib import Path
from common.phys_constants import THz_to_ev

if GPU_ACTIVE:
    from pydephasing.global_params import CUDA_SOURCE_DIR, gpu
    from pycuda.compiler import SourceModule
    import pycuda.driver as cuda
    from common.GPU_arrays_handler import GPU_ARRAY

#
#   set one-phonon processes solver
#

class OnephSolver(RealTimeSolver):
    def __init__(self, mode):
        self.LDBLD = None
        self.NMARK = None
        super(OnephSolver, self).__init__()
        if mode == 1:
            self.LDBLD = True
            self.NMARK = False
        elif mode == 2:
            self.LDBLD = False
            self.NMARK = True
        else:
            log.error("\t incorrect mode : mode = [1,2]")
    # evolve DM
    def evolve(self, dt, T, rho, H, Bfield, ph, sp_ph_inter, qgr):
        # set time arrays
        time = self.set_time_array(dt/2., T)
        # compute ext. magnetic field
        B = Bfield.set_pulse_oft(time)
        if mpi.rank == mpi.root:
            units = {'time': 'ps', 'quantity': 'T'}
            file_name = "B_oft.yml"
            file_name = "{}".format(p.write_dir + '/' + file_name)
            self.write_obj_to_file(time, B, units, file_name)
        mpi.comm.Barrier()
        # if LDBLD = True
        if self.LDBLD:
            if mpi.rank == mpi.root:
                log.info("\n")
                log.info("\t " + p.sep)
                log.info("\t COMPUTE SCATTERING MATRIX: \mathcal{P}")
                log.info("\t " + p.sep)
                log.info("\n")
            self.compute_scatter_matrix(H, ph, qgr)
    # scattering operator
    # calculation
    def compute_scatter_matrix(self, H, ph, qgr):
        n = len(H.basis_vectors)
        NST = np.int32(n)
        # n. modes
        NM = np.int32(ph.nmodes)
        # e-ph matrix
        P_eph = np.zeros((n, n, n, n), dtype=np.complex128)
        # GPU parallelization
        ilq_list = np.array(list(product(range(ph.nmodes), range(qgr.nq))))
        assert(len(ilq_list) == qgr.nq*ph.nmodes)
        INIT_INDEX, SIZE_LIST = gpu.distribute_data_on_grid(ilq_list)
        # eigenvalues array
        eig = np.zeros(n)
        for a in range(n):
            eig[a] = H.qs[a]['eig']
        EIG = GPU_ARRAY(eig, np.double)
        # phonon energies
        WQL = GPU_ARRAY(np.array(ph.uql) * THz_to_ev, np.double)
        
        # load file
        gpu_src = Path(CUDA_SOURCE_DIR+'compute_scatter_operator.cu').read_text()
        gpu_mod = SourceModule(gpu_src)
        compute_P_eph = gpu_mod.get_function("compute_P1_eph")
        compute_P_eph(NM, NST, cuda.Out(P_eph), block=gpu.block, grid=gpu.grid)