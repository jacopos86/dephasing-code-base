import numpy as np
from itertools import product
from pathlib import Path
from petsc4py import PETSc
from pydephasing.real_time.real_time_solver_base import RealTimeSolver, ElecPhDynamicSolverBase
from pydephasing.real_time.Liouville_solver import LiouvilleSolverElectronic
from pydephasing.utilities.log import log
from pydephasing.parallelization.mpi import mpi
from pydephasing.set_param_object import p
from pydephasing.global_params import GPU_ACTIVE
from pydephasing.common.phys_constants import THz_to_ev, hbar
from pydephasing.electr_ph_dens_matr import elec_ph_dmatr

if GPU_ACTIVE:
    from pydephasing.global_params import gpu
    from pycuda.compiler import SourceModule
    import pycuda.driver as cuda
    from pydephasing.parallelization.GPU_arrays_handler import GPU_ARRAY

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
        elif self.NMARK:
            if mpi.rank == mpi.root:
                log.info("\n")
                log.info("\t " + p.sep)
                log.info("\t E-PH DENSITY MATRIX")
                log.info("\t " + p.sep)
                log.info("\n")
            rho_q = elec_ph_dmatr().generate_instance()
            #rho_q.set_modes_list(qgr)
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
        exit()
        MODES_LIST = GPU_ARRAY(ilq_list, np.int32)
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
        compute_P_eph(NM, NST, cuda.In(INIT_INDEX.to_gpu()), cuda.In(SIZE_LIST.to_gpu()), cuda.In(MODES_LIST.to_gpu()),
                      cuda.Out(P_eph), block=gpu.block, grid=gpu.grid)

#  =================================================
#
#   ELECTRON-PHONON COUPLED SOLVER
#
#  =================================================

class ElecPhCouplSolver(ElecPhDynamicSolverBase):
    TITLE="EPH COUPLED DYNAMICS"
    #  initalization
    def __init__(self, evol_params):
        super().__init__(evol_params)
        # internal local objects
        self._He = None
        self._phi = None
        self._gql = None
        self._omega_q = None
    # --------------------------------------------------
    #   Monitor
    # --------------------------------------------------
    def monitor(self, ts, step, t, y):
        if step % self.save_every != 0:
            return
        # save index
        idx = step // self.save_every
        # -- unpack full state
        rho_flat, Bq = self._unpack_state(y)
        # -- unpack rho blocks
        offset = 0
        for ik in range(self._rho_e.nkpt):
            for isp in range(self._rho_e.nspin):
                # rho(k)
                rho_k = rho_flat[offset: offset+self._nbnd ** 2].reshape(self._nbnd, self._nbnd, order="F")
                self._rho_e.store_rho_time(ik, isp, idx, rho_k)
                # optional diagnostics
                tr = np.trace(rho_k).real
                self._rho_e.store_traces(ik, isp, idx, tr)
                offset += self._nbnd ** 2
        # -- store Ehrenfest field
        self._phi.store_Bq_time(idx, Bq.copy())
    # --------------------------------------------------
    #   RENORMALIZE g_ql -> Tr(rho_eq g_ql) = 0
    # --------------------------------------------------
    def _renorm_gql(self, y):
        g_avg = np.zeros(self._gql.shape[2], dtype=np.complex128)
        # -- unpack state
        rho_flat = self._unpack_rho_state(y)
        wk = np.ones(self._rho_e.nkpt) / self._rho_e.nkpt
        for iql in range(self._gql.shape[2]):
            gqp = self._gql[:,:,iql].conj()
            # iterate over k
            offset = 0
            for ik in range(self._rho_e.nkpt):
                for isp in range(self._rho_e.nspin):
                    rho_k = rho_flat[offset: offset+self._nbnd ** 2].reshape(self._nbnd, self._nbnd, order="F")
                    g_avg[iql] += wk[ik] * np.einsum(
                        'ab,ab->',
                        gqp,
                        rho_k,
                        optimize=True
                    )
                    offset += self._nbnd ** 2
            # normalize gql
            self._gql[:,:,iql] -= g_avg[iql]
    # --------------------------------------------------
    #   UNPACK STATE
    # --------------------------------------------------
    def _unpack_state(self, y):
        y_np = y.getArray(readonly=True)
        # -- unpack
        dim_rho = self._rho_e.nkpt * self._rho_e.nspin * self._nbnd ** 2
        rho_flat = y_np[:dim_rho]
        # Ehr field
        B_flat = y_np[dim_rho:]
        B_shape = (self._phi.nqpt, self._phi.nmd)
        Bq = B_flat.reshape(B_shape, order="C")
        return rho_flat, Bq
    def _unpack_rho_state(self, y):
        y_np = y.getArray(readonly=True)
        # -- unpack
        dim_rho = self._rho_e.nkpt * self._rho_e.nspin * self._nbnd ** 2
        rho_flat = y_np[:dim_rho]
        return rho_flat
    # --------------------------------------------------
    #   RHS
    # --------------------------------------------------
    def _rhs(self, ts, t, y, ydot):
        # -- unpack state
        rho_flat, Bq = self._unpack_state(y)
        # -- electronic section
        rho_np = []
        drho_flat = np.zeros_like(rho_flat)
        offset = 0
        # iterate (ik,isp)
        for ik in range(self._rho_e.nkpt):
            for isp in range(self._rho_e.nspin):
                rho_k = rho_flat[offset: offset+self._nbnd ** 2].reshape(self._nbnd, self._nbnd, order="F")
                rho_np.append(rho_k)
                # get polaron Hamilt.
                TilHe_k = self._He.compute_Hp_k(ik, isp, Bq, self._gql, self._phi.map_qtomq)
                # Liouville action: drho/dt = -i/hbar [H, rho]
                drho_k = -1j / hbar * (TilHe_k @ rho_k - rho_k @ TilHe_k)
                drho_flat[offset: offset+self._nbnd ** 2] = drho_k.flatten(order="F")
                offset += self._nbnd ** 2
        # --- Ehrenfest part ---
        if self.solver_type == "ARKIMEX":
            dBq = self._phi.compute_dBq_dt(Bq, None, self._gql, rho_np, t)
        else:
            dBq = self._phi.compute_dBq_dt(Bq, self._omega_q, self._gql, rho_np, t)
        dBq_flat = dBq.reshape(-1, order="C")
        ydot.setArray(np.concatenate([drho_flat, dBq_flat]))
    def _rhs_linear(self, ts, t, y, ydot, F):
        """
        Implicit (stiff) linear part:
            dBq/dt = -i/hbar * omega_q * Bq
            drho/dt = 0
        """
        # -- unpack state
        rho_flat, Bq = self._unpack_state(y)
        # -- electronic section
        drho_flat = np.zeros_like(rho_flat)
        # --- phonons: harmonic part ---
        assert self._omega_q.shape == Bq.shape
        dBq = -1j / hbar * self._omega_q * Bq
        dBq_flat = dBq.reshape(-1, order="C")
        F.setArray(np.concatenate([drho_flat, dBq_flat]))
    '''
    def _test_rhs(self, y):
        self._renorm_gql(y)
        # -- unpack state
        rho_flat, Bq = self._unpack_state(y)
        # -- electronic section
        rho_np = []
        drho_flat = np.zeros_like(rho_flat)
        offset = 0
        for ik in range(self._rho_e.nkpt):
            for isp in range(self._rho_e.nspin):
                rho_k = rho_flat[offset: offset+self._nbnd ** 2].reshape(self._nbnd, self._nbnd, order="F")
                rho_np.append(rho_k)
                # get polaron Hamilt.
                TilHe_k = self._He.compute_Hp_k(ik, isp, Bq, self._gql, self._phi.map_qtomq)
                # Liouville action: drho/dt = -i/hbar [H, rho]
                drho_k = -1j / hbar * (TilHe_k @ rho_k - rho_k @ TilHe_k)
                drho_flat[offset: offset+self._nbnd ** 2] = drho_k.flatten(order="F")
                offset += self._nbnd ** 2
        # --- Ehrenfest part ---
        dBq = self._phi.compute_dBq_dt(Bq, self._omega_q, self._gql, rho_np)
        dBq_flat = dBq.reshape(-1, order="C")
        ydot.setArray(np.concatenate([drho_flat, dBq_flat]))
    '''
    # --------------------------------------------------
    #  set ODE solver
    # --------------------------------------------------
    def _initialize_ODE_solver(self, y):
        # reset solver first
        self._reset_ts(y)
        self.ts.setProblemType(PETSc.TS.ProblemType.NONLINEAR)
        self.ts.setRHSFunction(self._rhs)              # explicit
        if self.solver_type == "ARKIMEX":
            self.ts.setIFunction(self._rhs_linear)     # implicit
    # --------------------------------------------------
    #  Initial state
    # --------------------------------------------------
    def _set_initial_state(self):
        """ Build the global initial state vector:
        y = [ vec(rho(k=0,isp=0)),
              vec(rho(k=0,isp=1)),
              ...
              vec(rho(k=Nk-1,isp=Ns-1)),
              B_q ]
        """
        rho_blocks = []
        for ik in range(self._rho_e.nkpt):
            for isp in range(self._rho_e.nspin):
                # get DM(k)
                rho_k = self._rho_e.get_PETSc_rhok(ik, isp)
                rho_np = rho_k.getDenseArray()
                rho_vec = rho_np.flatten(order="F")
                rho_blocks.append(rho_vec)
        rho_full = np.concatenate(rho_blocks)
        assert rho_full.shape[0] == self._rho_e.nkpt * self._rho_e.nspin * self._nbnd ** 2
        # Ehrenfest / ph. field
        Bq0 = self._phi.Bq.copy()
        B_vec = Bq0.reshape(-1, order="C")
        # --- full state vector ---
        y0 = np.concatenate([rho_full, B_vec])
        # PETSc vector
        y = PETSc.Vec().createSeq(len(y0), comm=PETSc.COMM_SELF)
        y.setArray(y0)
        y.assemble()
        return y
    # ---------------------------------------------
    # Perform time evolution
    # ---------------------------------------------
    def propagate(self, *, He, rho_e, ehr_field, gql, omega_q, ph_drive, **kwargs):
        if gql is None or ehr_field is None:
            log.error(
                "ElecPhCoupledSolver requires eph and ehr_field"
            )
        # set internal objects
        self._nbnd = He.nbnd
        self._rho_e = rho_e
        self._phi = ehr_field
        # set drive
        self._phi.set_drive(ph_drive)
        self._He = He
        self._gql = gql
        self._omega_q = omega_q     # eV
        print("max omega_q:", np.max(self._omega_q))
        print("dt * omega_q:", self.dt * np.max(self._omega_q / hbar))
        # check variables consistency
        assert self._He.nkpt == self._rho_e.nkpt
        assert self._He.nspin == self._rho_e.nspin
        assert self._He.nspin == self._rho_e.nspin
        # set td objects
        self._rho_e.init_td_arrays(self.nsteps, self.save_every)
        self._phi.init_td_arrays(self.nsteps, self.save_every)
        # set initial state
        y = self._set_initial_state()
        self._renorm_gql(y)
        # set propagator
        self._initialize_ODE_solver(y)
        # monitor
        self.ts.setMonitor(self.monitor)
        # solve dynamics
        self.ts.solve(y)
        return [self._rho_e, self._phi]