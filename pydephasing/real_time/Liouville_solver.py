import numpy as np
from pydephasing.parallelization.petsc import *
from pydephasing.real_time.real_time_solver_base import RealTimeSolver
from pydephasing.utilities.log import log
from pydephasing.parallelization.mpi import mpi
from mpi4py import MPI  ## AG may need to adjust later
from pydephasing.set_param_object import p
from pydephasing.common.phys_constants import hbar
from pydephasing.common.matrix_operations import commute

#
#      TODO rewrite -> implement Liouvile solver from base class in real_time_solver
#      Liouville solver without nuclear spins
#

class LiouvilleSolverSpin(RealTimeSolver):
    def __init__(self):
        super(LiouvilleSolverSpin, self).__init__()
    # evolve DM
    def evolve(self, dt, T, rho, H, Bfield):
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
        # dynamics purely
        # drho/dt = -i[H, rho]
        # propagate DM
        time, rh = self.propagate(rho, H, B, dt, T)
        rho.set_density_matr(time, rh)
        if mpi.rank == mpi.root:
            units = {'time': 'ps', 'quantity': ''}
            file_name = "rho_oft.yml"
            file_name = "{}".format(p.write_dir + '/' + file_name)
            self.write_obj_to_file(time, rho.matr, units, file_name)
        mpi.comm.Barrier()
        # compute observables
        # spin 
        Mt = compute_spin_mag(H, rho)
        if mpi.rank == mpi.root:
            units = {'time': 'ps', 'quantity': ''}
            file_name = "spin_vec.yml"
            file_name = "{}".format(p.write_dir + '/' + file_name)
            self.write_obj_to_file(time, Mt, units, file_name)
        mpi.comm.Barrier()
    #
    #  set time propagation function
    def propagate(self, rho, H, B, dt, T):
        # initialize DM
        rh0 = rho.matr0
        n = rh0.shape[0]
        # time variables
        time = self.set_time_array(dt, T)
        nt = len(time)
        if mpi.rank == mpi.root:
            log.info("\n")
            log.info("\t " + p.sep)
            log.info("\t STARTING REAL TIME EVOLUTION")
            log.info("\t nt: " + str(nt))
            log.info("\t dt (ps): " + str(dt))
            log.info("\t T (ps): " + str(T))
            log.info("\t " + p.sep)
        rh = np.zeros((n, n, nt), dtype=np.complex128)
        rh[:,:,0] = rh0[:,:]
        # iterate over time step
        for t in range(nt-1):
            y = np.zeros((n,n), dtype=np.complex128)
            y[:,:] = rh[:,:,t]
            # F1 (ps^-1)
            h = H.set_hamilt_oft(2*t, B)
            F1 = 1j * commute(h, y) / hbar
            K1 = dt * F1
            y1 = y + K1 / 2.0
            # F2
            h = H.set_hamilt_oft(2*t+1, B)
            F2 = 1j * commute(h, y1) / hbar
            K2 = dt * F2
            y2 = y + K2 / 2.0
            # F3
            F3 = 1j * commute(h, y2) / hbar
            K3 = dt * F3
            y3 = y + K3
            # F4
            h = H.set_hamilt_oft(2*t+2, B)
            F4 = 1j * commute(h, y3) / hbar
            K4 = dt * F4
            rh[:,:,t+1] = y + (K1 + 2.*K2 + 2.*K3 + K4) / 6.0
        return time, rh

#
#       Liouville solver with nuclear spins
#

class LiouvilleSolverHFI(RealTimeSolver):
    def __init__(self):
        super(LiouvilleSolverHFI, self).__init__()
    # evolve DM
    def evolve(self, dt, T, rho, H, Bfield):
        pass
    def propagate(self, rho, H, B, dt, T):
        rh0 = rho.matr0
        n = rh0.shape[0]
        # time variables
        time = self.set_time_array(dt, T)
        nt = len(time)
        if mpi.rank == mpi.root:
            log.info("\n")
            log.info("\t " + p.sep)
            log.info("\t STARTING REAL TIME EVOLUTION")
            log.info("\t nt: " + str(nt))
            log.info("\t dt (ps): " + str(dt))
            log.info("\t T (ps): " + str(T))
            log.info("\t " + p.sep)
        rh = np.zeros((n, n, nt), dtype=np.complex128)

# ========================================================================
# 
#
#    ELECTRON DYNAMICS SOLVER
#          IT MUST SUPPORT:
#    1)  LIOUVILLE DYNAMICS
#
#
# ========================================================================

class LiouvilleSolverElectronic(ElecPhDynamicSolverBase):
    """
    Minimal Liouville-von Neumann solver for density matrix dynamics:
        drho/dt = -i/hbar [H, rho]
    Can be extended later for electron-phonon coupling or other dissipators.
    """
    TITLE="LIOUVILE COHERENT DYNAMICS"
    def __init__(self, evol_params):
        super().__init__(evol_params)
        # Lindbladian
        self.L = None
    # ---------------------------------------------
    # PETSc monitor
    # ---------------------------------------------
    def monitor(self, ts, step, t, y):
        if step % self.save_every != 0:
            return

        # 1. Gather full vector to Rank 0
        if not hasattr(self, '_scatter_ctx'):
            self._scatter_ctx, self._v_full = PETSc.Scatter.toZero(y)

        self._scatter_ctx.begin(y, self._v_full, PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD)
        self._scatter_ctx.end(y, self._v_full, PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD)

        if mpi.rank == mpi.root:
            nb = self._nbnd
            nb2 = nb ** 2
            nspin = self._rho_e.nspin
            nkpt = self._rho_e.nkpt
                
            v_full_np = self._v_full.getArray(readonly=True)
            rho_all_blocks = v_full_np.reshape(-1, nb, nb) 
            for ik in range(nkpt):
                for isp in range(nspin):

                    # Calculate ik and isp based on the block_idx
                    block_idx = (ik * nspin) + isp
                    rho_block = rho_all_blocks[block_idx]
        
                    # Store values
                    self._rho_e.store_rho_time(ik, isp, step // self.save_every, rho_block)
                    self._rho_e.store_traces(ik, isp, step // self.save_every, np.trace(rho_block).real)

    # --------------------------------------------------
    # Liouville action: ydot = -i/hbar (H*rho - rho*H)
    # --------------------------------------------------
    def Liouvillian(self, Hk):
        dim = np.prod(Hk.shape)
        I = np.eye(Hk.shape[0], dtype=complex)
        # Liouvillian in column-stacked (Fortran) order: 
        #       L*rho = -i/hbar (H*rho - rho*H)
        L_np = -1j / hbar * (np.kron(Hk, I) - np.kron(I, Hk.T))
        L = PETSc.Mat().createDense(
            size=(dim,dim), 
            array=L_np.astype(np.complex128),
            comm=PETSc.COMM_SELF
        )
        L.assemble()
        return L

    def PETScLiouvillian(self, He):

        local_H = He.H0.getDiagonalBlock()

        n_size = He.nkpt * He.nspin
        nbnd = self._nbnd
        block_size = nbnd * nbnd  # (nBand x nBand flattened)
        global_dim = n_size * block_size

        L = PETSc.Mat().create(comm=PETSc.COMM_WORLD)
        L.setSizes((global_dim, global_dim))
        L.setType(PETSc.Mat.Type.BAIJ)
        L.setBlockSize(block_size)
        
        # Preallocate 1 block per row (diagonal only)
        L.setPreallocationNNZ(1) 

        rstart, rend = L.getOwnershipRange()
        block_size_L = L.getBlockSize()
        block_size_H = He.H0.getBlockSize()
        
        h_start, _ = He.H0.getOwnershipRange()
        # Iterate only over the systems assigned to this rank
        first = rstart // block_size_L
        last = rend // block_size_L

        I = np.eye(nbnd, dtype=complex)
        for i in range(first, last):
            ik = i // He.nspin
            isp = i % He.nspin
            
            # Get Hamiltonian on this rank from global index
            h_i_global = np.arange(i * block_size_H, (i + 1) * block_size_H, dtype=np.int32)
            h_i_local = h_i_global - h_start
            Hk = local_H.getValues(h_i_local, h_i_local).reshape(block_size_H, block_size_H)
            # Compute the Liouvillian 
            # L*rho = -i/hbar (H*rho - rho*H)
            L_np = -1j / hbar * (np.kron(Hk, I) - np.kron(I, Hk.T))
            
            # Insert the block into the global matrix
            L.setValuesBlocked([i], [i], L_np.flatten(), addv=PETSc.InsertMode.INSERT_VALUES)


        L.assemblyBegin()
        L.assemblyEnd()
        return L

    # --------------------------------------------------
    # RHS callback (explicit solvers)
    # --------------------------------------------------
    def _rhs_linear(self, ts, t, y, ydot):
        self.L.mult(y, ydot)
    # --------------------------------------------------
    # Attach linear ODE to TS
    # --------------------------------------------------
    def _initialize_linear_ODE_solver(self, y):
        # reset solver first
        self._reset_ts(y)
        self.ts.setProblemType(PETSc.TS.ProblemType.LINEAR)
        if self.solver_type in ("RK4", "RK45", "EULER"):
            self.ts.setRHSFunction(self._rhs_linear)
        elif self.solver_type == "CN":
            self.ts.setRHSJacobian(self.L, self.L)
    # ---------------------------------------------
    # Initial state
    # ---------------------------------------------
    def _set_initial_state(self):
        # get DM (k)
        rho_k = self._rho_e.get_PETSc_rhok(self._ik, self._isp)
        # create PETSc arrays
        # Flatten rho into a vector for PETSc TS
        rho_np = rho_k.getDenseArray()
        y0 = rho_np.flatten(order="F")
        # y vector
        y = PETSc.Vec().createSeq(len(y0), comm=PETSc.COMM_SELF)
        y.setValues(range(len(y0)), y0)
        y.assemble()
        return y
    # ---------------------------------------------
    # Perform time evolution
    # ---------------------------------------------
    def propagate(self, *, He, rho_e, **kwargs):
        # set objects
        self._nbnd = He.nbnd
        self._rho_e = rho_e
        # check variables consistency
        nkpt = He.nkpt
        nspin= He.nspin
        assert self._rho_e.nbnd == self._nbnd
        assert self._rho_e.nkpt == nkpt
        assert self._rho_e.nspin == nspin
        # set time dependent object
        self._rho_e.init_td_arrays(self.nsteps, self.save_every)
        # iterate over k / spin
        self.L = self.PETScLiouvillian(He)
        y = self.L.createVecRight()
        self._rho_e.rho.getDiagonal(y)
        # set propagator
        self._initialize_linear_ODE_solver(y)
        # attach monitor for storing rho and trace
        self.ts.setMonitor(self.monitor)
        #  solve dynamics
        self.ts.solve(y)
        return self._rho_e
    # summary mode
    def summary(self):
        if mpi.rank == mpi.root:
            log.info("\n")
            log.info("\t " + p.sep)
            log.info("\t " + self.TITLE)
            log.info(f"\t TIME STEP: {self.dt} ps")
            log.info(f"\t NUMBER OF STEPS: {self.nsteps}")
            log.info("\t REAL TIME SOLVER: " + self.solver_type.upper())
            log.info("\t " + p.sep)
            #log.info("Shape of Rho Property " + f"{self._rho_e.shape}")
            #for _i in dir(self.monitor):
            #    log.info(f"\t{_i}")
            log.info("\n")
