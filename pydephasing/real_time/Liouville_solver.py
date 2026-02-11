import numpy as np
from petsc4py import PETSc
from pydephasing.real_time.real_time_solver_base import RealTimeSolver, ElecPhDynamicSolverBase
from pydephasing.utilities.log import log
from pydephasing.parallelization.mpi import mpi
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
        if step % self.save_every == 0:
            # reshape vector back to density matrix
            rho_np = y.getArray().reshape(self._nbnd, self._nbnd, order="F")
            # store into rho_e object
            self._rho_e.store_rho_time(self._ik, self._isp, step // self.save_every, rho_np)
            # compute trace for diagnostics
            tr = np.trace(rho_np).real
            self._rho_e.store_traces(self._ik, self._isp, step // self.save_every, tr)
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
        for ik in range(nkpt):
            self._ik = ik
            for isp in range(nspin):
                self._isp = isp
                # get Hamiltonian H(k)
                Hk = He.get_PETSc_Hk(ik, isp)
                # 1. Build Liouvillian operator (implicit for CN)
                # Liouville action: ydot = -i/hbar (H*rho - rho*H)
                self.L = self.Liouvillian(Hk.getDenseArray())
                # initial vector
                y = self._set_initial_state()
                # set propagator
                self._initialize_linear_ODE_solver(y)
                # attach monitor for storing rho and trace
                self.ts.setMonitor(self.monitor)
                #  solve dynamics
                self.ts.solve(y)
        return [self._rho_e]