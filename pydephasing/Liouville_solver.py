from pydephasing.real_time_solver_base import RealTimeSolver
from pydephasing.log import log
from pydephasing.mpi import mpi
from pydephasing.set_param_object import p
from common.phys_constants import hbar
from common.matrix_operations import commute
import numpy as np
from pydephasing.magnetic_field import magnetic_field

#
#    This module implements
#    the Liouville solver for DM dynamics
#

class LiouvilleSolverSpin(RealTimeSolver):
    def __init__(self):
        super(LiouvilleSolverSpin, self).__init__()
    # evolve DM
    def evolve(self, dt, T, rho, H):
        # set time arrays
        self.set_time_arrays(dt, T)
        # set magnetic field
        B = magnetic_field().set_pulse(self.time_dense)
        # dynamics
        # drho/dt = -i[H, rho]
        # propagate DM
        self.propagate(rho, H, B)
        # compute observables
    # time propagation DM
    # drho/dt = i[H,\rho]/\hbar
    def propagate(self, rho, H, B):
        # initialize DM
        rh0 = rho.matr0
        n = rh0.shape[0]
        if mpi.rank == mpi.root:
            log.info("\n")
            log.info("\t " + p.sep)
            log.info("\t STARTING REAL TIME EVOLUTION")
            log.info("\t nt: " + str(self.nt))
            log.info("\t dt (ps): " + str(self.dt))
            log.info("\t T (ps): " + str(self.T))
            log.info("\t " + p.sep)
        rh = np.zeros((n, n, self.nt), dtype=np.complex128)
        rh[:,:,0] = rh0[:,:]
        # iterate over time step
        for t in range(int(self.nt/2)-1):
            y = np.zeros((n,n), dtype=np.complex128)
            y[:,:] = rh[:,:,t]
            # F1 (ps^-1)
            #h = H.set_hamilt_oft(2*self.time_dense[2*t], unprt_struct, B[2*t], nuclear_config)
            #F1 = 1j * commute(h, y) / hbar
           # K1 = self.dt * F1
            #y1 = y + K1 / 2.0

class LiouvilleSolverHFI(RealTimeSolver):
    def __init__(self):
        super(LiouvilleSolverHFI, self).__init__()
    # evolve DM with HFI
    def evolve(self, dt, T, rho, H):
        # set time arrays
        self.set_time_arrays(dt, T)