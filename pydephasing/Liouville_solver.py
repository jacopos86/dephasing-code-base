from pydephasing.real_time_solver_base import RealTimeSolver
from pydephasing.log import log
from pydephasing.mpi import mpi
from pydephasing.set_param_object import p
from common.phys_constants import hbar
from common.matrix_operations import commute
import numpy as np

#
#      Liouville solver without nuclear spins
#

class LiouvilleSolver(RealTimeSolver):
    def __init__(self):
        super().__init__(LiouvilleSolver, self).__init__()
    # evolve DM
    def evolve(self, dt, T, rho, H, Bfield):
        # set time arrays
        time = self.set_time_array(dt/2., T)
        # compute ext. magnetic field
        B = Bfield.of_t(time)
        # dynamics purely
        # drho/dt = -i[H, rho]
        # propagate DM
        self.propagate(rho, H, B, dt, T)
        # compute observables
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
        for t in range(nt):
            y = np.zeros((n,n), dtype=np.complex128)
            y[:,:] = rh[:,:,t]
            # F1 (ps^-1)
            h = H.set_hamilt_oft(2*t, B)
            F1 = 1j * commute(h, y) / hbar
            K1 = self.dt * F1
            y1 = y + K1 / 2.0