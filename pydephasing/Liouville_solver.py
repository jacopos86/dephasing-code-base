from pydephasing.real_time_solver_base import RealTimeSolver
from pydephasing.log import log
from pydephasing.mpi import mpi
from pydephasing.set_param_object import p
from common.phys_constants import hbar
from common.matrix_operations import commute
from pydephasing.observables import compute_spin_mag
import numpy as np

#
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
        import matplotlib.pyplot as plt
        plt.plot(time[:], B[2,:])
        plt.show()
        # dynamics purely
        # drho/dt = -i[H, rho]
        # propagate DM
        time, rh = self.propagate(rho, H, B, dt, T)
        rho.set_density_matr(time, rh)
        tr = rho.trace_oft()
        plt.plot(time, tr)
        plt.show()
        print(tr, rh[0,0,:])
        # compute observables
        # spin 
        Mt = compute_spin_mag(H, rho)
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