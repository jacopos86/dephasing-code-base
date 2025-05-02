from abc import ABC
import numpy as np

#    This module implements
#    the real time solver class
#    Takes Hamiltonian + density matrix object -> real time dynamics
#    ABSTRACT class

class RealTimeSolver(ABC):
    def __init__(self):
        # time interval
        self.dt = None
        # total simulation time
        self.T = None
        # n. time steps
        self.nt = None
    # set time arrays
    def set_time_arrays(self, dt, T):
        # n. time steps
        self.nt = int(T / dt)
        self.time = np.linspace(0., T, self.nt)
        # build dense array
        nt2 = int(T / (dt/2.))
        self.time_dense = np.linspace(0., T, nt2)
        # everything must be in fs