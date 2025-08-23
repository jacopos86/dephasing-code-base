from abc import ABC
import numpy as np

#    This module implements
#    the real time solver class
#    Takes Hamiltonian + density matrix object -> real time dynamics
#    ABSTRACT class

class RealTimeSolver(ABC):
    def __init__(self):
        pass
    # set time arrays
    def set_time_array(self, dt, T):
        # n. time steps
        nt = int(T / dt)
        time = np.linspace(0., T, nt)
        # build dense array
        #nt2 = int(T / (dt/2.))
        #self.time_dense = np.linspace(0., T, nt2)
        # everything must be in ps
        #self.T = T
        #self.dt = dt
        return time