from abc import ABC

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