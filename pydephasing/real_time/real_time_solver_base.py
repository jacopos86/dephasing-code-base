from abc import ABC
import numpy as np
from pathlib import Path
import yaml

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
    # write time object to file
    def write_obj_to_file(self, t, var_oft, units, file_name):
        # write data on file
        fil = Path(file_name)
        if not fil.exists():
            data = {'time': t, 'quantity': var_oft, 'units': units}
            with open(file_name, 'w') as out_file:
                yaml.dump(data, out_file)