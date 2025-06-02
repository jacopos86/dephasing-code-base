import numpy as np
from math import *

#
#  magnetic field class
#

class magnetic_field(object):
    def __init__(self, B_dict):
        # magnetic field expression
        self.expr_x = B_dict['expr_x']
        self.expr_y = B_dict['expr_y']
        self.expr_z = B_dict['expr_z']
        # list of safe methods
        self.safe_dict = {'acos': acos, 'asin': asin, 'atan': atan, 'atan2': atan2, 'cos': cos,
            'cosh': cosh, 'exp': exp, 'log': log, 'log10': log10, 'pi': pi, 'pow': pow, 'sin': sin, 
            'sinh': sinh, 'sqrt': sqrt, 'tan': tan, 'tanh': tanh}
        # creating a dictionary of safe methods
        self.var = B_dict['var']
        print(self.var, self.safe_dict)
    def set_pulse_oft(self, time):
        nt = len(time)
        Bt = np.zeros((3, nt))
        for t in range(nt):
            x = time[t]
            self.safe_dict[self.var] = x
            Bt[0,t] = eval(self.expr_x, {}, self.safe_dict)
            Bt[1,t] = eval(self.expr_y, {}, self.safe_dict)
            Bt[2,t] = eval(self.expr_z, {}, self.safe_dict)
        return Bt
    def set_pulse_sequence(self):
        pass