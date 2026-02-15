import numpy as np
from math import *
from dataclasses import dataclass
from typing import Union, Optional, Dict
from pydephasing.common.units import Q_, T

@dataclass
class MagneticField:
    """
    Data container for magnetic field.

    Attributes:
        B0: static field (Tesla) as np.ndarray (3,)
        Bt_expr: optional dictionary with expression strings for time-dependent field
        var: name of the variable in the expressions
        unit: unit of B0 and Bt_expr (default Tesla)
    """
    B0: Union[np.ndarray, list]
    unit : Union[str, object] = T
    Bt_expr: Optional[Dict[str, str]] = None
    var: str = 't'
    # initialization
    def __post_init__(self):
        self.B0 = self._to_tesla(self.B0, self.unit)
        self.unit = T
        if self.B0.shape != (3,):
            raise ValueError(f"B0 must be a 3-element vector, got shape {self.B0.shape}")
        # Optional: ensure Bt_expr has correct keys
        self.Bt_expr = self._set_Bt_expr(self.Bt_expr, self.var)
    @staticmethod
    def _to_tesla(array, unit) -> np.ndarray:
        ''' convert array to Tesla ndarray'''
        q = Q_(array, unit).to(T)
        return np.asarray(q.magnitude, dtype=float)
    @staticmethod
    def _set_Bt_expr(Bt_expr: dict | None, var: str) -> dict | None:
        if Bt_expr is None:
            return None
        return {
            "expr_x": Bt_expr.get("expr_x", "0"),
            "expr_y": Bt_expr.get("expr_y", "0"),
            "expr_z": Bt_expr.get("expr_z", "0"),
            "var":    Bt_expr.get("var", var)
        }

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