from pydephasing.set_param_object import p
from utilities.log import log
import numpy as np

#
#   Pauli polynomial class definition
#

class PauliPolynomial:
    def __init__(self, repr_mode, pol=None):
        self._repr_mode = repr_mode
        self._pol = []
        if pol is not None:
            self._pol.extend(pol)
            self._reduce()
    def return_polynomial(self):
        return self._pol
    def count_number_terms(self):
        return len(self._pol)
    def add_term(self, pt):
        self._pol.append(pt)
    def __add__(self, pp):
        if not isinstance(pp, PauliPolynomial):
            return NotImplemented
        assert self._repr_mode == pp._repr_mode
        new_pol = self._pol + pp.return_polynomial()
        return PauliPolynomial(self._repr_mode, new_pol)
    def __iadd__(self, pp):
        if not isinstance(pp, PauliPolynomial):
            return NotImplemented
        assert self._repr_mode == pp._repr_mode
        self._pol.extend(pp.return_polynomial())
        self._reduce()
        return self
    def __mul__(self, pp):
        if not isinstance(pp, PauliPolynomial):
            return NotImplemented
        assert self._repr_mode == pp._repr_mode
        pp_new = PauliPolynomial(self._repr_mode)
        n1 = self.count_number_terms()
        n2 = pp.count_number_terms()
        for i1 in range(n1):
            pt_1 = self._pol[i1]
            for i2 in range(n2):
                pt_2 = pp.return_polynomial()[i2]
                pt_new = pt_1 * pt_2
                pp_new.add_term(pt_new)
        return pp_new._reduce()
    def _reduce(self):
        pol_temp = list(np.copy(self._pol))
        self._pol = []
        while len(pol_temp) > 0:
            pt = pol_temp.pop()
            strng = pt.pw
            equal = False
            for i in range(len(self._pol)):
                if self._pol[i].pw == strng:
                    equal = True
                    self._pol[i].p_coeff += pt.p_coeff
                    break
            if not equal:
                self._pol.append(pt)
    def visualize_polynomial(self):
        n = self.count_number_terms()
        log.info("\t " + p.sep)
        for i in range(n):
            pt = self._pol[i]
            strng = "\t " + str(i) + " -> " + str(pt.p_coeff) + " "
            for iq in range(len(pt.pw)):
                strng += pt.pw[iq].symbol
            log.info(strng)
        log.info("\t " + p.sep)