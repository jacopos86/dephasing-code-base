#
#   Pauli polynomial class definition
#

class PauliPolynomial:
    def __init__(self, pol=None):
        self.__pol = []
        if pol is not None:
            self.__pol.extend(pol)
            self._reduce()
    def return_polynomial(self):
        return self.__pol
    def count_number_terms(self):
        return len(self.__pol)
    def __add__(self, pp):
        if not isinstance(pp, PauliPolynomial):
            return NotImplemented
        new_pol = self.__pol + pp.return_polynomial()
        return PauliPolynomial(new_pol)
    def _reduce(self):
        pass