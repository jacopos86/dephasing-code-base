from utilities.log import log

#
#   This module defines the basic Pauli term entity
#   pt = c_i |zyxz...xxz>
#   each term is a dictionary with key -> string of |xyxz...>
#   pointing to a complex coefficient
#   we override basic operations like +, +=, -, -=

class PauliTerm:
    def __init__(self, ps=None, pc=None):
        self.p_str = ""
        self.p_coeff = 0j
        self.__nq = None
        if ps is not None and pc is not None:
            self.p_str = ps
            self.p_coeff = pc
        elif ps is None and pc is not None:
            log.error("WRONG INITIALIZATION")
        elif ps is not None and pc is None:
            log.error("WRONG INITIALIZATION")
    def set_number_qubits(self, nq):
        self.__nq = nq