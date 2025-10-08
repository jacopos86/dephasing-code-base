from utilities.log import log
from quantum.pauli_polynomial_class import PauliPolynomial
from quantum.pauli_letters_module import PauliLetter
from set_param_object import p

#
#   This module defines the basic Pauli term entity
#   pt = c_i |zyxz...xxz>
#   each term is a dictionary with key -> string of |xyxz...>
#   pointing to a complex coefficient
#   we override basic operations like +, +=, -, -=

class PauliTerm:
    def __init__(self, nq, ps=None, pc=None, pl_seq=None):
        # pw is a list of pauli letters -> word
        self.p_coeff = None
        self.pw = None
        self.__nq = nq
        if ps is not None:
            assert len(ps) == self.__nq
        if ps is not None and pc is not None:
            self.__set_pauli_term(ps, pc)
        elif pl_seq is not None and pc is not None:
            self.__set_pauli_term_from_letters(pl_seq, pc)
        else:
            log.error("WRONG INITIALIZATION")
    def __set_pauli_term(self, ps, pc):
        p_seq = []
        for iq in range(self.__nq):
            pl = PauliLetter(ps[iq])
            p_seq.append(pl)
        self.pw = p_seq
        self.p_coeff = pc
        print(self.pw)
    def __set_pauli_term_from_letters(self, pl_seq, p_coeff):
        self.p_coeff = p_coeff
        self.pw = ""
        for pl in pl_seq:
            self.p_coeff *= pl.phase
            self.pw += pl.symbol
    def __mul__(self, pt):
        if not isinstance(pt, PauliTerm):
            return NotImplemented
        pl_seq = []
        for iq in range(self.__nq):
            r = self.pw[iq]*pt.pw[iq]
            print(self.pw[iq].symbol, pt.pw[iq].symbol, r.symbol)
            pl_seq.append(r)
        return PauliTerm(self.__nq, pc=self.p_coeff*pt.p_coeff, pl_seq=pl_seq)
    def visualize(self):
        strng = "\t " + str(self.p_coeff) + self.pw
        log.info("\t " + p.sep)
        log.info(strng)
        log.info("\t " + p.sep)
#
#   Define cj^+ fermionic operator
#   in qubit representation

class fermion_plus_operator(PauliPolynomial):
    def __init__(self, repr_mode, nq, j):
        super().__init__(repr_mode)
        if j < 0 or j >= nq:
            log.error("index j out of range -> 0 <= j < nq")
        if self._repr_mode == "JW":
            self.__set_JW_operator(nq, j)
    def __set_JW_operator(self, nq, j):
        strng = ""
        for _ in range(nq, j+1, -1):
            strng += "e"
        strng += "x"
        for _ in range(j, 0, -1):
            strng += "z"
        self._pol.append(PauliTerm(nq, ps=strng, pc=0.5))
        strng = ""
        for _ in range(nq, j+1, -1):
            strng += "e"
        strng += "y"
        for _ in range(j, 0, -1):
            strng += "z"
        self._pol.append(PauliTerm(nq, ps=strng, pc=-0.5j))
        self._reduce()

#
#   Define cj fermionic operator
#   in qubit representation

class fermion_minus_operator(PauliPolynomial):
    def __init__(self, repr_mode, nq, j):
        super().__init__(repr_mode)
        if j < 0 or j >= nq:
            log.error("index j out of range -> 0 <= j < nq")
        if self._repr_mode == "JW":
            self.__set_JW_operator(nq, j)
    def __set_JW_operator(self, nq, j):
        strng = ""
        for _ in range(nq, j+1, -1):
            strng += "e"
        strng += "x"
        for _ in range(j, 0, -1):
            strng += "z"
        self._pol.append(PauliTerm(nq, ps=strng, pc=0.5))
        strng = ""
        for _ in range(nq, j+1, -1):
            strng += "e"
        strng += "y"
        for _ in range(j, 0, -1):
            strng += "z"
        self._pol.append(PauliTerm(nq, ps=strng, pc=0.5j))
        self._reduce()