from parallelization.mpi import mpi
from utilities.log import log
from pydephasing.set_param_object import p
from pydephasing.pauli_polynomial_class import PauliPolynomial

#
#   This module defines the function to build
#   the qubit Hamiltonian
#   H = \sum_i c_i |{x,y,z}...{x,y,z}>
#   H is written in the form of pauli polynomial
#

def qubitize_spin_hamilt(H):
    #
    #  This function convert the spin Hamiltonian
    #  from fermion basis -> qubit basis
    #
    nq = H.shape[0]
    if mpi.rank == mpi.root:
        log.info("\t " + p.sep)
        log.info("\t " + "n. fermionic qubits in simulation: " + str(nq))
        log.info("\t " + p.sep)
    #  build Pauli polynomial
    pp = PauliPolynomial(nq)