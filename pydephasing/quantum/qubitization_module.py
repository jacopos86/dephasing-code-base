from parallelization.mpi import mpi
from utilities.log import log
from pydephasing.set_param_object import p
from quantum.pauli_polynomial_class import PauliPolynomial

#
#   This module defines the function to build
#   the qubit Hamiltonian
#   H = \sum_i c_i |{x,y,z}...{x,y,z}>
#   H is written in the form of pauli polynomial
#

