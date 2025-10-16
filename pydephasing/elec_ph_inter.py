import numpy as np
from abc import ABC
from parallelization.mpi import mpi
from utilities.log import log
#
#   This module computes the electron-phonon matrix
#   g_munu(k,q)
#   read external data grad H0
#   and compute the matrix using the phonon modes
#
class ElectronPhononClass(ABC):
    def __init__(self):
        self.g_ql = None
    #
    # compute e-ph matrix from
    # grad_X He
    def compute_gql(self, ql_list, He, gHe):
        n = len(He.basis_vectors)
        g_ql = np.zeros((n, n, len(ql_list)), dtype=np.complex128)
        # phonons amplitudes