import numpy as np
#
#   This module computes the electron-phonon matrix
#   g_munu(k,q)
#   read external data grad H0
#   and compute the matrix using the phonon modes
#
class electron_phonon:
    def __init__(self, data_dir):
        self.data_dir = data_dir
    # read data
    # from saved file
    def read_eph_matrix(self):
        pass
    #
    # compute e-ph matrix from
    # grad_X He
    def compute_eph_matrix(self, nat, ql_list, He):
        n = len(He.basis_vectors)
        g_ql = np.zeros((n, n, len(ql_list)), dtype=np.complex128)
        # phonons amplitudes
        