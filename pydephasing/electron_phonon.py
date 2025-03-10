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
    def read_eph_matrix(self):
        pass