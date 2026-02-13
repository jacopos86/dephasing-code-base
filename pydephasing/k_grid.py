import numpy as np
from pydephasing.common.print_objects import print_1D_array

#
#   1D Q grid -> USED in MODEL
#

class KGrid_1D:
    def __init__(self, nkpt):
        self.nk = nkpt
        self.kpts = None
        self.wk = None
    def set_kgrid(self, L):
        self.kpts = np.linspace(0.5, -0.5, self.nk) * 2.*np.pi / L
    def set_wk(self):
        self.wk = np.ones(self.nk) / self.nk
        assert np.isclose(np.sum(self.wk), 1.0)

    def map_kmq(self, ik, q):
        """
        return index ik' such that k[ik'] = k[ik] - q (mod G)
        """
        kmq = self.kpts[ik] - q
        diff = self.kpts - kmq
        ikp = np.argmin(np.linalg.norm(diff, axis=1))
        return ikp
    def get_kpts(self):
        return self.kpts
    def get_k_weights(self):
        return self.wk
    def show_kgr(self):
        print_1D_array(self.kpts)
    def show_kw(self):
        print_1D_array(self.wk)
