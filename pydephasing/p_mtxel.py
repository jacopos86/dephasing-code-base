import numpy as np
from abc import ABC, abstractmethod
from pydephasing.common.phys_constants import hbar

#
#   linear momentum classes
#

class LinearMomentum(ABC):
    """ Abstract base class for momentum matrix p_nm(k) """
    def __init__(self):
        self.p_k = None
    #  set p(k) matrix
    @abstractmethod
    def set_p_matrix(self, kgr):
        """ set momentum matrix """
        pass

#
#   Two-bands model linear momentum class
#

class TwoBandsLinearMomentum(LinearMomentum):
    def __init__(self, mtxel_params):
        """
        mtxel_params  : 
            interband coefficients
            model_type
            k0  : optional Gaussian cutoff
        """
        super().__init__()
        self.mtxel_params = mtxel_params
    def compute_interband_coeffs(self, k):
        """ Model for <u_n| -i∇ |u_m> """
        ptype = self.mtxel_params["type"].lower()
        p01 = self.mtxel_params["p01"]
        if ptype == "constant":
            return p01
        elif ptype == "gaussian":
            k0 = self.mtxel_params["k0"]
            return p01 * np.exp(-(k**2) / (2 * k0**2))
        log.error(f"Unknown model type: {ptype}")
    def set_p_matrix(self, kgr):
        # k pts
        nk = kgr.nk
        kp = kgr.get_kpts()
        # 2x2 matrix
        pe_k = np.zeros((2, 2, nk), dtype=np.complex128)
        # Intraband (ħ k δ_nm)
        # eV ps / A
        pe_k[0, 0, :] = hbar * kp[:]
        pe_k[1, 1, :] = hbar * kp[:]
        # Interband term
        for ik in range(nk):
            pe_k[0, 1, ik] = self.compute_interband_coeffs(kp[ik])
            pe_k[1, 0, ik] = pe_k[0, 1, ik].conj()
        return pe_k