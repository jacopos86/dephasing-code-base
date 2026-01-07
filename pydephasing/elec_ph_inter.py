import numpy as np
from abc import ABC, abstractmethod
from pydephasing.parallelization.mpi import mpi
from pydephasing.utilities.log import log
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
    @abstractmethod
    def compute_gql(self):
        """
        compute g_ql
        """
        pass

#
# ============================================================
#   Deformation Potential Electron-Phonon Coupling
# ============================================================
#

class DeformationPotentialElectronPhonon(ElectronPhononClass):
    """
    Deformation potential electron-phonon coupling.
    Supports 1-band or 2-band electronic models.
    """
    def __init__(self, deformation_potentials, density, volume):
        """
        Parameters
        ----------
        deformation_potentials : array-like
            D_n for each band [eV]
        density : float
            Mass density (amu / Å³ or consistent units)
        volume : float
            Simulation cell volume (Å³)
        """
        super().__init__()
        self.D = np.asarray(deformation_potentials, dtype=float)
        self.rho = density
        self.volume = volume
        if mpi.rank == mpi.root:
            log.info("\t " + p.sep)
            log.info("\t Deformation Potential E-P Model")
            log.info("\t Bands : %d" % len(self.D))
            log.info("\t " + p.sep)
    # --------------------------------------------------------
    #   Compute g_{mn}(q,λ)
    # --------------------------------------------------------
    def compute_gql(self, q_vectors, phonon_freqs, phonon_polarizations):
        """
        Parameters
        ----------
        q_vectors : ndarray (nq, 3)
        phonon_freqs : ndarray (nq, nmode)
            Phonon frequencies ω_{qλ}
        phonon_polarizations : ndarray (nq, nmode, 3)

        Returns
        -------
        g_ql : ndarray (nbnd, nbnd, nq, nmode)
        """
        nbnd = len(self.D)
        nq = q_vectors.shape[0]
        nmode = phonon_freqs.shape[1]
        g_ql = np.zeros(
            (nbnd, nbnd, nq, nmode),
            dtype=np.complex128
        )
        # ħ in eV·fs
        hbar = 0.6582119514
        for iq in range(nq):
            q = q_vectors[iq]
            qnorm = np.linalg.norm(q)
            if qnorm < 1e-10:
                continue
            for il in range(nmode):
                wql = phonon_freqs[iq, il]
                if wql <= 0.0:
                    continue
                eq = phonon_polarizations[iq, il]
                prefactor = np.sqrt(
                    hbar / (2.0 * self.rho * self.volume * wql)
                )
                q_dot_e = np.dot(q, eq)
                for ib in range(nbnd):
                    g_ql[ib, ib, iq, il] = (
                        self.D[ib] * prefactor * q_dot_e
                    )
        self.g_ql = g_ql
        return g_ql