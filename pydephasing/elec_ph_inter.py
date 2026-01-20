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

#
#.  initialization function
#

def eph_initialization(APPROX_MODEL, band_range_idx, nModes, nBands):
    if APPROX_MODEL == "CCA":
        return ElectronPhononCentralCellApprox(band_range_idx, nModes, nBands)

#
# The abstract class for electron-phonon coupling
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

#
# ============================================================
#   central cell approx. Electron-Phonon Coupling
# ============================================================
#

class ElectronPhononCentralCellApprox(ElectronPhononClass):
    def __init__(self, band_range_idx, nModes, nBands):
        super().__init__()
        self.band_range_idx = band_range_idx
        self.nModes = nModes
        self.nBandsG = nBands # Global number of bands
        self.nBandsL = band_range_idx[1] - band_range_idx[0] + 1 # Range of bands for computation of g_ql
    def compute_gql(self, atoms, gradH, ph, qgr):
        # --------------------------------------------------------
        #   Compute g_{mn}(q,λ)
        # --------------------------------------------------------

        # Read atoms information
        nat = atoms.nat  # number of atoms in the unit cell
        atom_ia_map = atoms.index_to_ia_map  # indices to map from atom idx to mode index
        atom_masses = np.array([atoms.atoms_mass[atom_ia_map[jax]] for jax in range(self.nModes)])  # masses ordered by mode index

        # Load hamiltonian (He) gradient
        gradHe = gradH.read_grad_He_matrix(self.band_range_idx, self.nModes, self.nBandsG)
        print("gradH dim: ", gradHe.shape)

        # # Set up q grid
        nq, qpts = qgr.nq, qgr.qpts # q grid dim and points
        print("Num q points: ", nq)
        print("qpts shape: ", qpts.shape)

        # Compute phonon states on q grid
        omega_q, Wq = ph.compute_ph_state_q(qpts) # Compute phonon states at all q points. Eigenvalues and eigenvectors
        print("Phonon eigenvals: ", omega_q.shape)
        # print("Phonon eigenvals sample: ", omega_q[10])
        print("Phonon eigenvectors: ", Wq.shape)

        # Build ql list
        ql_list = mpi.split_ph_modes(nq, self.nModes)
        # # Initialize g_ql matrix
        gql = np.zeros((self.nBandsL, self.nBandsL, len(ql_list)), dtype=np.complex128)  # Initialize g_ql matrix
        # phonon amplitude
        A_ql = ph.compute_ph_amplitude_q(ql_list)
        # Compute g_ql
        iql = 0
        for iq, il in ql_list:
            # Iterate over q points and phonon modes
            phase_factor = qgr.compute_phase_factor(iq, nat)
            for jax in range(self.nModes):
                # Iterate over phonon modes
                # Calculate g_ql contribution
                gql[:,:,iql] += A_ql[iql] * phase_factor[jax] * Wq[iq][jax,il] * gradHe[:,:,jax] * 1/np.sqrt(atom_masses[jax])
                # [eV/ang * ang/eV^1/2 *ps^-1 * eV^1/2 ps]
                # = eV
            iql += 1
        print("gql shape: ", gql.shape)
        return gql

        # for iq in range(nq):
        #     # Iterate over q points
        #     # Wq[iq] matrix with dimension Nmodes x (Nmodes)
        #     phase_factor = qgr.compute_phase_factor(iq, nat)  # compute phase factor e^{iq R_k}
        #     phonon_amplitude = 1/np.sqrt(2*omega_q[iq][:,None,None])  # phonon amplitude factor
        #     sum = 0
        #     for il in range(self.nModes):
        #         # Iterate over phonon modes
        #         # Wq[iq,il] eigenvector at q for mode il  -> Dim = Nmodes
        #         term = Wq[iq,il][:,None,None] * gradHe * phase_factor[:,None,None] * 1/np.sqrt(atom_masses[il])
        #         sum += term
        #     gql[:,:,:,iq] = sum * phonon_amplitude # Final g_ql for q point iq
        # return gql






