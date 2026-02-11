import numpy as np
from abc import ABC, abstractmethod
from pydephasing.parallelization.mpi import mpi
from pydephasing.utilities.log import log
from pydephasing.set_param_object import p
from pydephasing.common.phys_constants import hbar
from pydephasing.common.print_objects import print_2D_matrix

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
    else:
        log.error("Do not call eph_initialize to initialize eph model")

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
    def __init__(self, eph_params, nBands, sys_size):
        """
        Parameters
        ----------
        deformation_potentials : array-like
            D_n for each band [eV]
        density : float
            Mass density (eV*fs^2/Å^5)
        form_factor:
            f(k)
        """
        super().__init__()
        # n. bands
        self.nbnd = nBands
        # deformation pot.
        self.D = np.asarray(eph_params['deformation_pot'], dtype=float)
        self.check_D_dimensions()
        # define form factors
        self.ff_data = eph_params['form_factor']
        print(self.ff_data)
        # density / volume
        self.rho = eph_params['rho']
        # [rho] -> eV fs^2 / Ang^5
        self.rho = self.rho * (1.e-3) ** 2
        # [rho] -> eV ps^2 / Ang^5
        self.volume = sys_size ** 3
        # Ang^3
    def check_D_dimensions(self):
        # check dimensions
        if self.D.ndim == 2:
            # 2D array: must be square and match nbnd
            if self.D.shape != (self.nbnd, self.nbnd):
                log.error(
                    f"deformation_pot matrix shape {self.D.shape} "
                    f"does not match (nbnd, nbnd)=({nbnd}, {nbnd})"
                )
            # optionally enforce symmetry
            if not np.allclose(self.D, self.D.T):
                log.error("deformation_pot matrix must be symmetric")
        else:
            log.error("deformation_pot must be 2D array")
    # --------------------------------------------------------
    #   form factors calculations
    # --------------------------------------------------------
    def set_form_factors(self, kp, qp):
        kabs = np.abs(kp)
        print(self.ff_data["type"], self.ff_data["k0"])
        ftype = self.ff_data["type"].lower()
        if ftype == "constant":
            return np.ones_like(kabs)
        k0 = self.ff_data["k0"]
        if ftype == "gaussian":
            return np.exp(-kabs**2 / (2 * k0**2))
        if ftype == "lorentzian":
            return 1.0 / (1.0 + (kabs / k0)**2)
        log.error(f"Unknown form factor: {ftype}")
    # --------------------------------------------------------
    #   Compute g_{mn}(q,λ)
    # --------------------------------------------------------
    def compute_gql(self, qgr, kgr, ph):
        """ g_ql : ndarray (nbnd, nbnd, nk, nq*nmode) """
        nq = qgr.nq
        nk = kgr.nk
        # q / k vec.
        qp = qgr.get_qpts()
        kp = kgr.get_kpts()
        g_ql = np.zeros(
            (self.nbnd, self.nbnd, nk, nq*ph.nmodes), 
            dtype=np.complex128
        )
        # set up form factors
        f_kq = self.set_form_factors(kp, qp)
        exit()
        # ħ in eV·ps
        omegaSq, Wq = ph.compute_ph_state_q(qp)
        omega_q = np.sqrt(omegaSq)
        # THz
        iql = 0
        for iq in range(nq):
            qnorm = np.linalg.norm(qp[iq])
            if qnorm < 1e-10:
                continue
            qv = np.array([qp[iq], 0.0, 0.0])
            for il in range(ph.nmodes):
                wql = omega_q[iq,il]
                if wql <= 0.0:
                    continue
                eq = Wq[iq,il,:]
                prefactor = np.sqrt(
                    hbar / (2.0 * self.rho * self.volume * wql)
                )
                q_dot_e = np.dot(qv, eq)
                g_ql[:,:,iql] = (
                    self.D[:,:] * prefactor * q_dot_e
                )
                iql += 1
        return g_ql
    # print summary info
    def summary(self):
        """
        Print basic info
        """
        if mpi.rank == mpi.root:
            log.info("\t " + p.sep)
            log.info("\t Deformation Potential E-P Model")
            log.info("\t Bands : %d" % self.nbnd)
            log.info("\t D:\n")
            print_2D_matrix(self.D)
            log.info(f"\t volume: {self.volume} ang^3")
            log.info(f"\t density: {self.rho} eV ps^2/ang^5")
            log.info("\t " + p.sep)

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
        self.nBands = nBands
    def compute_gql(self, gradH):
        gradHe = gradH.read_grad_He_matrix(self.band_range_idx, self.nModes, self.nBands)