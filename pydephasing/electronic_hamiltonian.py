import numpy as np
import gc
from abc import ABC, abstractmethod
from petsc4py import PETSc
from pydephasing.parallelization.mpi import mpi
from pydephasing.utilities.log import log
from pydephasing.set_param_object import p
from pydephasing.wannier_interface.wannier import Wannier
from pydephasing.common.phys_constants import hartree2ev, hbar, me
from pydephasing.utilities.plot_functions import plot_elec_struct

#
#   This module defines
#   the electronic Hamiltonian class
#

class AbstractElectronicHamiltonian(ABC):
    """
    Abstract base class for electronic Hamiltonians.
    Enforces a common interface and provides shared utilities.
    """
    def __init__(self, Ewin_Ha=None):
        # spectrum-related
        self.enk = None
        self.mu = None
        # Hamiltonian matrices
        self.H0 = None
        # dimensions
        self.nbnd = None
        self.nkpt = None
        self.nspin = None
        # plotting window (Hartree)
        self.Ewin_Ha = Ewin_Ha
    # ==========================================================
    #   REQUIRED INTERFACE
    # ==========================================================
    @abstractmethod
    def set_energy_spectrum(self, *args, **kwargs):
        """
        Must populate:
          self.enk [nbnd, nkpt, nspin]
          self.mu
        """
        pass
    @abstractmethod
    def set_H0_matr(self):
        """
        Must build:
          self.H0[k][spin] as PETSc.Mat
        """
        pass
    # ==========================================================
    #   SHARED FUNCTIONALITY
    # ==========================================================
    def plot_band_structure(self):
        """
        Generic band structure plotter.
        """
        if self.enk is None or self.mu is None:
            log.error("Band structure not initialized")
            return
        if self.Ewin_Ha is not None:
            Ew = np.array(self.Ewin_Ha) * hartree2ev
        else:
            Ew = None
        if mpi.rank == mpi.root:
            plot_elec_struct(self.enk, self.mu, Ylim=Ew)
        mpi.comm.Barrier()
    # clean PETSc objects
    def clean_up(self):
        """
        Robust PETSc cleanup (safe for repeated calls).
        """
        if self.H0 is not None:
            for k_list in self.H0:
                for mat in k_list:
                    if isinstance(mat, PETSc.Mat):
                        mat.destroy()
            self.H0 = None
        gc.collect()
    # ==========================================================
    #   DEBUG / INFO
    # ==========================================================
    def print_info(self, title="ELECTRONIC HAMILTONIAN"):
        if mpi.rank == mpi.root:
            log.info("\t " + p.sep)
            log.info(f"\t {title}")
            log.info(f"\t nbnd  : {self.nbnd}")
            log.info(f"\t nkpt  : {self.nkpt}")
            log.info(f"\t nspin : {self.nspin}")
            log.info("\t " + p.sep)

#
#   function : set electronic Hamiltonian
#

class electronic_hamiltonian(AbstractElectronicHamiltonian):
    def __init__(self, Ewin_Ha, wann=False):
        super().__init__(Ewin_Ha)
        # unpert. spectrum
        self.Vnk = None
        self.WANNIER = wann
    # set energy spectrum
    def set_energy_spectrum(self, elec_struct):
        if self.WANNIER:
            if mpi.rank == mpi.root:
                log.info("\n")
                log.info("\t " + p.sep)
                log.info("\t PERFORM WANNIER INTERPOLATION")
                log.info("\t " + p.sep)
                log.info("\n")
            wan_obj = Wannier(p.work_dir, elec_struct)
            wan_obj.plot_band_structure(n_interp=10, Ylim=self.Ewin_Ha)
            # extract elec. struct. (eV)
            self.enk, self.Vnk = wan_obj.get_band_structure(n_interp=1, units="eV")
        else:
            self.enk = elec_struct.get_band_structure(units="eV")
        self.mu = elec_struct.get_chem_potential(units="eV")
    # set H0 hamiltonian
    def set_H0_matr(self):
        Ew = np.array(self.Ewin_Ha) * hartree2ev
        d_enk = self.enk - self.mu
        # keep only bands within energy window
        Emin, Emax = Ew[0], Ew[1]
        band_set = []
        for ib in range(d_enk.shape[0]):
            # extract energies for band ib over all k-points
            eband = d_enk[ib,:,:]
            # check if ANY value is inside the energy window
            if np.any((eband >= Emin) & (eband <= Emax)):
                band_set.append(ib)
        self.nbnd = band_set[-1] - band_set[0] + 1
        self.nkpt = d_enk.shape[1]
        self.nspin = d_enk.shape[2]
        if mpi.rank == mpi.root:
            log.info("\t " + p.sep)
            log.info("\t H0 parameters: ")
            log.info("\t nbnd: " + str(self.nbnd))
            log.info("\t nkpt: " + str(self.nkpt))
            log.info("\t nspin: " + str(self.nspin))
            log.info("\t " + p.sep)
        # build PETSc matrices: H0[k][spin] = (nbnd_new x nbnd_new)
        self.H0 = []
        for ik in range(self.nkpt):
            Hk_list = []
            for ispin in range(self.nspin):
                # extract diagonal band energies inside window
                diag_vals = d_enk[band_set[0]:band_set[-1]+1, ik, ispin]
                # dense nbnd x nbnd matrix
                Hks_np = np.diag(diag_vals)
                # build PETSc matrix
                Hmat = PETSc.Mat().createDense(
                    size=(self.nbnd, self.nbnd),
                    array=Hks_np
                )
                Hmat.assemble()
                Hk_list.append(Hmat)
            self.H0.append(Hk_list)
    # electric dipole
    def add_dipole_interact(self):
        pass

#
# ============================================================
#   MODEL ELECTRONIC HAMILTONIAN
# ============================================================
#

class model_electronic_hamiltonian(AbstractElectronicHamiltonian):
    """
    Analytical electronic Hamiltonian for model calculations.
    Supports 1-band or 2-band effective mass models.
    """
    def __init__(self, elec_bands, nkpt, eff_mass, band_offset, chem_pot, Ewin_Ha):
        """
        En(k)=En0-hbar/2mn^2|k|^2
        Parameters
        ----------
        elec_bands: number of bands (1,2)
        nkpt : number k points
        eff_mass : array-like
            Effective masses (in units of m_e)
            length = nbnd
        band_offset : array-like
            Band edge energies [eV]
            length = nbnd (1 or 2)
        chem_pot : chemical potential [eV]
        Ewin_Ha : elec. energy window (Ha)
        """
        super().__init__(Ewin_Ha)
        # Sanity checks
        if elec_bands not in (1, 2):
            log.error("Model supports only 1 or 2 bands for now")
        if len(eff_mass) != elec_bands or len(band_offset) != elec_bands:
            log.error("Length of eff_mass and band_offset must match number of bands")
        # set parameters
        self.nbnd = elec_bands
        self.nkpt = nkpt
        self.nspin = 1     # spinless model for now
        self.L = 1.        # box size (Ang)
        # eff. masses in units hbar^2/(2*me) -> eV ps^2 / ang^2
        self.effective_masses = np.array(eff_mass, dtype=float)
        self.band_offset = np.array(band_offset, dtype=float)
        self.kgr = np.linspace(0.5, -0.5, nkpt) * 2.*np.pi / self.L
        self.enk = np.zeros((self.nbnd, self.nkpt, self.nspin))
        self.mu = chem_pot
        self.H0 = None
        # print info
        self.print_info(title="MODEL ELECTRONIC HAMILTONIAN")
    # --------------------------------------------------------
    #   Build H0(k)
    # --------------------------------------------------------
    def set_H0_matr(self):
        """
        Construct analytical H0(k) matrices.
        """
        d_enk = self.enk - self.mu
        self.H0 = []
        for ik in range(self.nkpt):
            Hk_list = []
            for ispin in range(self.nspin):
                # diagonal Hamiltonian
                H_np = np.diag(d_enk[:, ik, ispin])
                Hmat = PETSc.Mat().createDense(
                    size=(self.nbnd, self.nbnd),
                    array=H_np
                )
                Hmat.assemble()
                Hk_list.append(Hmat)
            self.H0.append(Hk_list)
    # --------------------------------------------------------
    #   spin unpolarized model
    # --------------------------------------------------------
    def set_energy_spectrum(self):
        """
        band energies enk[n, k, s]
        """
        # ħ² / 2mₑ in eV·Å²
        hbar2_2m = hbar ** 2 / (2*me)
        for ik in range(self.nkpt):
            k2 = np.dot(self.kgr[ik], self.kgr[ik])
            for ib in range(self.nbnd):
                self.enk[ib, ik, :] = (
                    self.band_offset[ib]
                    + hbar2_2m * k2 / self.effective_masses[ib]
                )