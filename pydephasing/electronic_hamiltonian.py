import numpy as np
import gc
from petsc4py import PETSc
from pydephasing.parallelization.mpi import mpi
from pydephasing.utilities.log import log
from pydephasing.set_param_object import p
from pydephasing.wannier_interface.wannier import Wannier
from pydephasing.common.phys_constants import hartree2ev
from pydephasing.utilities.plot_functions import plot_elec_struct

#
#   This module defines
#   the electronic Hamiltonian class
#

#
#   function : set electronic Hamiltonian
#

class electronic_hamiltonian:
    def __init__(self, Ewin_Ha, wann=False):
        # unpert. spectrum
        self.mu = None
        self.enk = None
        self.Vnk = None
        self.H0 = None
        self.WANNIER = wann
        self.Ewin_Ha = Ewin_Ha
        # H0 parameters
        self.nbnd = None
        self.nkpt = None
        self.nspin = None
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
    def plot_band_structure(self):
        Ew = np.array(self.Ewin_Ha) * hartree2ev
        # call exter. plot function
        if mpi.rank == mpi.root:
            plot_elec_struct(self.enk, self.mu, Ylim=Ew)
        mpi.comm.Barrier()
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
    def add_dipole_interact(self):
        pass
    # clean PETSc objects
    def clean_up(self):
        if self.H0 is not None:
            for k_list in self.H0:
                for mat in k_list:
                    if isinstance(mat, PETSc.Mat):
                        mat.destroy()
            self.H0 = None
        gc.collect()