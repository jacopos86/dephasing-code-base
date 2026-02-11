import numpy as np
from petsc4py import PETSc
from pydephasing.parallelization.mpi import mpi
from pydephasing.utilities.log import log

#
#   define electron density matrix
#

class elec_dmatr(object):
    # initialization
    def __init__(self, smearing, Te, Nel):
        self.nbnd = None
        self.nkpt = None
        self.nspin = None
        self.rho = None
        # set smearing
        self.smearing = smearing
        # elec. temperature
        self.Te = Te
        # temperature must be in eV
        self.nel = Nel
        # time arrays
        self.nsnap = None
        self.rho_t = None
        self.traces = None
    def initialize_dmatr(self, He, nel_tol=1.e-7):
        # set num. bands / k pts
        self.nbnd = He.nbnd
        self.nkpt = He.nkpt
        self.nspin = He.nspin
        # chem. pot (eV)
        mu = self.update_chem_pot(He)
        He.set_chem_pot(mu)
        # beta_e = 1/Te
        Te = self.Te
        beta = np.inf if Te == 0.0 else 1.0 / Te
        # define density matrix
        self.rho = []
        for ik in range(self.nkpt):
            for isp in range(self.nspin):
                # diagonal occupations
                focc = np.zeros(self.nbnd)
                for ib in range(self.nbnd):
                    Ek = He.enk[ib,ik,isp]
                    eps = Ek - mu
                    if self.smearing == 'FD':
                        if Te == 0.0:
                            focc[ib] = 1.0 if eps < 0.0 else 0.0
                        else:
                            focc[ib] = 1.0 / (np.exp(beta * eps) + 1.0)
                rho_np = np.diag(focc)
                rho_mat = PETSc.Mat().createDense(
                    size=(self.nbnd, self.nbnd),
                    array=rho_np
                )
                rho_mat.assemble()
                self.rho.append(rho_mat)
        # check electron number
        Nel = self.total_electrons_from_rho(He)
        assert abs(Nel - self.nel) < nel_tol
        if mpi.rank == mpi.root:
            log.info("\t Electronic density matrix initialized")
            log.info(f"\t Total electrons: {Nel.real:.6f}")
    def update_chem_pot(self, He, tol=1e-10, maxiter=200):
        """ Find chemical potential such that
        sum_k w_k sum_alpha f(E_{alpha k} - mu) = Nel """
        Te = self.Te
        beta = np.inf if Te == 0.0 else 1.0 / Te
        # bounds for chem. pot.
        emin = He.enk.min()
        emax = He.enk.max()
        mu_1 = emin - 0.5
        mu_2 = emax + 0.5
        # converge on chem. pot.
        for _ in range(maxiter):
            mu = 0.5 * (mu_1 + mu_2)
            Ntmp = self.total_electrons(He, mu)
            if abs(Ntmp - self.nel) < tol:
                return mu
            # update chem pot
            if Ntmp > self.nel:
                mu_2 = mu
            else:
                mu_1 = mu
        log.error("Chemical potential did not converge")
    def total_electrons(self, He, mu):
        """ Compute N = sum_k w_k Tr[rho(k)] """
        Te = self.Te
        beta = np.inf if Te == 0.0 else 1.0 / Te
        # compute N elec.
        N = 0.0
        for ik in range(self.nkpt):
            for isp in range(self.nspin):
                for ib in range(self.nbnd):
                    eps = He.enk[ib,ik,isp] - mu
                    if Te == 0.0:
                        f = 1.0 if eps < 0.0 else 0.0
                    else:
                        f = 1.0 / (np.exp(beta * eps) + 1.0)
                    N += He.wk[ik] * f
        return N
    def total_electrons_from_rho(self, He):
        """Compute total number of electrons from the density matrix."""
        if len(self.rho) == 0:
            log.error("Density matrix not initialized")
        # num. elec.
        iksp = 0
        N = 0.0
        for ik in range(self.nkpt):
            for isp in range(self.nspin):
                # rho[ik] is a PETSc Mat
                N += He.wk[ik] * self.rho[iksp].getDiagonal().sum()
                iksp += 1
        return N
    def get_PETSc_rhok(self, ik, isp):
        """ Return PETSc.Mat rho(k) for a given k-point and spin """
        if self.rho is None:
            log.error("Density matrix is not set")
        if ik < 0 or ik >= self.nkpt:
            log.error(f"k-point index {ik} out of bounds [0,{self.nkpt-1}]")
        if isp < 0 or isp >= self.nspin:
            log.error(f"spin index {isp} out of bounds [0,{self.nspin-1}]")
        iksp = ik * self.nspin + isp
        return self.rho[iksp]
    def init_td_arrays(self, nsteps, save_every):
        # Number of snapshots we will store
        self.nsnap = (nsteps + save_every - 1) // save_every + 1
        # Initialize storage array: k, spin, snapshot, nbnd, nbnd
        self.rho_t = np.zeros(
            (self.nkpt, self.nspin, self.nsnap, self.nbnd, self.nbnd), 
            dtype=np.complex128
        )
        self.traces = np.zeros((self.nkpt, self.nspin, self.nsnap))
    def store_rho_time(self, ik, isp, istep, rho_k):
        """ Store the density matrix rho_k at time step 'istep'"""
        self.rho_t[ik, isp, istep, :, :] = rho_k
    def store_traces(self, ik, isp, istep, traces):
        self.traces[ik, isp, istep] = traces
    def summary(self):
        """
        Print basic info about the electronic system
        """
        if mpi.rank == mpi.root:
            log.info("\t Electronic system info")
            log.info(f"\t Number bands: {self.nbnd}")
            log.info(f"\t Number k pts: {self.nkpt}")
            log.info(f"\t Number of spins: {self.nspin}")
            log.info(f"\t Number of electrons: {self.nel.real}")