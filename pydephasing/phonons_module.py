import h5py
import logging
import numpy as np
from abc import ABC
from math import exp
from pydephasing.parallelization.mpi import mpi
from pydephasing.common.phys_constants import eps, kb, hbar
from pydephasing.set_param_object import p
from pydephasing.utilities.log import log
from pydephasing.common.phys_constants import hartree2ev, THz_to_ev
from pydephasing.utilities.plot_functions import plot_ph_band_struct, plot_lph_struct
from pydephasing.atomic_list_struct import atoms

#
#  phonons class
#

class PhononsClass(ABC):
    def __init__(self):
        self.eql = None
        #  phonon eigenvectors
        self.uql = None
        #  phonon eigenvalues
        self.nmodes = None
        # n. of ph. modes
    #  function -> phonon occup.
    def ph_occup(self, E, T):
        if T < eps:
            n = 0.
        else:
            x = E / (kb*T)
            if x > 100.:
                n = 0.
            else:
                n = 1./(exp(E / (kb * T)) - 1.)
        return n

#
#   phonopy phonons
#

class PhonopyPhonons(PhononsClass):
    def __init__(self):
        super().__init__()
        self.eq_key = ''
        self.uq_key = ''
    # phonons amplitudes
    def compute_ph_amplitude_q(self, nat, ql_list):
        # A_lq = [hbar/(2*N*w_lq)]^1/2
        # at a given q vector
        # [eV^1/2 ps]
        A_ql = np.zeros(len(ql_list))
        # run over ph. modes
        # run over local (q,l) list
        iql = 0
        for iq, il in ql_list:
            # freq.
            wuq = self.uql[iq]
            # amplitude
            if wuq[il] > p.min_freq:
                A_ql[iql] = np.sqrt(hbar / (4.*np.pi*wuq[il]*nat))
                # eV^0.5*ps
            iql += 1
        return A_ql
    #  get phonon keys
    def get_phonon_keys(self):
        # open file
        with h5py.File(p.hd5_eigen_file, 'r') as f:
            # dict keys
            keys = list(f.keys())
            for k in keys:
                if k == "eigenvector" or k == "Eigenvector" or k == "modes" or k == "Modes":
                    self.eq_key = k
            if self.eq_key == '':
                log.error("phonon eigenvector key not found")
            for k in keys:
                if k == "frequency" or k == "Frequency":
                    self.uq_key = k
            if self.uq_key == '':
                log.error("phonon frequency key not found")
    #   set phonon energies
    def set_ph_data(self, qgr):
        self.get_phonon_keys()
        # open file
        with h5py.File(p.hd5_eigen_file, 'r') as f:
            # ph. frequency
            self.uql = list(f[self.uq_key])
            self.nmodes = len(self.uql[0])
            if mpi.rank == mpi.root:
                log.info("\t number phonon modes: " + str(self.nmodes))
            # ph. eigenvectors
            # Eigenvectors is a numpy array of three dimension.
            # The first index runs through q-points.
            # In the second and third indices, eigenvectors obtained
            # using numpy.linalg.eigh are stored.
            # The third index corresponds to the eigenvalue's index.
            # The second index is for atoms [x1, y1, z1, x2, y2, z2, ...].
            self.eql = list(f[self.eq_key])
        if log.level <= logging.INFO:
            self.check_eq_data(qgr)
            self.check_uq_data(qgr)
    #
    # phonons amplitudes
    def compute_ph_amplitude_q(self, ql_list):
        # A_lq = [hbar/(2*w_lq)]^1/2
        # at a given q vector
        # [eV^1/2 ps]
        A_ql = np.zeros(len(ql_list))
        # run over ph. modes
        # run over local (q,l) list
        iql = 0
        for iq, il in ql_list:
            # freq.
            wuq = self.uql[iq]
            # amplitude
            if wuq[il] > p.min_freq:
                A_ql[iql] = np.sqrt(hbar / (4.*np.pi*wuq[il]))
                # eV^0.5*ps
            iql += 1
        return A_ql
    #
    # check eigenv data
    def check_eq_data(self, qgr):
        # check that e_mu,q = e_mu,-q^*
        qplist = qgr.set_q2mq_list()
        for [iq, iqp] in qplist:
            eq = self.eql[iq]
            eqp= self.eql[iqp]
            # run over modes
            for il in range(eq.shape[1]):
                assert np.array_equal(eq[:,il], eqp[:,il].conjugate())
        if mpi.rank == mpi.root:
            log.info("\n")
            log.info("\t " + p.sep)
            log.info("\t EIGENVECTOR TEST    ->    PASSED")
            log.info("\t " + p.sep)
            log.info("\n")
    # check frequencies
    def check_uq_data(self, qgr):
        # check u(mu,q)=u(mu,-q)
        qplist = qgr.set_q2mq_list()
        for [iq, iqp] in qplist:
            wuq = self.uql[iq]
            wuqp= self.uql[iqp]
            assert np.array_equal(wuq, wuqp)
        if mpi.rank == mpi.root:
            log.info("\n")
            log.info("\t " + p.sep)
            log.info("\t FREQUENCY TEST    ->    PASSED")
            log.info("\t " + p.sep)
            log.info("\n")

#
#   JDFTx phonons
#

class JDFTxPhonons(PhononsClass):
    def __init__(self, CELLMAP_FILE, EIGENV_FILE, OUT_FILE, PHBASIS_FILE, TR_SYM=True):
        super().__init__()
        self.cellMapPh = None
        self.nCellsPh = None
        self.phBasis = None
        self.Hph = None
        self.ForceMatrix = None
        self.supercell = None
        self.lph_eq = None
        self.Mph = None
        # input files
        self.CELLMAP_FILE = CELLMAP_FILE
        self.EIGENV_FILE = EIGENV_FILE
        self.OUT_FILE = OUT_FILE
        self.PHBASIS_FILE = PHBASIS_FILE
        self.TR_SYM = TR_SYM
    def read_ph_cell_map(self):
        self.cellMapPh = np.loadtxt(self.CELLMAP_FILE)[:,:3].astype(int)
        self.nCellsPh = self.cellMapPh.shape[0]
    def read_force_matrix(self):
        self.read_ph_cell_map()
        Forces = np.fromfile(self.EIGENV_FILE, dtype=np.float64)
        # n. modes
        self.nmodes = int(np.sqrt(Forces.shape[0] / self.nCellsPh))
        if mpi.rank == mpi.root:
            log.info("\t " + p.sep)
            log.info("\t n. ph. modes: " + str(self.nmodes))
            log.info("\t " + p.sep)
        self.ForceMatrix = np.reshape(Forces, (self.nCellsPh,self.nmodes,self.nmodes))
    def get_ph_supercell(self):
        for line in open(self.OUT_FILE):
            tokens = line.split()
            if len(tokens) == 5:
                if tokens[0] == 'supercell' and tokens[4] == '\\':
                    self.supercell = np.array([int(token) for token in tokens[1:4]])
        if mpi.rank == mpi.root:
            log.info("\t ph. supercell grid: " + str(self.supercell))
            log.info("\t " + p.sep)
    def read_phonon_basis(self):
        self.phBasis = np.loadtxt(self.PHBASIS_FILE, usecols=[2,3,4])
        self.phBasis = np.sqrt(np.sum(self.phBasis ** 2, axis=1))
    def compute_energy_dispersion(self, qgr, n_interp=10):
        # set q grid
        qp, n = qgr.set_qgr_plot(n_interp)
        if mpi.rank == mpi.root:
            log.info("\t COMPUTE PHONON DISPERSION")
            log.info("\t " + p.sep)
        TildeForceMatrix = np.tensordot(np.exp((2j*np.pi)*np.dot(qp, self.cellMapPh.T)), self.ForceMatrix, axes=1)
        # diagonalization
        omegaSq, normalModes = np.linalg.eigh(TildeForceMatrix)   # energies in Hartree
        omega = np.copysign(np.sqrt(np.abs(omegaSq)), omegaSq)
        if mpi.rank == mpi.root:
            log.info("\t shape wq^2: " + str(omega.shape))
            log.info("\t " + p.sep)
            plot_ph_band_struct(omega * hartree2ev * 1.E3, n)  # energies in meV
        mpi.comm.Barrier()
    def compute_ph_state_q(self, q):
        phase = np.exp((2j*np.pi)*np.dot(q, self.cellMapPh.T))
        D = np.tensordot(phase, self.ForceMatrix, axes=1)
        omegaSq, Wq = np.linalg.eigh(D)
        omega = np.copysign(np.sqrt(np.abs(omegaSq)), omegaSq)
        omega_q = omega * hartree2ev / THz_to_ev        # THz
        return omega_q, Wq
    def compute_eq_ph_angular_momentum_dispersion(self, qgr, n_interp=10):
        # set q grid
        qp, n = qgr.set_qgr_plot(n_interp)
        self.lph_eq = np.zeros((3,self.nmodes,n), dtype=np.complex128)
        if mpi.rank == mpi.root:
            log.info("\t TR SYMMETRY: " + str(self.TR_SYM))
            log.info("\t " + p.sep)
        # compute lph at each q
        wq_list = []
        for i, q in enumerate(qp):
            qm = -q
            omega_q, Wq = self.compute_ph_state_q(q)
            omega_qm, Wqm = self.compute_ph_state_q(qm)
            if self.TR_SYM:
                if not np.allclose(omega_q, omega_qm):
                    log.error(f"Energy mismatch detected between q = {q} and q' = {qm}")
                for l in range(self.nmodes):
                    if not np.allclose(Wq[:,l], np.conj(Wqm[:,l])):
                        log.error(f"Time-reversal symmetry violation detected between q = {q} and q' = {qm}")
            for l in range(self.nmodes):
                Wql = Wq[:,l].reshape(-1,3)
                Wqml = Wqm[:,l].reshape(-1,3)
                for k in range(atoms.nat):
                    self.lph_eq[:,l,i] += -1j*np.cross(Wql[k,:], Wqml[k,:])
            wq_list.append(omega_q)
        wq_list = np.array(wq_list)
        if mpi.rank == mpi.root:
            plot_lph_struct(wq_list * THz_to_ev * 1.E3, self.lph_eq, n, n_interp=n_interp)
        mpi.comm.Barrier()
    def compute_full_ph_angular_momentum_matrix(self, qgr):
        self.Mph = np.zeros((3,self.nmodes,self.nmodes,qgr.nq), dtype=np.complex128)
        # compute at each q
        for i, q in enumerate(qgr.qpts):
            qm = -q
            omega_q, Wq = self.compute_ph_state_q(q)
            omega_qm, Wqm = self.compute_ph_state_q(qm)
            if self.TR_SYM:
                if not np.allclose(omega_q, omega_qm):
                    log.error(f"Energy mismatch detected between q = {q} and q' = {qm}")
                for l in range(self.nmodes):
                    if not np.allclose(Wq[:,l], np.conj(Wqm[:,l])):
                        log.error(f"Time-reversal symmetry violation detected between q = {q} and q' = {qm}")
            for l1 in range(self.nmodes):
                # in THz
                if omega_q[l1] > p.min_freq:
                    Wql1 = Wq[:,l1].reshape(-1,3)
                    for l2 in range(self.nmodes):
                        Wqml2 = Wqm[:,l2].reshape(-1,3)
                        for k in range(atoms.nat):
                            self.Mph[:,l1,l2,i] += -1j*np.sqrt(omega_qm[l2] / omega_q[l1])*np.cross(Wql1[k,:], Wqml2[k,:])
    def read_ph_hamilt(self, qgr):
        # read ph basis
        self.read_phonon_basis()
        # read force matrix from file
        self.read_force_matrix()
        # read K points
        self.compute_energy_dispersion(qgr)