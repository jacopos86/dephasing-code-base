import h5py
import logging
import numpy as np
from abc import ABC
from parallelization.mpi import mpi
from common.phys_constants import eps, kb, hbar
from pydephasing.set_param_object import p
from utilities.log import log
from math import exp
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
    def __init__(self, CELLMAP_FILE, EIGENV_FILE, OUT_FILE):
        super().__init__()
        self.CELLMAP_FILE = CELLMAP_FILE
        self.EIGENV_FILE = EIGENV_FILE
        self.OUT_FILE = OUT_FILE
        self.nCellsPh = None
        self.Hph = None
        self.supercell = None
    def read_ph_cell_map(self):
        cellMapPh = np.loadtxt(self.CELLMAP_FILE)[:,:3].astype(int)
        self.nCellsPh = cellMapPh.shape[0]
        print(self.nCellsPh)
    def get_ph_supercell(self):
        for line in open(self.OUT_FILE):
            tokens = line.split()
            if len(tokens) == 5:
                if tokens[0] == 'supercell' and tokens[4] == '\\':
                    self.supercell = np.array([int(token) for token in tokens[1:4]])
        if mpi.rank == mpi.root:
            log.info("\t ph. supercell grid: " + str(self.supercell))
            log.info("\t " + p.sep)
    def read_ph_hamilt(self):
        self.read_ph_cell_map()
        Hphonon = np.fromfile(self.EIGENV_FILE, dtype=np.float64)
        self.nmodes = int(np.sqrt(Hphonon.shape[0] / self.nCellsPh))
        if mpi.rank == mpi.root:
            log.info("\t " + p.sep)
            log.info("\t n. ph. modes: " + str(self.nmodes))
            log.info("\t " + p.sep)
        self.Hph = np.reshape(Hphonon, (self.nCellsPh,self.nmodes,self.nmodes)).swapaxes(1,2)
        # read ph. basis