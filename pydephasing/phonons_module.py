import h5py
import logging
import numpy as np
from pydephasing.mpi import mpi
from common.phys_constants import eps, kb
from pydephasing.set_param_object import p
from pydephasing.log import log
from math import exp
#
#  phonons class
#
class PhononsClass:
    def __init__(self):
        self.eq_key = ''
        self.eql = None
        #  phonon eigenvectors
        self.uq_key = ''
        self.uql = None
        #  phonon eigenvalues
    #
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
    #
    #  function -> phonon occup.
    #
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
    #   set phonon energies
    def set_ph_data(self, qgr):
        self.get_phonon_keys()
        # open file
        with h5py.File(p.hd5_eigen_file, 'r') as f:
            # ph. frequency
            self.uql = list(f[self.uq_key])
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
    #
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