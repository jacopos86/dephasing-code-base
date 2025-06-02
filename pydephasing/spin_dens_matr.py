import numpy as np
from pydephasing.log import log
from pydephasing.mpi import mpi
from pydephasing.set_param_object import p
from common.print_objects import print_2D_matrix

#
# spin density matrix class -> rho_{a,b}

class spin_dmatr(object):
    # initialization
    def __init__(self):
        # hilbert space dim.
        self.n = None
        self.matr0 = None
        self.matr = None
        self.time = None
    def initialize(self, psi0):
        self.n = len(psi0)
        if mpi.rank == mpi.root:
            log.info("\t spin space dimension: " + str(self.n))
            log.info("\t " + p.sep)
        self.matr0 = np.kron(psi0, psi0.conj()).reshape((self.n, self.n))
        np.testing.assert_almost_equal(self.matr0.trace().real, 1.0)
        if mpi.rank == mpi.root:
            log.info("\n")
            log.info("\t " + p.sep)
            log.info("\t PRINT DENSITY MATRIX(t=0)")
            print_2D_matrix(self.matr0)
    def set_density_matr(self, time, rh):
        self.matr = rh
        self.time = time
        assert(self.matr.shape[0] == self.n)
        assert(self.matr.shape[1] == self.n)
        assert(self.matr.shape[2] == self.time.shape[0])
    def trace_oft(self):
        nt = len(self.time)