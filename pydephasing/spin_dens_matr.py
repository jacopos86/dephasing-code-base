import numpy as np
from pydephasing.log import log
from pydephasing.mpi import mpi
from pydephasing.set_param_object import p

#
# spin density matrix class -> rho_{a,b}

class spin_dmatr(object):
    # initialization
    def __init__(self):
        # hilbert space dim.
        self.n = None
        self.matr0 = None
    def initialize(self, psi0):
        self.n = len(psi0)
        if mpi.rank == mpi.root:
            log.info("\t spin space dimension: " + str(self.n))
            log.info("\t " + p.sep)
        self.matr0 = np.kron(psi0, psi0.conj()).reshape((self.n, self.n))
        np.testing.assert_almost_equal(self.matr0.trace().real, 1.0)