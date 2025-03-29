import h5py
import math
import numpy as np
import logging
from pydephasing.mpi import mpi
from pydephasing.set_param_object import p
from pydephasing.log import log
from pydephasing.phys_constants import eps
#
#   q grid class
#
class qgridClass:
    def __init__(self):
        self.mesh_key = ''
        self.grid_size = None
        self.nq = None
        # grid size
        self.qpts_key = ''
        self.qpts = None
        # q points list
        self.wq_key = ''
        self.wq = None
        # q weight
    def get_grid_keys(self):
        # open file
        with h5py.File(p.hd5_eigen_file, 'r') as f:
            # dict keys
            keys = list(f.keys())
            for k in keys:
                if k == "mesh" or k == "Mesh" or k == "MESH":
                    self.mesh_key = k
            if self.mesh_key == '':
                log.error("mesh key not found")
            for k in keys:
                if k == "qpoint" or k == "qpoints":
                    self.qpts_key = k
            if self.qpts_key == '':
                log.error("qpts key not found")
            for k in keys:
                if k == "weight" or k == "Weight":
                    self.wq_key = k
            if self.wq_key == '':
                log.error("wq key not found")
    # q grid
    def set_qgrid(self):
        self.get_grid_keys()
        # open file
        with h5py.File(p.hd5_eigen_file, 'r') as f:
            # dict keys -> size
            self.grid_size = list(f[self.mesh_key])
            self.nq = math.prod(self.grid_size)
            if mpi.rank == mpi.root:
                log.info("\t q grid size: " + str(self.nq))
            # dict keys -> qpts
            self.qpts = list(f[self.qpts_key])
            # dict keys -> wq
            self.wq = list(f[self.wq_key])
            self.wq = np.array(self.wq, dtype=float)
            r = sum(self.wq)
            self.wq[:] = self.wq[:] / r
        assert len(self.qpts) == self.nq
        if log.level <= logging.INFO:
            self.check_weights()
    #
    #  this method creates a list of pairs
    #  (q, -q)
    #
    def set_q2mq_list(self):
        qvlist = []
        qplist = []
        nqp = 0
        for iq1 in range(self.nq):
            q1 = self.qpts[iq1]
            for iq2 in range(self.nq):
                q2 = self.qpts[iq2]
                adding = False
                if np.abs(q1[0]+q2[0]) < eps and np.abs(q1[1]+q2[1]) < eps and np.abs(q1[2]+q2[2]) < eps:
                    if np.sqrt(q1[0]**2+q1[1]**2+q1[2]**2) > eps:
                        adding = True
                        for qp in qvlist:
                            if np.array_equal(np.array([q1, q2]), np.asarray(qp)) or np.array_equal(np.array([q2, q1]), np.asarray(qp)):
                                adding = False
                        if adding:
                            nqp += 2
                            qvlist.append([q1, q2])
                            qplist.append([iq1, iq2])
                    else:
                        nqp += 1
                        qvlist.append([q1, q2])
                        qplist.append([iq1, iq2])
        assert nqp == self.nq
        return qplist
    #
    # check weights
    def check_weights(self):
        # check w(q) = w(-q)
        qplist = self.set_q2mq_list()
        for [iq, iqp] in qplist:
            assert self.wq[iq] == self.wq[iqp]
        if mpi.rank == mpi.root:
            log.info("\n")
            log.info("\t " + p.sep)
            log.info("\t W(Q) TEST    ->    PASSED")
            log.info("\t " + p.sep)
            log.info("\n")