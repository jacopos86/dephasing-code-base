# This routine extract the phonon data information
# and return it for processing
import numpy as np
import h5py
from pydephasing.input_parameters import p
from pydephasing.phys_constants import eps
from pydephasing.log import log
from pydephasing.mpi import mpi
import logging
#
def set_q_to_mq_list(qpts, nq):
    qvlist = []
    qplist = []
    nqp = 0
    for iq1 in range(nq):
        q1 = qpts[iq1]
        for iq2 in range(nq):
            q2 = qpts[iq2]
            if np.abs(q1[0]+q2[0]) < eps and np.abs(q1[1]+q2[1]) < eps and np.abs(q1[2]+q2[2]) < eps:
                if np.sqrt(q1[0]**2+q1[1]**2+q1[2]**2) > eps:
                    adding = True
                    for qp in qvlist:
                        if np.array_equal(np.array([q1, q2]), np.asarray(qp)) or np.array_equal(np.array([q2, q1]), np.asarray(qp)):
                            adding = False
                            break
                        else:
                            adding = True
                    if adding:
                        nqp += 2
                        qvlist.append([q1, q2])
                        qplist.append([iq1, iq2])
                else:
                    nqp += 1
                    qvlist.append([q1, q2])
                    qplist.append([iq1, iq2])
    assert nqp == nq
    return qplist
# check eigenv data
def check_eigenv_data(qpts, eigenv, nq):
    # check that e_mu,q = e_mu,-q^*
    qplist = set_q_to_mq_list(qpts, nq)
    for [iq, iqp] in qplist:
        euq = eigenv[iq]
        euqp= eigenv[iqp]
        # run over modes
        for il in range(euq.shape[1]):
            assert np.array_equal(euq[:,il], euqp[:,il].conjugate())
    if mpi.rank == mpi.root:
        log.info("eigenvector test passed")
def check_freq_data(qpts, freq, nq):
    # check w(u,q)=w(u,-q)
    qplist = set_q_to_mq_list(qpts, nq)
    for [iq, iqp] in qplist:
        wuq = freq[iq]
        wuqp= freq[iqp]
        assert np.array_equal(wuq, wuqp)
    if mpi.rank == mpi.root:
        log.info("freq. test passed")
#
def extract_ph_data():
    # input_params -> input data structure
    # q pt. index
    # open file
    with h5py.File(p.hd5_eigen_file, 'r') as f:
        # dict keys
        eig_key = list(f.keys())[0]
        # get the eigenvectors
        # Eigenvectors is a numpy array of three dimension.
        # The first index runs through q-points.
        # In the second and third indices, eigenvectors obtained
        # using numpy.linalg.eigh are stored.
        # The third index corresponds to the eigenvalue's index.
        # The second index is for atoms [x1, y1, z1, x2, y2, z2, ...].
        eigenv = list(f[eig_key])
        # get frequencies
        f_key = list(f.keys())[1]
        freq = list(f[f_key])
        # n. q pts.
        nq = len(freq)
        # q mesh
        m_key = list(f.keys())[2]
        mesh = list(f[m_key])
        # q pts.
        qpts_key = list(f.keys())[3]
        qpts = list(f[qpts_key])
        # q pts. weight
        wq_key = list(f.keys())[4]
        wq = list(f[wq_key])
        wq = np.array(wq, dtype=float)
        r = sum(wq)
        wq[:] = wq[:] / r
    if log.level <= logging.INFO:
        # check eigenv data
        check_eigenv_data(qpts, eigenv, nq)
        # check frequencies
        check_freq_data(qpts, freq, nq)
    # return data
    return eigenv, freq, nq, qpts, wq, mesh