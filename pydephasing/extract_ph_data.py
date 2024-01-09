# This routine extract the phonon data information
# and return it for processing
import numpy as np
import h5py
from pydephasing.input_parameters import p
from pydephasing.phys_constants import eps, THz_to_ev
from pydephasing.utility_functions import lorentzian, bose_occup
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
#
# set ql' list
def set_ql_list_red_qgrid(qpts, nat, wu):
    nq = len(qpts)
    # q -> -q map
    qmq_list = set_q_to_mq_list(qpts, nq)
    # only q>0 list
    qlp_list = []
    qmq_map = []
    for iqpair1 in qmq_list:
        iq1 = iqpair1[0]
        for il in range(3*nat):
            if wu[iq1][il] > p.min_freq:
                qlp_list.append((iq1,il))
        miq1 = iqpair1[1]
        qmq_map.append((iq1,miq1))
    # make dict q -> -q
    qmq_map = dict(qmq_map)
    # make local ql_list
    ql_list = mpi.split_list(qlp_list)
    return ql_list, qlp_list, qmq_map
#
# list of (l',q') pairs
def set_iqlp_list(il, iq, qlp_list_full, wu, H):
    T = np.max(p.temperatures)
    # dE
    dE = 0.0
    if p.relax:
        # quantum states
        iqs0 = p.index_qs0
        iqs1 = p.index_qs1
        dE = H.eig[iqs1] - H.eig[iqs0]
        # eV
    # set w_ql (eV)
    E_ql = wu[iq][il] * THz_to_ev
    nql = bose_occup(E_ql, T)
    # if freq. resolved calculation
    if p.w_resolved:
        ltz_max = 0.
        for iqp, ilp in qlp_list_full:
            if iqp != iq or ilp != il:
                E_qlp = wu[iqp][ilp] * THz_to_ev
                # eV
                x = dE + E_qlp - E_ql
                if lorentzian(x, p.eta) > ltz_max:
                    ltz_max = lorentzian(x, p.eta)
    # run over (q',l')
    qlp_list = []
    for iqp, ilp in qlp_list_full:
        if iqp != iq or ilp != il:
            E_qlp = wu[iqp][ilp] * THz_to_ev
            # E (eV)
            nqlp = bose_occup(E_qlp, T)
            A_th = nql * (1. + nqlp)
            if p.w_resolved:
                x = dE + E_qlp - E_ql
                if lorentzian(x, p.eta) / ltz_max > p.lorentz_thres and A_th > 1.E-7:
                    #print(dE, E_ql, E_qlp, x, lorentzian(x, p.eta))
                    qlp_list.append((iqp,ilp))
            else:
                if A_th > 1.E-7:
                    qlp_list.append((iqp,ilp))
    return qlp_list
#
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
        log.info("\n")
        log.info("\t " + p.sep)
        log.info("\t EIGENVECTOR TEST    ->    PASSED")
        log.info("\t " + p.sep)
        log.info("\n")
#
# check frequencies
def check_freq_data(qpts, freq, nq):
    # check w(u,q)=w(u,-q)
    qplist = set_q_to_mq_list(qpts, nq)
    for [iq, iqp] in qplist:
        wuq = freq[iq]
        wuqp= freq[iqp]
        assert np.array_equal(wuq, wuqp)
    if mpi.rank == mpi.root:
        log.info("\n")
        log.info("\t " + p.sep)
        log.info("\t FREQUENCY TEST    ->    PASSED")
        log.info("\t " + p.sep)
        log.info("\n")
# check weights
def check_weights(qpts, wq, nq):
    # check w(q) = w(-q)
    qplist = set_q_to_mq_list(qpts, nq)
    for [iq, iqp] in qplist:
        assert wq[iq] == wq[iqp]
    if mpi.rank == mpi.root:
        log.info("\n")
        log.info("\t " + p.sep)
        log.info("\t W(Q) TEST    ->    PASSED")
        log.info("\t " + p.sep)
        log.info("\n")
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
        # check the weights
        check_weights(qpts, wq, nq)
    # return data
    return eigenv, freq, nq, qpts, wq, mesh