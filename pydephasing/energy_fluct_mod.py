#  This module defines the auto correlation
#  functions to be used in the different
#  calculations
#
import numpy as np
import cmath
from parallelization.mpi import mpi
from pydephasing.log import log
from pydephasing.set_param_object import p
from common.phys_constants import THz_to_ev, mp
from pydephasing.atomic_list_struct import atoms
#
# energy levels fluctuations
class energy_level_fluctuations_oft:
    # initialization
    def __init__(self):
        self.deltaE_oft = np.zeros((p.nt,p.ntmp))
    # compute energy fluctuations
    def compute_deltaE_oft(self, wq, wu, ql_list, A_lq, F_lq):
        #
        # compute deltaE(t)
        #
        iql = 0
        # iterate over ph. modes
        for iq, il in ql_list:
            if mpi.rank == mpi.root:
                log.debug(str(il) + ',' + str(iq) + ' -> ' + str(len(ql_list)))
            wuq = wu[iq]
            if wuq[il] > p.min_freq:
                # Eql
                Eql = wuq[il] * THz_to_ev
                wql = 2.*np.pi*wuq[il]
                # time part
                eiwt = np.zeros(p.nt, dtype=np.complex128)
                eiwt[:] = np.exp(1j*wql*p.time[:])
                # run over the temperature list
                for iT in range(p.ntmp):
                    T = p.temperatures[iT]
                    # bose occup.
                    nph = bose_occup(Eql, T)
                    ft = np.zeros(p.nt, dtype=np.complex128)
                    ft[:] = (1.+nph) * eiwt[:] + nph * eiwt[:].conjugate()
                    # compute energy fluct.
                    self.deltaE_oft[:,iT] += wq[iq] * A_lq[iql] ** 2 * ft[:] * F_lq[iql] * F_lq[iql].conjugate()
                    # eV units
                    #
            iql += 1
#
# static spin fluctuations class
class ZFS_ph_fluctuations:
    # expression 2nd order ZFS energy
    # fluctuations
    # dV = \sum_lambda \sum_q w(q) A_lambda(q) A_lambda(-q) (1+2n_lambda(q)) Delta F_lambda,q;lambda,-q
    def __init__(self):
        self.dw_zfs = None
        # eV units
    # set (q,-q,l) transitions
    # list
    def set_qqp_list(self, qpts):
        nq = len(qpts)
        # q -> -q map
        qqp_list = set_q_to_mq_list(qpts, nq)
        # split list between procs.
        qqp_list = mpi.split_list(qqp_list)
        return qqp_list
    # compute eff. force
    # only Delta F(2) -> Delta F^(R) = 0
    # see notes
    def transf_2nd_order_force_phr(self, u, nat, qpts, Faxby, qqp_list):
        F_lqqp = []
        # run over (q,q',l)
        for iqql in qqp_list:
            # q
            iq = iqql[0]
            q = qpts[iq]
            euq = u[iq]
            # q'
            iqp= iqql[1]
            qp = qpts[iqp]
            euqp = u[iqp]
            # set e^iqR / e^iqpR
            eiqR = np.zeros(3*nat, dtype=np.complex128)
            eiqpR= np.zeros(3*nat, dtype=np.complex128)
            for jax in range(3*nat):
                ia = atoms.index_to_ia_map[jax]-1
                # atom coordinate
                Ra = atoms.atoms_dict[ia]['coordinates']
                eiqR[jax] = cmath.exp(1j*2.*np.pi*np.dot(q,Ra))
                eiqpR[jax]= cmath.exp(1j*2.*np.pi*np.dot(qp,Ra))
            # eu/M**0.5
            euq_sqrtM = np.zeros(euq.shape, dtype=np.complex128)
            euqp_sqrtM= np.zeros(euqp.shape, dtype=np.complex128)
            for jax in range(3*nat):
                ia = atoms.index_to_ia_map[jax]-1
                # atom mass
                m_ia = atoms.atoms_dict[ia]['mass']
                m_ia = m_ia * mp
                euq_sqrtM[jax,:] = euq[jax,:] / np.sqrt(m_ia) * eiqR[jax]
                euqp_sqrtM[jax,:]= euqp[jax,:] / np.sqrt(m_ia) * eiqpR[jax]
            # A^2 / eV / ps^2 * eV / A^2 = ps^-2
            for il in range(3*nat):
                f_lqlqp = np.einsum("i,ij,j->", euq_sqrtM[:,il], Faxby[:,:], euqp_sqrtM[:,il])
                F_lqqp.append(f_lqlqp)
        return F_lqqp
    # compute energy 
    # fluctuations
    def compute_DW_factor(self, ph, spph):
        # first set transitions list
        # here compute :
        # Delta_{a,b} = \sum_q g_{a,b}^{qp-}(1 + 2n(q))
        qpl_list = spph.local_qpl_list()
        # compute eff. forces -> different for each rank
        g_qpl = spph.compute_gqqp(qpl_list)
        nbnd = spph.space_size()
        self.dw_zfs = np.zeros((nbnd, nbnd, p.ntmp))
        # sum over (q,p,l)
        for iqpl in range(len(qpl_list)):
            iq = qpl_list[0]
            ip = qpl_list[1]
            il = qpl_list[2]
            assert np.abs(ph.uql[iq][il]-ph.uql[ip][il]) < 1.E-4
            # -> wql
            wql = ph.uql[iq][il] * THz_to_ev
            # bands
            for i1 in range(nbnd):
                for i2 in range(nbnd):
                    # run over temperatures
                    for iT in range(p.ntmp):
                        # compute n. phonons
                        T = p.temperatures[iT]
                        nq0 = ph.ph_occup(wql, T)
                        # compute DW energy correction
                        self.dw_zfs[i1,i2,iT] += g_qpl[i1,i2,iqpl] * (1.+2.*nq0)
                        #### eV
        self.dw_zfs[:,:,:] = self.dw_zfs[:,:,:].real
    # collect data
    def collect_acf_from_processes(self):
        for iT in range(p.ntmp):
            dE_list = mpi.comm.gather(self.dE_sp[iT], root=mpi.root)
            dE = 0.
            if mpi.rank == mpi.root:
                for x in dE_list:
                    dE += x
            self.dE_eV[iT] = mpi.comm.bcast(dE, root=mpi.root)