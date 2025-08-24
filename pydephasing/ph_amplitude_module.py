#
#  This module defines the atomic displacement
#  needed to compute the ACF
#
import numpy as np
from mpmath import coth
from cmath import exp
from pydephasing.set_param_object import p
from pydephasing.atomic_list_struct import atoms
from common.phys_constants import hbar, THz_to_ev, mp, kb
from parallelization.mpi import mpi
import yaml
#
class PhononAmplitude:
    # initialization
    def __init__(self, nat):
        # set atomic ph. amplitude
        self.nph = 3*nat
        self.u_ja_t = np.zeros((p.ntmp,3*nat,p.nt))
        if p.ph_resolved:
            self.u_jal_t = np.zeros((p.ntmp,3*nat,p.nphr,p.nt2))
        #
    def compute_ph_amplq(self, euq, wuq, nat):
        self.Q_ql_ja = np.zeros((self.nph,3*nat,p.ntmp))
        #
        # run over ph modes
        #
        for im in range(self.nph):
            if wuq[im] > p.min_freq:
                for jax in range(3*nat):
                    # atom index in list
                    ia = atoms.index_to_ia_map[jax] - 1
                    m_ia = atoms.atoms_dict[ia]['mass']
                    m_ia = m_ia * mp
                    #
                    # eV ps^2 / ang^2 units
                    #
                    e_uq_ja = np.sqrt((euq[jax,im] * np.conjugate(euq[jax,im])).real)
                    # T displ.
                    for iT in range(p.ntmp):
                        T = p.temperatures[iT]
                        # K units
                        E = wuq[im] * THz_to_ev    # eV
                        nph = bose_occup(E, T)
                        #
                        # phonon amplitude
                        # A_lam = [(1+2n_ph)*hbar/(2*N*m_a*w_lam)]^1/2 eps(lam,ja)
                        A_q = np.sqrt(hbar*(1+2*nph)/(4.*np.pi*wuq[im]*nat))
                        self.Q_ql_ja[im,jax,iT] = A_q * e_uq_ja / np.sqrt(m_ia)
                        # ang units
    #
    # compute classical ph. ampl.
    #
    def compute_classical_ph_amplq(self, euq, wuq, nat):
        self.Q_ql_ja = np.zeros((self.nph,3*nat,p.ntmp))
        # run over ph. mode
        for im in range(self.nph):
            if wuq[im] > p.min_freq:
                for jax in range(3*nat):
                    # atom index
                    ia = atoms.index_to_ia_map[jax] - 1
                    m_ia = atoms.atoms_dict[ia]['mass']
                    m_ia = m_ia * mp
                    #
                    # eV ps^2 / ang^2
                    #
                    e_uq_ja = np.sqrt((euq[jax,im] * np.conjugate(euq[jax,im])).real)
                    # displ
                    for iT in range(p.ntmp):
                        T = p.temperatures[iT]
                        # K units
                        E = wuq[im] * THz_to_ev
                        ctgh = coth(E/(2.*kb*T))
                        A_q = np.sqrt(hbar*ctgh/(4.*np.pi*wuq[im]*nat))
                        self.Q_ql_ja[im,jax,iT] = A_q * e_uq_ja / np.sqrt(m_ia)
                        # ang units
    # compute phase factor : exp(-iqRa)
    def set_phase_factor(self, qv, nat):
        exp_iqR = np.zeros(3*nat, dtype=np.complex128)
        for jax in range(3*nat):
            ia = atoms.index_to_ia_map[jax] - 1
            # direct atom coordinate
            Ra = atoms.atoms_dict[ia]['coordinates']
            exp_iqR[jax] = exp(-1j*2.*np.pi*np.dot(qv, Ra))
        return exp_iqR
    #
    # compute phonon modulation
    #
    def compute_atom_dipl_q(self, qv, wuq, nat):
        uq_ja = np.zeros((p.ntmp,3*nat,p.nt), dtype=np.complex128)
        # compute phase factor
        exp_iqR = self.set_phase_factor(qv, nat)
        # cycle over temperatures
        for iT in range(p.ntmp):
            # run over phonon modes
            for im in range(self.nph):
                if wuq[im] > p.min_freq:
                    exp_iwt = np.zeros(p.nt, dtype=np.complex128)
                    for t in range(p.nt):
                        exp_iwt[t] = exp(1j*2.*np.pi*wuq[im]*p.time[t])
                    # run over atom index 
                    for jax in range(3*nat):
                        # compute temporal fluctuation
                        uq_ja[iT,jax,:] += self.Q_ql_ja[im,jax,iT] * exp_iwt[:] * exp_iqR[jax]
        return uq_ja.real
    # phonon resolved displacement
    def ph_res_ampl(self, nat, iphr, u, wu, nq, qpts, wq):
        u_jal_t = np.zeros((p.ntmp,3*nat,p.nt2), dtype=np.complex128)
        # run over qv
        for iq in range(nq):
            # q vect.
            qv = qpts[iq]
            # phase factor
            exp_iqR = self.set_phase_factor(qv, nat)
            # freq / modes
            wuq= wu[iq]
            if wuq[iphr] > p.min_freq:
                euq= u[iq]
                # e^iwt
                exp_iwt = np.zeros(p.nt2, dtype=np.complex128)
                for t in range(p.nt2):
                    exp_iwt[t] = exp(1j*2.*np.pi*wuq[iphr]*p.time2[t])
                A_qlja = np.zeros((p.ntmp,3*nat), dtype=np.complex128)
                # jax cycle
                for jax in range(3*nat):
                    e_uqja = np.sqrt((euq[jax,iphr] * np.conjugate(euq[jax,iphr])).real)
                    # atom mass
                    ia = atoms.index_to_ia_map[jax] - 1
                    m_ia = atoms.atoms_dict[ia]['mass']
                    m_ia = m_ia * mp
                    # run over temperatures
                    for iT in range(p.ntmp):
                        T = p.temperatures[iT]
                        E = wuq[iphr] * THz_to_ev
                        nph = bose_occup(E, T)
                        # ph. ampl.
                        A = np.sqrt(hbar*(1.+2.*nph)/(4.*np.pi*wuq[iphr]*nat))
                        # ang units
                        A_qlja[iT,jax] = A * e_uqja * exp_iqR[jax] / np.sqrt(m_ia)
                u_jal_t[iT,jax,:] += wq[iq] * A_qlja[iT,jax] * exp_iwt[:]
        return u_jal_t[:,:,:].real
    #
    # update atomic displacement
    #
    def update_atom_displ(self, wq, qv, wuq, nat):
        uq_ja = self.compute_atom_dipl_q(qv, wuq, nat)
        self.u_ja_t[:,:,:] += wq * uq_ja[:,:,:]
    #
    # collect atom displacement between processors
    def collect_displ_between_proc(self, nat):
        for iT in range(p.ntmp):
            for jax in range(3*nat):
                f_oft = np.zeros(p.nt)
                f_oft = self.u_ja_t[iT,jax,:]
                ff_oft = mpi.collect_time_array(f_oft)
                self.u_ja_t[iT,jax,:] = 0.
                self.u_ja_t[iT,jax,:] = ff_oft[:]
    # print atomic displ. on file
    def print_atom_displ(self, out_dir):
	    # write tensor to file
        file_name = "atom_displ.yml"
        file_name = "{}".format(out_dir + '/' + file_name)
        data = {'u_ja' : self.u_ja_t, 'u_jal' : self.u_jal_t}
        # ang units
        with open(file_name, 'w') as out_file:
            yaml.dump(data, out_file)