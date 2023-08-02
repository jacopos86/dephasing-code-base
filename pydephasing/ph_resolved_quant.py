import numpy as np
import cmath
from pydephasing.phys_constants import hbar, mp, THz_to_ev
from pydephasing.atomic_list_struct import atoms
from pydephasing.input_parameters import p
from pydephasing.log import log
#
def compute_ph_amplitude_q(wu, nat, ql_list):
    # A_lq = [hbar/(2*N*w_lq)]^1/2
    # at a given q vector
    # [eV^1/2 ps]
    A_lq = np.zeros(len(ql_list))
    # run over ph. modes
    # run over local (q,l) list
    iql = 0
    for iq, il in ql_list:
        # freq.
        wuq = wu[iq]
        # amplitude
        if wuq[il] > p.min_freq:
            A_lq[iql] = np.sqrt(hbar / (4.*np.pi*wuq[il]*nat))
        # eV^0.5*ps
        iql += 1
    #
    return A_lq
#
# set ZFS gradient (lambda,q)
def transf_1st_order_force_phr(u, qpts, nat, Fax, ql_list):
    # only relax calculation
    if p.deph:
        return
    iqs0 = p.index_qs0
    iqs1 = p.index_qs1
    # eff_Fax units : [eV/ang]
    F_lq = np.zeros((3*nat,len(ql_list)), dtype=np.complex128)
    eff_Fax = np.zeros(3*nat, dtype=np.complex128)
    eff_Fax[:] = Fax[iqs0,iqs1,:]
    # run over ph. modes
    for jax in range(3*nat):
        ia = atoms.index_to_ia_map[jax] - 1
        # atom coordinates
        Ra = atoms.atoms_dict[ia]['coordinates']
        # atomic mass
        m_ia = atoms.atoms_dict[ia]['mass']
        m_ia = m_ia * mp
        # eV ps^2 / ang^2
        F_ax = eff_Fax[jax] / np.sqrt(m_ia)
        # (q,l) list
        iql = 0
        for iq, il in ql_list:
            # e^iqR
            qv = qpts[iq]
            eiqR = cmath.exp(1j*2.*np.pi*np.dot(qv,Ra))
            # u(q,l)
            euq = u[iq]
            F_lq[jax,iql] = euq[jax,il] * eiqR * F_ax
            # [eV/ang * ang/eV^1/2 *ps^-1 = eV^1/2 / ps]
            # update iter
            iql += 1
    return F_lq
#
# set ZFS force at 2nd order
def transf_2nd_order_force_phr(il, iq, wu, u, qpts, nat, Fax, Faxby, qlp_list, H):
    # Fax units -> eV / ang
    # Faxby units -> eV / ang^2
    F_lq_lqp = np.zeros((3*nat, len(qlp_list)), dtype=np.complex128)
    # wql (eV)
    wql = wu[iq][il] * THz_to_ev
    # remember index jax -> atom,idx
    # second index (n,p)
    # q vector
    qv = qpts[iq]
    # eq
    euq = u[iq]
    # set e^iqR
    eiqR = np.zeros(3*nat, dtype=np.complex128)
    for jax in range(3*nat):
        ia = atoms.index_to_ia_map[jax] - 1
        # atom coordinate
        Ra = atoms.atoms_dict[ia]['coordinates']
        eiqR[jax] = cmath.exp(1j*2.*np.pi*np.dot(qv,Ra))
    # run over list of modes
    for jby in range(3*nat):
        ib = atoms.index_to_ia_map[jby] - 1
        Rb = atoms.atoms_dict[ib]['coordinates']
        m_ib = atoms.atoms_dict[ib]['mass']
        m_ib = m_ib * mp
        # compute ph. resolved force
        iqlp = 0
        for iqp, ilp in qlp_list:
            # e^iqpR
            qpv = qpts[iqp]
            eiqpR = cmath.exp(1j*2.*np.pi*np.dot(qpv, Rb))
            # euqp
            euqp = u[iqp]
            # wqlp (eV)
            wqlp = wu[iqp][ilp] * THz_to_ev
            # compute Raman contribution to eff_Faxby
            # eff. force : F(R) + F(2)
            eff_Faxby = np.zeros(3*nat, dtype=np.complex128)
            eff_Faxby[:] += Faxby[:,jby]
            # compute raman in spin deph obj
            eff_Faxby[:] += compute_raman(nat, jby, Fax, H, wql, wqlp)
            # update F_ql,qlp
            F_lq_lqp[:,iqlp] += eff_Faxby[:] * eiqpR * euqp[jby,ilp] / np.sqrt(m_ib)
            # eV/ang^2 * ang/eV^0.5/ps = eV^0.5/ang/ps
            iqlp += 1
    # compute e^iqR e[q] F[jax,qlp]
    for jax in range(3*nat):
        ia = atoms.index_to_ia_map[jax] - 1
        # atom mass
        m_ia = atoms.atoms_dict[ia]['mass']
        m_ia = m_ia * mp
        # effective force
        F_lq_lqp[jax,:] = eiqR[jax] * euq[jax,il] * F_lq_lqp[jax,:] / np.sqrt(m_ia)
        # [eV^0.5/ang/ps *ang/eV^0.5/ps] = 1/ps^2
    return F_lq_lqp
#
# compute the Raman contribution to 
# force matrix elements
# -> p.deph  -> <0|gX|ms><ms|gX'|0>-<1|gX|ms><ms|gX'|1>
# -> p.relax -> <1|gX|ms><ms|gX'|0>
def compute_raman(nat, jby, Fax, H, wql, wqlp):
    iqs0 = p.index_qs0
    iqs1 = p.index_qs1
    nqs = H.eig.shape[0]
    # Raman F_axby vector
    Faxby_raman = np.zeros(3*nat, dtype=np.complex128)
    # eV / ang^2 units
    # dephasing -> raman term
    if p.deph:
        for ms in range(nqs):
            Faxby_raman[:] += Fax[iqs0,ms,:] * Fax[ms,iqs0,jby] / (H.eig[iqs0]-H.eig[ms]+wql)
            Faxby_raman[:] -= Fax[iqs1,ms,:] * Fax[ms,iqs1,jby] / (H.eig[iqs1]-H.eig[ms]+wql)
            Faxby_raman[:] += Fax[iqs0,ms,:] * Fax[ms,iqs0,jby] / (H.eig[iqs0]-H.eig[ms]-wqlp)
            Faxby_raman[:] -= Fax[iqs1,ms,:] * Fax[ms,iqs1,jby] / (H.eig[iqs1]-H.eig[ms]-wqlp)
    elif p.relax:
        for ms in range(nqs):
            Faxby_raman[:] += Fax[iqs0,ms,:] * Fax[ms,iqs1,jby] / (H.eig[iqs1]-H.eig[ms]+wqlp)
            Faxby_raman[:] += Fax[iqs0,ms,:] * Fax[ms,iqs1,jby] / (H.eig[iqs1]-H.eig[ms]-wql)
    else:
        log.error("--- relax or deph calculation only ---")
    return Faxby_raman