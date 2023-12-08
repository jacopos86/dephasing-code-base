import numpy as np
import cmath
from pydephasing.phys_constants import hbar, mp, THz_to_ev
from pydephasing.atomic_list_struct import atoms
from pydephasing.input_parameters import p
from pydephasing.global_params import GPU_ACTIVE
from pydephasing.log import log
from pydephasing.mpi import mpi
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
    # ph. resolved forces
    F_lq = np.zeros((3*nat,len(ql_list)), dtype=np.complex128)
    # quantum states
    iqs0 = p.index_qs0
    iqs1 = p.index_qs1
    # eff_Fax units : [eV/ang]
    eff_Fax = np.zeros(3*nat, dtype=np.complex128)
    if p.relax:
        eff_Fax[:] = Fax[iqs0,iqs1,:]
    elif p.deph:
        eff_Fax[:] = Fax[iqs0,iqs0,:] - Fax[iqs1,iqs1,:]
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
if GPU_ACTIVE:
    from pathlib import Path
    from pydephasing.global_params import gpu
    from pycuda.compiler import SourceModule
    import pycuda.driver as cuda
# --------------------------------------------------------
#
#  order 2 phr. force -> global class
#
# --------------------------------------------------------
class phr_force_2nd_order(object):
    def __init__(self, raman=True):
        self.calc_raman = raman
        # tolerance
        self.toler = 1.E-7
    #
    #  instance CPU / GPU
    #
    def generate_instance(self):
        if GPU_ACTIVE:
            return GPU_phr_force_2nd_order ()
        else:
            return CPU_phr_force_2nd_order ()
# --------------------------------------------------------
#       GPU class
# --------------------------------------------------------
class GPU_phr_force_2nd_order(phr_force_2nd_order):
    def __init__(self):
        super(GPU_phr_force_2nd_order, self).__init__()
        # atom res. forces
        self.FAX = None
        # Q vectors
        self.NQ = 0
        self.QV = None
        # atom list
        self.R_LST = None
        # mass list
        self.M_LST = None
    #
    # prepare order 2 phr force calculation
    def set_up_2nd_order_force_phr(self, nat, qpts, Fax, Faxby, H):
        #
        if p.deph:
            self.CALCTYP = np.int32(0)
        elif p.relax:
            self.CALCTYP = np.int32(1)
        # prepare force arrays
        nqs = H.eig.shape[0]
        n = int(Faxby.shape[0])
        F0 = np.abs(np.max(Fax))
        self.JAX_LST = []
        for jax in range(n):
            for msr in range(nqs):
                for msc in range(nqs):
                    if np.abs(Fax[msr,msc,jax])/F0 > self.toler:
                        self.JAX_LST.append(jax)
        self.JAX_LST = list(set(self.JAX_LST))
        # define FAX matrix
        nax = len(self.JAX_LST)
        self.FAX = np.zeros(nqs*nqs*nax, dtype=np.complex128)
        self.IFAX_LST = np.zeros(nax, dtype=np.int32)
        for jjax in range(nax):
            jax = self.JAX_LST[jjax]
            iax = nqs*nqs*jjax
            self.IFAX_LST[jjax] = iax
            for msr in range(nqs):
                for msc in range(nqs):
                    self.FAX[iax+msr*nqs+msc] = Fax[msr,msc,jax]
        #
        # define Faxby
        F0 = np.abs(np.max(Faxby))
        jaxby_lst = []
        for jax in range(n):
            for jby in range(n):
                if np.abs(Faxby[jax,jby])/F0 > self.toler:
                    jaxby_lst.append((jax,jby))
        # define local Faxby
        naxby = len(self.jaxby_lst)
        self.FAXBY = collections.defaultdict(list)
        self.JAXBY_LST = collections.defaultdict(list)
        for jaxby in range(naxby):
            jax, jby = self.jaxby_lst[jaxby]
            self.FAXBY[jax].append(Faxby[jax,jby])
            self.JAXBY_LST[jax].append(jby)
        # set GPU input arrays
        self.RAMAN_IND = collections.defaultdict(list)
        self.FAXBY_IND = collections.defaultdict(list)
        for jax in range(3*nat):
            if jax in self.JAX_LST and jax not in self.JAXBY_LST.dict_keys():
                self.RAMAN_IND[jax] = self.JAX_LST
                self.FAXBY_IND[jax] =-np.ones(len(self.JAX_LST), dtype=np.int32)
            elif jax not in self.JAX_LST and jax in self.JAXBY_LST.dict_keys():
                self.FAXBY_IND[jax] = self.JAXBY_LST[jax]
                self.RAMAN_IND[jax] =-np.ones(len(self.JAXBY_LST[jax]), dtype=np.int32)
            elif jax in self.JAX_LST and jax in self.JAXBY_LST:
                TMP_LST = list(set(self.JAX_LST + self.JAXBY_LST[jax]))
                RAMAN_LST =-np.ones(len(TMP_LST), dtype=np.int32)
                FAXBY_LST =-np.ones(len(TMP_LST), dtype=np.int32)
                for ij in range(len(TMP_LST)):
                    jby = TMP_LST[ij]
                    if jby in self.JAX_LST:
                        RAMAN_LST[ij] = jby
                    if jby in self.JAXBY_LST:
                        FAXBY_LST[ij] = jby       
                self.RAMAN_IND[jax] = RAMAN_LST
                self.FAXBY_IND[jax] = FAXBY_LST
            else:
                pass
        for jax in self.FAXBY.keys():
            self.FAXBY[jax] = np.array(self.FAXBY[jax], dtype=np.double)
        # Q vectors list
        nq = len(qpts)
        self.NQ = np.int32(nq)
        self.QV = np.zeros(3*nq, dtype=np.double)
        iiq = 0
        for iq in range(nq):
            qv = qpts[iq]
            for ix in range(3):
                self.QV[iiq] = qv[ix]
                iiq += 1
        # Ra list
        self.R_LST = np.zeros(n, dtype=np.double)
        for jax in range(n):
            ia = atoms.index_to_ia_map[jax] - 1
            Ra = atoms.atoms_dict[ia]['coordinates']
            idx = jax%3
            self.R_LST[jax] = Ra[idx]
        # M list
        self.M_LST = np.zeros(n, dtype=np.double)
        for jax in range(n):
            ia = atoms.index_to_ia_map[jax] - 1
            m_ia = atoms.atoms_dict[ia]['mass']
            m_ia = m_ia * mp
            self.M_LST[jax] = m_ia
        # EIG
        self.EIG = np.zeros(nqs, dtype=np.double)
        self.NQS = np.int32(nqs)
        for qs in range(nqs):
            self.EIG[qs] = H.eig[qs]
        # eV
        self.IQS0 = np.int32(p.index_qs0)
        self.IQS1 = np.int32(p.index_qs1)
    #
    # driver function
    def transf_2nd_order_force_phr(self, il, iq, wu, u, nat, qlp_list):
        # initialize out array
        F_lq_lqp = np.zeros((4,3*nat,len(qlp_list)), dtype=np.complex128)
        # Fax units -> eV / ang
        # Faxby units -> eV / ang^2
        # load file
        gpu_src = Path('./pydephasing/gpu_source/compute_phr_forces.cu').read_text()
        mod = SourceModule(gpu_src)
        compute_F_raman = mod.get_function("compute_raman_force")
        # prepare input quantities
        NAT = np.int32(nat)
        wql = wu[iq][il] * THz_to_ev
        WQL = np.double(wql)
        # q vector
        qv = np.zeros(3)
        for ix in range(3):
            qv[ix] = self.QV[3*iq+ix]
        # eq
        euq = u[iq]
        EUQ = np.zeros(3*nat, dtype=np.complex128)
        for jax in range(3*nat):
            EUQ[jax] = euq[jax,il]
        # set e^iqR
        EIQR = np.zeros(3*nat, dtype=np.complex128)
        for jax in range(3*nat):
            ia = atoms.index_to_ia_map[jax] - 1
            # atomic coordinates
            Ra = atoms.atoms_dict[ia]['coordinates']
            EIQR[jax] = cmath.exp(1j*2.*np.pi*np.dot(qv,Ra))
        # (q',l') list
        QP_LST = np.zeros(len(qlp_list), dtype=np.int32)
        ILP_LST= np.zeros(len(qlp_list), dtype=np.int32)
        WQLP = np.zeros(len(qlp_list), dtype=np.double)
        iqlp = 0
        for iqp, ilp in qlp_list:
            QP_LST[iqlp] = iqp
            ILP_LST[iqlp] = ilp
            WQLP[iqlp] = wu[iqp][ilp] * THz_to_ev
            iqlp += 1
        # set eq'(l') array
        EUQLP = np.zeros(len(qlp_list)*3*nat, dtype=np.complex128)
        jj = 0
        for iqp, ilp in qlp_list:
            euqp = u[iqp]
            for jax in range(3*nat):
                EUQLP[jj] = euqp[jax,ilp]
                jj += 1
        # iterate over (q',l')
        nqlp = len(qlp_list)
        # furst compute raman term
        # if needed
        if self.calc_raman:
            # LEN IFAX_LST
            naxr = len(self.IFAX_LST)
            # Raman force array
            Fr = np.zeros((naxr,naxr,nqlp), dtype=np.complex128)
            for jjax in range(naxr):
                IAX = np.int32(self.IFAX_LST[jjax])
                # first set GRID calculations
                jjby0 = 0
                while jjby0 < naxr:
                    jjby1 = jjby0 + gpu.GRID_SIZE[0]*gpu.GRID_SIZE[1]
                    nby = min(jjby1,naxr) - jjby0
                    NBY = np.int32(nby)
                    # build local IFBY_LST
                    IFBY_LST = np.zeros(nby, dtype=np.int32)
                    IFBY_LST[:] = self.IFAX_LST[jjby0:min(jjby1,naxr)]
                    # run over (qp,lp)
                    iqlp0 = 0
                    while (iqlp0 < nqlp):
                        iqlp1 = iqlp0 + gpu.BLOCK_SIZE[0]*gpu.BLOCK_SIZE[1]*gpu.BLOCK_SIZE[2]
                        size = min(iqlp1,nqlp) - iqlp0
                        SIZE = np.int32(size)
                        # QLP_LST
                        QLP_LST = np.zeros(size, dtype=np.int32)
                        for iqlp in range(iqlp0, min(iqlp1,nqlp)):
                            QLP_LST[iqlp-iqlp0] = iqlp
                        # Raman force
                        F_RAMAN = np.zeros(gpu.gpu_size, dtype=np.complex128)
                        compute_F_raman(self.IQS0, self.IQS1, self.NQS, IAX, cuda.In(IFBY_LST), NBY, cuda.In(QLP_LST), WQL, 
                            cuda.In(WQLP), SIZE, cuda.In(self.FAX), cuda.In(self.EIG), self.CALCTYP, cuda.Out(F_RAMAN), 
                            block=gpu.block, grid=gpu.grid)
                        Fr[jjax,jjby0:min(jjby1,naxr),iqlp0:min(iqlp1,nqlp)] += gpu.recover_raman_force_from_grid(F_RAMAN, nby, size)
                        # new iqlp0
                        iqlp0 = iqlp1
                    jjby0 = jjby1
                print(np.max(F_RAMAN))
        import sys
        sys.exit()
        iqlp0 = 0
        nqlp = len(qlp_list)
        while (iqlp0 < nqlp):
            iqlp1= iqlp0 + gpu.BLOCK_SIZE[0]*gpu.BLOCK_SIZE[1]*gpu.BLOCK_SIZE[2]
            size = min(iqlp1, nqlp)-iqlp0
            SIZE = np.int32(size)
            # run over (jax)
            jax0 = 0
            while jax0 < 3*nat:
                jax1 = jax0 + gpu.GRID_SIZE[0]*gpu.GRID_SIZE[1]
                nax = min(jax1, 3*nat) - jax0
                JAX_LST = np.zeros(nax, dtype=np.int32)
                for jax in range(jax0, min(jax1, 3*nat)):
                    JAX_LST[jax-jax0] = jax
                NAX = np.int32(nax)
                # F_lq_lqp array
                FLQLQP  = np.zeros(gpu.gpu_size, dtype=np.complex128)
                FLMQLQP = np.zeros(gpu.gpu_size, dtype=np.complex128)
                FLQLMQP = np.zeros(gpu.gpu_size, dtype=np.complex128)
                FLMQLMQP= np.zeros(gpu.gpu_size, dtype=np.complex128)
                #print(self.CALCTYP, self.calc_raman)
                #import sys
                #sys.exit()
                # call gpu function
                compute_Flq_lqp(cuda.In(QP_LST), cuda.In(ILP_LST), cuda.In(JAX_LST), SIZE, NAX, NAT, WQL, cuda.In(WQLP),
                                cuda.In(EUQ), cuda.In(EUQLP), cuda.In(self.R_LST), cuda.In(self.QV), cuda.In(self.M_LST), cuda.In(EIQR), self.NQS, 
                                self.IQS0, self.IQS1, cuda.In(self.EIG), cuda.In(self.FAX), self.CALCRAMAN, self.CALCTYP,
                                cuda.Out(FLQLQP), cuda.Out(FLQLQP), cuda.Out(FLMQLQP), cuda.Out(FLQLMQP), cuda.Out(FLMQLMQP), block=gpu.block, grid=gpu.grid)
                # cuda.In(self.FAX)
                # cuda.In(self.FAXBY)
                #                 block=gpu.block, grid=gpu.grid)
                F_LQ_LQP = gpu.recover_eff_force_from_grid(FLQLQP, FLMQLQP, FLQLMQP, FLMQLMQP, nax, size)
                # reconstruct final array
                for iqlp in range(iqlp0, min(iqlp1,nqlp)):
                    for jax in range(jax0, min(jax1,3*nat)):
                        F_lq_lqp[:,jax,iqlp] += F_LQ_LQP[:,jax-jax0,iqlp-iqlp0]
                jax0 = jax1
            iqlp0 = iqlp1
        return F_lq_lqp       
# -------------------------------------------------------------------
#       CPU class
# -------------------------------------------------------------------
class CPU_phr_force_2nd_order(phr_force_2nd_order):
    def __init__(self):
        super(CPU_phr_force_2nd_order, self).__init__()
        self.Fax = None
        self.Faxby = None
        self.qv = None
        self.nq = 0
        self.nqs = 0
        self.eig = None
    #
    # prepare order 2 phr force calculation
    def set_up_2nd_order_force_phr(self, qpts, Fax, Faxby, H):
        #
        # prepare force arrays
        nqs = H.eig.shape[0]
        n = int(Faxby.shape[0])
        self.jax_lst = []
        for jax in range(n):
            for msr in range(nqs):
                for msc in range(nqs):
                    if np.abs(Fax[msr,msc,jax]) > 1.e-7:
                        self.jax_lst.append(jax)
        self.jax_lst = list(set(self.jax_lst))
        # define FAX matrix
        nax = len(self.jax_lst)
        self.Fax = np.zeros((nqs,nqs,nax), dtype=np.complex128)
        for iax in range(nax):
            jax = self.jax_lst[iax]
            self.Fax[:,:,iax] = Fax[:,:,jax]
        # 
        # prepare 2nd force arrays
        self.Faxby = Faxby
        # Q vectors list
        self.nq = len(qpts)
        self.qv = qpts
        # H eigv
        self.eig = H.eig
        self.nqs = H.eig.shape[0]
    #
    def transf_2nd_order_force_phr(self, il, iq, wu, u, nat, qlp_list):
        # Fax units -> eV / ang
        # Faxby units -> eV / ang^2
        F_lq_lqp = np.zeros((4, 3*nat, len(qlp_list)), dtype=np.complex128)
        # 0 -> (lq,l'q'); 1 -> (l-q,l'q'); 2 -> (lq,l'-q'); 3 -> (l-q,l'-q')
        # wql (eV)
        wql = wu[iq][il] * THz_to_ev
        # remember index jax -> atom,idx
        # second index (n,p)
        euq = u[iq]
        q = self.qv[iq]
        # set e^iqR
        eiqR = np.zeros(3*nat, dtype=np.complex128)
        for jax in range(3*nat):
            ia = atoms.index_to_ia_map[jax] - 1
            # atom coordinate
            Ra = atoms.atoms_dict[ia]['coordinates']
            eiqR[jax] = cmath.exp(1j*2.*np.pi*np.dot(q,Ra))
        # run over list of modes
        for jby in range(3*nat):
            ib = atoms.index_to_ia_map[jby] - 1
            Rb = atoms.atoms_dict[ib]['coordinates']
            m_ib = atoms.atoms_dict[ib]['mass']
            m_ib = m_ib * mp
            # compute Raman force
            naxr = len(self.jax_lst)
            if self.calc_raman and jby in self.jax_lst:
                Fr = self.compute_raman(jby, wql, qlp_list, wu)
            else:
                Fr = np.zeros((naxr, len(qlp_list)), dtype=np.complex128)
            # compute ph. resolved force
            iqlp = 0
            for iqp, ilp in qlp_list:
                # e^iqpR
                qp = self.qv[iqp]
                eiqpR = cmath.exp(1j*2.*np.pi*np.dot(qp, Rb))
                # euqp
                euqp = u[iqp]
                # compute Raman contribution to eff_Faxby
                # eff. force : F(R) + F(2)
                eff_Faxby = np.zeros(3*nat, dtype=np.complex128)
                eff_Faxby[:] += self.Faxby[:,jby]
                for jjax in range(naxr):
                    jax = self.jax_lst[jjax]
                    eff_Faxby[jax] += Fr[jjax,iqlp]
                # update F_ql,qlp
                F_lq_lqp[:2,:,iqlp] += eff_Faxby[:] * eiqpR * euqp[jby,ilp] / np.sqrt(m_ib)
                F_lq_lqp[2:,:,iqlp] += eff_Faxby[:] * np.conj(eiqpR) * np.conj(euqp[jby,ilp]) / np.sqrt(m_ib)
                # eV/ang^2 * ang/eV^0.5/ps = eV^0.5/ang/ps
                iqlp += 1
        # compute e^iqR e[q] F[jax,qlp]
        for jax in range(3*nat):
            ia = atoms.index_to_ia_map[jax] - 1
            # atom mass
            m_ia = atoms.atoms_dict[ia]['mass']
            m_ia = m_ia * mp
            # effective force
            F_lq_lqp[0,jax,:] = eiqR[jax] * euq[jax,il] * F_lq_lqp[0,jax,:] / np.sqrt(m_ia)
            F_lq_lqp[2,jax,:] = eiqR[jax] * euq[jax,il] * F_lq_lqp[2,jax,:] / np.sqrt(m_ia)
            F_lq_lqp[1,jax,:] = np.conj(eiqR[jax]) * np.conj(euq[jax,il]) * F_lq_lqp[1,jax,:] / np.sqrt(m_ia)
            F_lq_lqp[3,jax,:] = np.conj(eiqR[jax]) * np.conj(euq[jax,il]) * F_lq_lqp[3,jax,:] / np.sqrt(m_ia)
            # [eV^0.5/ang/ps *ang/eV^0.5/ps] = 1/ps^2
        return F_lq_lqp
    #
    # compute the Raman contribution to 
    # force matrix elements
    # -> p.deph  -> <0|gX|ms><ms|gX'|0>-<1|gX|ms><ms|gX'|1>
    # -> p.relax -> <1|gX|ms><ms|gX'|0>
    def compute_raman(self, jby, wql, qlp_list, wu):
        iqs0 = p.index_qs0
        iqs1 = p.index_qs1
        nqs = self.nqs
        naxr = len(self.jax_lst)
        # Raman F_axby vector
        Faxby_raman = np.zeros((naxr, len(qlp_list)), dtype=np.complex128)
        jjby = self.jax_lst.index(jby)
        # eV / ang^2 units
        iqlp = 0
        for iqp, ilp in qlp_list:
            # wqlp (eV)
            wqlp = wu[iqp][ilp] * THz_to_ev
            # dephasing -> raman term
            if p.deph:
                for ms in range(nqs):
                    Faxby_raman[:,iqlp] += self.Fax[iqs0,ms,:] * self.Fax[ms,iqs0,jjby] / (self.eig[iqs0]-self.eig[ms]+wql)
                    Faxby_raman[:,iqlp] -= self.Fax[iqs1,ms,:] * self.Fax[ms,iqs1,jjby] / (self.eig[iqs1]-self.eig[ms]+wql)
                    Faxby_raman[:,iqlp] += self.Fax[iqs0,ms,:] * self.Fax[ms,iqs0,jjby] / (self.eig[iqs0]-self.eig[ms]-wqlp)
                    Faxby_raman[:,iqlp] -= self.Fax[iqs1,ms,:] * self.Fax[ms,iqs1,jjby] / (self.eig[iqs1]-self.eig[ms]-wqlp)
            elif p.relax:
                for ms in range(nqs):
                    Faxby_raman[:,iqlp] += self.Fax[iqs0,ms,:] * self.Fax[ms,iqs1,jjby] / (self.eig[iqs1]-self.eig[ms]+wqlp)
                    Faxby_raman[:,iqlp] += self.Fax[iqs0,ms,:] * self.Fax[ms,iqs1,jjby] / (self.eig[iqs1]-self.eig[ms]-wql)
            else:
                if mpi.rank == mpi.root:
                    log.info("\n")
                    log.info("\t " + p.sep)
                log.error("\t relax or deph calculation only")
            iqlp += 1
        return Faxby_raman