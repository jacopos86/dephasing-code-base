import numpy as np
import cmath
import collections
from common.phys_constants import hbar, mp, THz_to_ev
from common.matrix_operations import compute_matr_elements
from pydephasing.atomic_list_struct import atoms
from pydephasing.set_param_object import p
from pydephasing.global_params import GPU_ACTIVE
from pydephasing.spin_ph_inter import SpinPhononClass
from pydephasing.log import log
from pydephasing.mpi import mpi
#

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

class SpinPhononSecndOrder(object):
    def __init__(self):
        pass
    #
    #  instance CPU / GPU
    #
    def generate_instance(self, ZFS_CALC, HFI_CALC, HESSIAN):
        if GPU_ACTIVE:
            return SpinPhononSecndOrderGPU (ZFS_CALC, HFI_CALC, HESSIAN)
        else:
            return SpinPhononSecndOrderCPU (ZFS_CALC, HFI_CALC, HESSIAN)
        
# --------------------------------------------------------
#
#   abstract second order spin phonon class
#
# --------------------------------------------------------

class SpinPhononSecndOrderBase(SpinPhononClass):
    def __init__(self):
        super(SpinPhononSecndOrderBase, self).__init__()
        # Hessian calculation
        self.hessian = None
    #
    # set HFI 2nd order gradient force
    def set_Faxby_hfi(self, grad2HFI, Hsp, displ_structs, sp_config):
        # nat
        nat = grad2HFI.struct_0.nat
        self.Fhf_axby = np.zeros((3*nat, 3*nat), dtype=np.complex128)
        Faxby = np.zeros((3*nat, 3*nat), dtype=np.complex128)
        isp_list = mpi.split_list(range(p.nsp))
        # run over spin index
        for isp in isp_list:
            # spin site
            aa = sp_config.nuclear_spins[isp]['site']-1
            Iaa= sp_config.nuclear_spins[isp]['I']
            # compute effective force
            Faxby += self.set_gaxbyA_force(aa, grad2HFI, Hsp, Iaa, displ_structs)
        # collect data
        self.Fhf_axby = mpi.collect_array(Faxby) * 2.*np.pi * hbar
        # eV / ang^2
    #
    #  compute secnd. order spin-phonon coupling
    #  and first order
    #  g_qqp = <s1|Hsp^(2)|s2>e_q(X) e_qp(Xp)^*
    #  g_ql = <s1|gX Hsp|s2> e_ql(X)
    def compute_spin_ph_coupl(self, nat, Hsp, ph, qgr, interact_dict, sp_config=None):
        # n. spin states
        n = len(Hsp.basis_vectors)
        Fax = np.zeros((n, n, 3*nat), dtype=np.complex128)
        Faxby = None
        # ZFS call
        if self.ZFS_CALC:
            gradZFS = interact_dict['gradZFS']
            Fax += self.set_gaxD_force(gradZFS, Hsp)
        # HFI call
        if self.HFI_CALC:
            gradHFI = interact_dict['gradHFI']
            Fax += self.set_Fax_hfi(self, gradHFI, Hsp, sp_config)
        # compute Hessian
        if self.hessian:
            self.compute_hessian_term(interact_dict, Hsp)
        # build ql_list
        ql_list = mpi.split_ph_modes(qgr.nq, ph.nmodes)
        # compute g_ql
        self.g_ql = self.compute_gql(nat, ql_list, qgr, ph, Hsp, Fax)
        nan_indices = np.isnan(self.g_ql)
        assert nan_indices.any() == False
        print(np.max(Fax.real))
        # compute g_qqp
        self.compute_gqqp(nat, qgr, ph, Hsp, Fax, Faxby)
    #
    #  compute Hessian term
    #
    def compute_hessian_term(self, interact_dict, Hsp):
        Faxby = None
        # ZFS Hessian
        if self.ZFS_CALC:
            if mpi.rank == mpi.root:
                log.info("\n")
                log.info("\t " + p.sep)
                log.info("\t START ZFS HESSIAN CALCULATION")
            hessZFS = interact_dict['grad2ZFS']
            Faxby = self.set_Faxby_zfs(hessZFS, Hsp)
            if mpi.rank == mpi.root:
                log.info("\t END ZFS HESSIAN CALCULATION")
                log.info("\t " + p.sep)
        if self.HFI_CALC:
            if mpi.rank == mpi.root:
                log.info("\n")
                log.info("\t " + p.sep)
                log.info("\t START HFI HESSIAN CALCULATION")
            hessHFI = interact_dict['grad2HFI']
            if mpi.rank == mpi.root:
                log.info("\t END HFI HESSIAN CALCULATION")
                log.info("\t " + p.sep)
        return Faxby
    #
    #
    # compute g_{ab}(q,qp)
    # = \sum_{nn';ss'} Aq e^{iq Rn}e_q(s) <a|H^(2)|b> Aqp e^{iqp Rn'}e_qp(s')
    #
    def compute_gqqp(self, nat, qgr, ph, Hsp, Fax, Faxby):
        if not self.hessian:
            assert Faxby == None
        # first compute raman
        # contribution
        FXXp = self.compute_raman(nat, Hsp, Fax)
        # if available compute add to Hessian
        if Faxby is not None:
            FXXp += 0.5 * Faxby
        # compute gqqp
        # make list of q vector pairs for each proc.

# --------------------------------------------------------
#       GPU class
# --------------------------------------------------------

class SpinPhononSecndOrderGPU(SpinPhononSecndOrderBase):
    def __init__(self, ZFS_CALC, HFI_CALC, HESSIAN):
        super(SpinPhononSecndOrderGPU, self).__init__()
        self.ZFS_CALC = ZFS_CALC
        self.HFI_CALC = HFI_CALC
        # add Hessian contribution
        self.hessian = HESSIAN
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
        n = 3*nat
        if p.deph:
            self.CALCTYP = np.int32(0)
        elif p.relax:
            self.CALCTYP = np.int32(1)
        if self.calc_raman:
            # prepare force arrays
            nqs = H.eig.shape[0]
            F0 = np.abs(np.max(Fax))
            jax_lst = []
            for jax in range(n):
                for msr in range(nqs):
                    for msc in range(nqs):
                        if np.abs(Fax[msr,msc,jax])/F0 > self.toler:
                            jax_lst.append(jax)
            jax_lst = list(set(jax_lst))
            # define FAX matrix
            nax = len(jax_lst)
            self.FAX = np.zeros(nqs*nqs*nax, dtype=np.complex128)
            IFAX_LST = np.zeros(nax, dtype=np.int32)
            for jjax in range(nax):
                jax = jax_lst[jjax]
                iax = nqs*nqs*jjax
                IFAX_LST[jjax] = iax
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
        jaxby_lst.append((110,1))
        jaxby_lst.append((110,2))
        # define local Faxby
        #
        naxby = len(jaxby_lst)
        self.JAXBY_LST = collections.defaultdict(list)
        for jaxby in range(naxby):
            jax, jby = jaxby_lst[jaxby]
            self.JAXBY_LST[jax].append(jby)
        #
        # set GPU input arrays RAMAN/FAXBY INDEX
        self.JAXBY_KEYS = [key for key, lst in self.JAXBY_LST.items() if len(lst) > 0]
        #
        # raman calc. indexes
        if self.calc_raman:
            # RAMAN_IND MAPS TO JAX_LST
            self.FBY_IND   = collections.defaultdict(list)
            self.FAX_IND   = collections.defaultdict(list)
            self.JBY_LST   = collections.defaultdict(list)
            self.FAXBY_IND = collections.defaultdict(list)
            self.FAXBY     = collections.defaultdict(list)
            for jax in range(3*nat):
                if jax in jax_lst and jax not in self.JAXBY_KEYS:
                    self.JBY_LST[jax] = np.array(jax_lst, dtype=np.int32)
                    self.FAX_IND[jax]   = np.int32(jax_lst.index(jax)*nqs*nqs)
                    self.FBY_IND[jax]   = np.array(IFAX_LST, dtype=np.int32)
                    self.FAXBY_IND[jax] =-np.ones(len(jax_lst), dtype=np.int32)
                    self.FAXBY[jax]     = np.zeros(len(jax_lst), dtype=np.double)
                elif jax not in jax_lst and jax in self.JAXBY_KEYS:
                    self.FAXBY_IND[jax] = np.array(range(len(self.JAXBY_LST[jax])), dtype=np.int32)
                    self.FAXBY[jax]     = np.array(self.FAXBY[jax], dtype=np.double)
                    self.JBY_LST[jax]   = np.array(self.JAXBY_LST[jax], dtype=np.int32)
                    self.FAX_IND[jax]   = np.int32(-1)
                    self.FBY_IND[jax]   =-np.ones(len(self.JAXBY_LST[jax]), dtype=np.int32)
                    self.FAXBY[jax]     = np.zeros(len(self.JAXBY_LST[jax]), dtype=np.double)
                    jjby = 0
                    for jby in self.JAXBY_LST[jax]:
                        self.FAXBY[jax][jjby] = Faxby[jax,jby]
                        jjby += 1
                elif jax in jax_lst and jax in self.JAXBY_KEYS:
                    self.JBY_LST[jax] = list(set(jax_lst + self.JAXBY_LST[jax]))
                    self.JBY_LST[jax] = np.array(self.JBY_LST[jax], dtype=np.int32)
                    FBY_TMP   =-np.ones(len(self.JBY_LST[jax]), dtype=np.int32)
                    FAXBY_TMP =-np.ones(len(self.JBY_LST[jax]), dtype=np.int32)
                    self.FAXBY[jax]   = np.zeros(len(self.JBY_LST[jax]), dtype=np.double)
                    for ij in range(len(self.JBY_LST[jax])):
                        jby = self.JBY_LST[jax][ij]
                        if jby in jax_lst:
                            FBY_TMP[ij]   = nqs*nqs*jax_lst.index(jby)
                        if jby in self.JAXBY_LST[jax]:
                            FAXBY_TMP[ij] = self.JAXBY_LST[jax].index(jby)
                            self.FAXBY[jax][ij] = Faxby[jax,jby]
                    self.FBY_IND[jax]   = FBY_TMP
                    self.FAXBY_IND[jax] = FAXBY_TMP
                    self.FAX_IND[jax]   = np.int32(jax_lst.index(jax)*nqs*nqs)
                else:
                    pass
            self.JAX_KEYS = [key for key, lst in self.FBY_IND.items() if len(lst) > 0]
        else:
            self.FAXBY     = collections.defaultdict(list)
            for jax in self.JAXBY_KEYS:
                self.FAXBY[jax] = np.zeros(len(self.JAXBY_LST[jax]), dtype=np.double)
                for ij in range(len(self.JAXBY_LST[jax])):
                    jby = self.JAXBY_LST[jax][ij]
                    self.FAXBY[jax][ij] = Faxby[jax,jby]
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
        if self.calc_raman:
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
        if self.calc_raman:
            compute_F_raman = mod.get_function("compute_raman_force")
            compute_Flq_lqp_raman = mod.get_function("compute_Flqlqp_raman")
        else:
            compute_Flq_lqp = mod.get_function("compute_Flqlqp")
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
            WQLP[iqlp] = wu[iqp][ilp]*THz_to_ev
            iqlp += 1
        # set e_q'(l') array
        EUQLP = np.zeros(len(qlp_list)*3*nat, dtype=np.complex128)
        jj = 0
        for iqp, ilp in qlp_list:
            euqp = u[iqp]
            for jax in range(3*nat):
                EUQLP[jj] = euqp[jax,ilp]
                jj += 1
        # iterate over (q',l')
        nqlp = len(qlp_list)
        # first compute raman term
        # if needed
        if self.calc_raman:
            # LEN FAX_IND
            nax = len(self.JAX_KEYS)
            # run over jax index
            for jjax in range(nax):
                jax = self.JAX_KEYS[jjax]
                IAX = self.FAX_IND[jax]
                # run over (qp,lp)
                iqlp0 = 0
                while (iqlp0 < nqlp):
                    iqlp1 = iqlp0 + gpu.BLOCK_SIZE[0]*gpu.BLOCK_SIZE[1]*gpu.BLOCK_SIZE[2]
                    size = min(iqlp1,nqlp) - iqlp0
                    SIZE = np.int32(size)
                    # local QLP_LST
                    QLP_LST = np.array(range(iqlp0, min(iqlp1,nqlp)), dtype=np.int32)
                    assert len(QLP_LST) == size
                    F_LQ_LQP = np.zeros((4,size), dtype=np.complex128)
                    # first set GRID calculations
                    nbyt = len(self.JBY_LST[jax])
                    jjby0 = 0
                    while jjby0 < nbyt:
                        jjby1 = jjby0 + gpu.GRID_SIZE[0]*gpu.GRID_SIZE[1]
                        nby = min(jjby1,nbyt) - jjby0
                        NBY = np.int32(nby)
                        # local jby list
                        JBY_LST = np.zeros(nby, dtype=np.int32)
                        JBY_LST[:] = self.JBY_LST[jax][jjby0:min(jjby1,nbyt)]
                        # build local IFBY_LST
                        FBY_IND = np.zeros(nby, dtype=np.int32)
                        FBY_IND[:] = self.FBY_IND[jax][jjby0:min(jjby1,nbyt)]
                        FAXBY_IND = np.zeros(nby, dtype=np.int32)
                        FAXBY_IND[:] = self.FAXBY_IND[jax][jjby0:min(jjby1,nbyt)]
                        FAXBY = np.zeros(nby, dtype=np.double)
                        FAXBY[:] = self.FAXBY[jax][jjby0:min(jjby1,nbyt)]
                        # Raman force
                        F_RAMAN = np.zeros(gpu.gpu_size, dtype=np.complex128)
                        if IAX > -1:
                            compute_F_raman(self.IQS0, self.IQS1, self.NQS, IAX, cuda.In(FBY_IND), NBY, cuda.In(QLP_LST), WQL, 
                                cuda.In(WQLP), SIZE, cuda.In(self.FAX), cuda.In(self.EIG), self.CALCTYP, cuda.Out(F_RAMAN), 
                                block=gpu.block, grid=gpu.grid)
                        # intead of doing this give F_RAMAN in input
                        # to fql_qlp calculation directly here
                        # THIS SHOULD SAVE A LOT OF TIME
                        # F_lq_lqp array
                        FLQLQP  = np.zeros(gpu.gpu_size, dtype=np.complex128)
                        FLMQLQP = np.zeros(gpu.gpu_size, dtype=np.complex128)
                        FLQLMQP = np.zeros(gpu.gpu_size, dtype=np.complex128)
                        FLMQLMQP= np.zeros(gpu.gpu_size, dtype=np.complex128)
                        # Fr[jjax,jjby0:min(jjby1,naxr),iqlp0:min(iqlp1,nqlp)] += gpu.recover_raman_force_from_grid(F_RAMAN, nby, size)
                        compute_Flq_lqp_raman(NAT, NBY, SIZE, cuda.In(EUQLP), cuda.In(self.R_LST),
                            cuda.In(self.QV), cuda.In(self.M_LST), cuda.In(QP_LST), cuda.In(QLP_LST), 
                            cuda.In(FAXBY_IND), cuda.In(JBY_LST), cuda.In(F_RAMAN), cuda.In(FAXBY),
                            cuda.Out(FLQLQP), cuda.Out(FLMQLQP), cuda.Out(FLQLMQP), cuda.Out(FLMQLMQP),
                            block=gpu.block, grid=gpu.grid)
                        # compute eff. force
                        F_LQ_LQP += gpu.recover_eff_force_from_grid(FLQLQP, FLMQLQP, FLQLMQP, FLMQLMQP, nby, size)
                        jjby0 = jjby1
                    # reconstruct final array
                    for iqlp in range(iqlp0,min(iqlp1,nqlp)):
                        F_lq_lqp[:,jax,iqlp] += F_LQ_LQP[:,iqlp-iqlp0]
                    # new iqlp0
                    iqlp0 = iqlp1
                # compute final force
                # each jax
                F_lq_lqp[0,jax,:] = EIQR[jax] * euq[jax,il] * F_lq_lqp[0,jax,:] / np.sqrt(self.M_LST[jax])
                F_lq_lqp[2,jax,:] = EIQR[jax] * euq[jax,il] * F_lq_lqp[2,jax,:] / np.sqrt(self.M_LST[jax])
                F_lq_lqp[1,jax,:] = np.conj(EIQR[jax]) * np.conj(euq[jax,il]) * F_lq_lqp[1,jax,:] / np.sqrt(self.M_LST[jax])
                F_lq_lqp[3,jax,:] = np.conj(EIQR[jax]) * np.conj(euq[jax,il]) * F_lq_lqp[3,jax,:] / np.sqrt(self.M_LST[jax])
        else:
            #
            #  compute ph. resolved
            #  force
            jaxby_keys = [key for key, lst in self.JAXBY_LST.items() if len(lst) > 0]
            for jax in jaxby_keys:
                # run over (q',l')
                iqlp0 = 0
                while (iqlp0 < nqlp):
                    iqlp1= iqlp0 + gpu.BLOCK_SIZE[0]*gpu.BLOCK_SIZE[1]*gpu.BLOCK_SIZE[2]
                    size = min(iqlp1, nqlp)-iqlp0
                    # local QLP_LST
                    QLP_LST = np.array(range(iqlp0, min(iqlp1,nqlp)), dtype=np.int32)
                    assert len(QLP_LST) == size
                    F_LQ_LQP = np.zeros((4,size), dtype=np.complex128)
                    SIZE = np.int32(size)
                    # run over (jby)
                    jjby0 = 0
                    while jjby0 < len(self.JAXBY_LST[jax]):
                        jjby1 = jjby0 + gpu.GRID_SIZE[0]*gpu.GRID_SIZE[1]
                        nby = min(jjby1, len(self.JAXBY_LST[jax])) - jjby0
                        JBY_LST = np.zeros(nby, dtype=np.int32)
                        FAXBY   = np.zeros(nby, dtype=np.double)
                        for jjby in range(jjby0, min(jjby1, len(self.JAXBY_LST[jax]))):
                            JBY_LST[jjby-jjby0] = self.JAXBY_LST[jax][jjby]
                            FAXBY[jjby-jjby0]   = self.FAXBY[jax][jjby]
                        NBY = np.int32(nby)
                        # F_lq_lqp array
                        FLQLQP  = np.zeros(gpu.gpu_size, dtype=np.complex128)
                        FLMQLQP = np.zeros(gpu.gpu_size, dtype=np.complex128)
                        FLQLMQP = np.zeros(gpu.gpu_size, dtype=np.complex128)
                        FLMQLMQP= np.zeros(gpu.gpu_size, dtype=np.complex128)
                        # call gpu function
                        compute_Flq_lqp(NAT, NBY, SIZE, cuda.In(QP_LST), cuda.In(QLP_LST), cuda.In(JBY_LST),
                                cuda.In(EUQLP), cuda.In(self.R_LST), cuda.In(self.QV), cuda.In(self.M_LST),
                                cuda.In(FAXBY), cuda.Out(FLQLQP), cuda.Out(FLMQLQP), cuda.Out(FLQLMQP), 
                                cuda.Out(FLMQLMQP), block=gpu.block, grid=gpu.grid)
                        #
                        # recover phr force
                        F_LQ_LQP += gpu.recover_eff_force_from_grid(FLQLQP, FLMQLQP, FLQLMQP, FLMQLMQP, nby, size)
                        jjby0 = jjby1
                    # reconstruct final array
                    for iqlp in range(iqlp0, min(iqlp1,nqlp)):
                        F_lq_lqp[:,jax,iqlp] += F_LQ_LQP[:,iqlp-iqlp0]
                    iqlp0 = iqlp1
                # compute final force
                # each jax
                F_lq_lqp[0,jax,:] = EIQR[jax] * euq[jax,il] * F_lq_lqp[0,jax,:] / np.sqrt(self.M_LST[jax])
                F_lq_lqp[2,jax,:] = EIQR[jax] * euq[jax,il] * F_lq_lqp[2,jax,:] / np.sqrt(self.M_LST[jax])
                F_lq_lqp[1,jax,:] = np.conj(EIQR[jax]) * np.conj(euq[jax,il]) * F_lq_lqp[1,jax,:] / np.sqrt(self.M_LST[jax])
                F_lq_lqp[3,jax,:] = np.conj(EIQR[jax]) * np.conj(euq[jax,il]) * F_lq_lqp[3,jax,:] / np.sqrt(self.M_LST[jax])
        print('OK')
        return F_lq_lqp
    
# -------------------------------------------------------------------
#       CPU class
# -------------------------------------------------------------------

class SpinPhononSecndOrderCPU(SpinPhononSecndOrderBase):
    def __init__(self, ZFS_CALC, HFI_CALC, HESSIAN):
        super(SpinPhononSecndOrderCPU, self).__init__()
        self.ZFS_CALC = ZFS_CALC
        self.HFI_CALC = HFI_CALC
        # add Hessian contribution
        self.hessian = HESSIAN
    # set up < s1 | S grad_axby D S | s2 > coefficients
    def set_Faxby_zfs(self, hessZFS, Hsp):
        nat = hessZFS.struct_0.nat
        n = len(Hsp.basis_vectors)
        # partition jax between proc.
        jax_list = mpi.random_split(range(3*nat))
        # S g^2D S matrix elements
        Faxby = np.zeros((n, n, 3*nat, 3*nat), dtype=np.complex128)
        for jax in jax_list:
            for jby in range(jax, 3*nat):
                SggDS = np.zeros((n,n), dtype=np.complex128)
                HessD = np.zeros((n,n))
                HessD[:,:] = hessZFS.U_grad2D_U[jax,jby,:,:]
                # THz / ang^2
                SggDS =  HessD[0,0] * np.matmul(Hsp.Sx, Hsp.Sx)
                SggDS += HessD[0,1] * np.matmul(Hsp.Sx, Hsp.Sy)
                SggDS += HessD[1,0] * np.matmul(Hsp.Sy, Hsp.Sx)
                SggDS += HessD[1,1] * np.matmul(Hsp.Sy, Hsp.Sy)
                SggDS += HessD[0,2] * np.matmul(Hsp.Sx, Hsp.Sz)
                SggDS += HessD[2,0] * np.matmul(Hsp.Sz, Hsp.Sx)
                SggDS += HessD[1,2] * np.matmul(Hsp.Sy, Hsp.Sz)
                SggDS += HessD[2,1] * np.matmul(Hsp.Sz, Hsp.Sy)
                SggDS += HessD[2,2] * np.matmul(Hsp.Sz, Hsp.Sz)
                # compute matrix elements
                for i1 in range(n):
                    qs1 = Hsp.qs[i1]['eigv']
                    for i2 in range(n):
                        qs2 = Hsp.qs[i2]['eigv']
                        # <qs1|SggDS|qs2>
                        Faxby[i1,i2,jax,jby] = compute_matr_elements(SggDS, qs1, qs2)
                if jby != jax:
                    Faxby[:,:,jby,jax] = Faxby[:,:,jax,jby]
        # THz / ang^2 units
        # collect data into single proc.
        mpi.comm.Barrier()
        Faxby =  mpi.collect_array(Faxby)
        assert np.max(Faxby.real)*2.*np.pi*hbar == np.max(Faxby.real)*THz_to_ev
        Faxby = Faxby * THz_to_ev
        if mpi.rank == mpi.root:
            log.info("\t Faxby shape: " + str(Faxby.shape))
        # eV / ang^2 units
        return Faxby
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
    # Fr_aa'(X,X') = sum_b' {Fab(X) Fba'(X')/(e_a' - e_b) + Fab(X) Fba'(X')/(e_a - e_b)}
    def compute_raman(self, nat, Hsp, Fax):
        # Raman F_axby vector
        n = len(Hsp.basis_vectors)
        FXXp = np.zeros((n, n, 3*nat, 3*nat), dtype=np.complex128)
        # result in eV / ang^2 units
        # partition jax between proc.
        deg = Hsp.check_degeneracy()
        if mpi.rank == mpi.root:
            log.info("spin Hamiltonian degenerate: " + str(deg))
        if deg:
            log.warning("Spin Hamiltonian is degenerate")
        jX_list = mpi.random_split(range(3*nat))
        if not deg:
            for jX in jX_list:
                for jXp in range(3*nat):
                    # matrix elements
                    for a in range(n):
                        e_a = Hsp.qs[a]['eig']
                        for ap in range(n):
                            e_ap = Hsp.qs[ap]['eig']
                            for b in range(n):
                                e_b = Hsp.qs[b]['eig']
                                if b != ap:
                                    FXXp[a,ap,jX,jXp] += Fax[a,b,jX] * Fax[b,ap,jXp] / (e_ap - e_b)
                                if b != a:
                                    FXXp[a,ap,jX,jXp] += Fax[a,b,jX] * Fax[b,ap,jXp] / (e_a - e_b)
        # eV / ang^2
        # collect into single proc.
        mpi.comm.Barrier()
        print(np.max(FXXp.real))
        FXXp = mpi.collect_array(FXXp)
        nan_indices = np.isnan(FXXp)
        assert nan_indices.any() == False
        return FXXp