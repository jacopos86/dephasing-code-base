import numpy as np
import cmath
import collections
from itertools import product
from common.phys_constants import hbar, mp, THz_to_ev
from common.matrix_operations import compute_matr_elements
from pydephasing.GPU_arrays_handler import GPU_ARRAY
from pydephasing.atomic_list_struct import atoms
from pydephasing.set_param_object import p
from pydephasing.global_params import GPU_ACTIVE, CUDA_SOURCE_DIR
from pydephasing.spin_ph_inter import SpinPhononClass
from pydephasing.log import log
from pydephasing.mpi import mpi
from pathlib import Path

if GPU_ACTIVE:
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
    # set ZFS 2nd order gradient force
    def set_Faxby_zfs(self, hessZFS, Hsp):
        n = len(Hsp.basis_vectors)
        nat = hessZFS.struct_0.nat
        # hessian force
        Faxby = np.zeros((n, n, 3*nat, 3*nat), dtype=np.complex128)
        # jax list
        jax_list = mpi.split_list(range(3*nat))
        # run over jax index
        for jax in jax_list:
            for jby in range(jax, 3*nat):
                ShDS = np.zeros((n,n), dtype=np.complex128)
                hD = np.zeros((3,3))
                hD[:,:] = hessZFS.U_grad2D_U[jax,jby,:,:]
                # THz / ang^2
                ShDS = hD[0,0] * np.matmul(Hsp.Sx, Hsp.Sx)
                ShDS+= hD[0,1] * np.matmul(Hsp.Sx, Hsp.Sy)
                ShDS+= hD[1,0] * np.matmul(Hsp.Sy, Hsp.Sx)
                ShDS+= hD[1,1] * np.matmul(Hsp.Sy, Hsp.Sy)
                ShDS+= hD[0,2] * np.matmul(Hsp.Sx, Hsp.Sz)
                ShDS+= hD[2,0] * np.matmul(Hsp.Sz, Hsp.Sx)
                ShDS+= hD[1,2] * np.matmul(Hsp.Sy, Hsp.Sz)
                ShDS+= hD[2,1] * np.matmul(Hsp.Sz, Hsp.Sy)
                ShDS+= hD[2,2] * np.matmul(Hsp.Sz, Hsp.Sz)
                # matrix elements in the basis
                # of unpert. Hamiltonian
                for i1 in range(len(Hsp.qs)):
                    qs1 = Hsp.qs[i1]['eigv']
                    for i2 in range(len(Hsp.qs)):
                        qs2 = Hsp.qs[i2]['eigv']
                        # <qs1|ShDS|qs2>
                        Faxby[i1,i2,jax,jby] = compute_matr_elements(ShDS, qs1, qs2)
                if jby != jax:
                    Faxby[:,:,jby,jax] = Faxby[:,:,jax,jby]
        mpi.comm.Barrier()
        # collect data
        Faxby = mpi.collect_array(Faxby)
        # convert THz / ang^2 -> eV / ang^2
        Faxby = Faxby * 2. * np.pi * hbar
        return Faxby
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
            Fax += self.set_Fax_hfi(gradHFI, Hsp, sp_config)
        # compute Hessian
        if self.hessian:
            Faxby = self.compute_hessian_term(interact_dict, Hsp)
        exit()
        # build ql_list
        ql_list = mpi.split_ph_modes(qgr.nq, ph.nmodes)
        # compute g_ql
        self.g_ql = self.compute_gql(nat, ql_list, qgr, ph, Hsp, Fax)
        nan_indices = np.isnan(self.g_ql)
        assert nan_indices.any() == False
        print("max Fx", np.max(Fax.real))
        # compute g_qqp
        self.set_gqqp_calculation(nat, qgr, ph, Hsp, Fax, Faxby)
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
            if mpi.rank == mpi.root:
                log.info("\t " + p.sep)
                log.info("\t Hess DZFS - shape: " + str(hessZFS.grad2Dtensor.shape))
                log.info("\t " + p.sep)
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
    def set_gqqp_calculation(self, nat, qgr, ph, Hsp, Fax, Faxby):
        if not self.hessian:
            assert Faxby == None
        # first compute raman wq=0 value
        # CPU algorithm keeps terms only within given fraction of FXXp0
        if not GPU_ACTIVE:
            FXXp0 = self.compute_raman(nat, Hsp, Fax)
            print("max FXX'", np.max(FXXp0.real))
        # compute gqqp
        # make list of q vector pairs for each proc.
        qqp_list = qgr.build_irred_qqp_pairs()
        # parallelize calculation over (q,q')
        qqp_list = mpi.split_list(qqp_list)
        # compute g_qqp
        for iq, iqp in qqp_list:
            file_name = 'G-iq-' + str(iq) + '-iqp-' + str(iqp) + '.npy'
            file_path = p.work_dir + '/restart/' + file_name
            file_path = "{}".format(file_path)
            fil = Path(file_path)
            if not fil.exists():
                if not GPU_ACTIVE:
                    gqqp = self.compute_gqqp(nat, iq, iqp, qgr, ph, Hsp, Fax, FXXp0, Faxby)
                else:
                    gqqp = self.compute_gqqp(nat, iq, iqp, qgr, ph, Hsp, Fax, Faxby)
                # save data
                np.savez(file_path, G=gqqp, illp=[])
            else:
                pass

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
    #  -------------------------------------------------------------------------
    #
    #       driver function
    #
    # --------------------------------------------------------------------------
    def compute_gqqp(self, nat, iq, iqp, qgr, ph, Hsp, Fax, Faxby=None):
        # FXXp units -> eV / ang^2
        n = len(Hsp.basis_vectors)
        gqqp = np.zeros((n, n, ph.nmodes, ph.nmodes), dtype=np.complex128)
        GQQP = GPU_ARRAY(gqqp, np.complex128)
        print(gqqp.shape)
        print("iq, iqp", mpi.rank, iq, iqp)
        # load file 
        gpu_src = Path(CUDA_SOURCE_DIR+'compute_two_phonons_matr.cu').read_text()
        gpu_mod = SourceModule(gpu_src, options=["-I"+CUDA_SOURCE_DIR])
        # prepare input quantities
        NAT = np.int32(nat)
        NMD = np.int32(ph.nmodes)
        # phonon energies
        WQL = GPU_ARRAY(np.array(ph.uql[iq]) * THz_to_ev, np.double)
        WQPL= GPU_ARRAY(np.array(ph.uql[iqp]) * THz_to_ev, np.double)
        # amplitudes
        ql_list = list(product([iq], range(ph.nmodes)))
        AQL = GPU_ARRAY(ph.compute_ph_amplitude_q(nat, ql_list), np.double)
        qpl_list = list(product([iqp], range(ph.nmodes)))
        AQPL = GPU_ARRAY(ph.compute_ph_amplitude_q(nat, qpl_list), np.double)
        # energy eigenvalues array -> EIG
        eig = np.zeros(n)
        for a in range(n):
            eig[a] = Hsp.qs[a]['eig']
        EIG = GPU_ARRAY(eig, np.double)
        NST = EIG.length()
        # FX -> allocate GPU array
        FX = GPU_ARRAY(Fax, np.complex128)
        # Hessian term
        if Faxby is not None:
            FXXp = GPU_ARRAY(Faxby, np.complex128)
        else:
            compute_gqqp = gpu_mod.get_function("compute_gqqp")
        # -> GPU parallelized arrays
        illp_list = np.array(list(product(range(ph.nmodes), range(ph.nmodes))))
        INIT_INDEX, SIZE_LIST = gpu.distribute_data_on_grid(illp_list)
        MODES_LIST = GPU_ARRAY(illp_list, np.int32)
        print(SIZE_LIST.cpu_array)
        print(illp_list[850])
        # set e^iqR
        eiqr = np.zeros(atoms.supercell_size, dtype=np.complex128)
        eiqpr= np.zeros(atoms.supercell_size, dtype=np.complex128)
        qv = qgr.qpts[iq]
        qpv= qgr.qpts[iqp]
        for iL in range(atoms.supercell_size):
            Rn = atoms.supercell_grid[iL]
            eiqr[iL] = cmath.exp(1j*2.*np.pi*np.dot(qv,Rn))
            eiqpr[iL]= cmath.exp(1j*2.*np.pi*np.dot(qpv,Rn))
        EIQR = GPU_ARRAY(eiqr, np.complex128)
        EIQPR= GPU_ARRAY(eiqpr, np.complex128)
        NL = EIQR.length()
        # ph. vectors -> EQ
        eq = ph.eql[iq]
        eqp = ph.eql[iqp]
        for jax in range(3*nat):
            ia = atoms.index_to_ia_map[jax]
            m_ia = atoms.atoms_mass[ia]
            eq[jax,:] = eq[jax,:] / np.sqrt(m_ia)
            eqp[jax,:] = eqp[jax,:] / np.sqrt(m_ia)
            # Ang/ev^1/2 ps^-1
        print(eq[10,0])
        EQ = GPU_ARRAY(eq, np.complex128)
        EQP = GPU_ARRAY(eqp, np.complex128)
        #print(EQ.cpu_array[2,100])
        #  call GPU function
        compute_gqqp(NAT, NL, NST, NMD, cuda.In(INIT_INDEX.to_gpu()), cuda.In(SIZE_LIST.to_gpu()), cuda.In(MODES_LIST.to_gpu()),
                cuda.In(AQL.to_gpu()), cuda.In(AQPL.to_gpu()), cuda.In(WQL.to_gpu()), cuda.In(WQPL.to_gpu()), cuda.In(EIG.to_gpu()), 
                cuda.In(FX.to_gpu()), cuda.In(EQ.to_gpu()), cuda.In(EQP.to_gpu()), cuda.In(EIQR.to_gpu()), cuda.In(EIQPR.to_gpu()), 
                cuda.Out(GQQP.to_gpu()), block=gpu.block, grid=gpu.grid)
        exit()
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
    #
    # compute the Raman contribution to 
    # force matrix elements
    # Fr_aa'(X,X') = sum_b' {Fab(X) Fba'(X')/(e_a' - e_b) + Fab(X) Fba'(X')/(e_a - e_b)}
    def compute_raman(self, nat, Hsp, Fax):
        # read function
        gpu_src = Path(CUDA_SOURCE_DIR+'compute_raman_matr.cu').read_text()
        gpu_mod = SourceModule(gpu_src)
        # Raman F_axby vector
        n = len(Hsp.basis_vectors)
        # result in eV / ang^2 units
        # partition jax between proc.
        deg = Hsp.check_degeneracy()
        if mpi.rank == mpi.root:
            log.info("\t spin Hamiltonian degenerate: " + str(deg))
        if deg:
            log.warning("\t Spin Hamiltonian is degenerate")
        jX_list = mpi.random_split(range(3*nat))
        jXp_list = np.array(range(3*nat))
        jXXp_list = np.array(list(product(jX_list, jXp_list)))
        # make list of (jX,jXp) on gpu grid
        INIT_INDEX, SIZE_LIST = gpu.distribute_data_on_grid(jXXp_list)
        JXXP_LIST = GPU_ARRAY(jXXp_list, np.int32)
        NJX = np.int32(3*nat)
        # EIG array -> allocate GPU memory
        # & copy data
        eig = np.zeros(n)
        for a in range(n):
            eig[a] = Hsp.qs[a]['eig']
        EIG = GPU_ARRAY(eig, np.float32)
        # FX -> ALLOCATE GPU array
        FX = GPU_ARRAY(Fax, np.complex128)
        print(FX.to_gpu()[0,0,0], FX.to_gpu()[1,0,0], FX.to_gpu()[2,0,0], FX.to_gpu()[0,1,0], FX.to_gpu()[1,1,0], FX.to_gpu()[2,1,0])
        # raman matrix elements
        FXXP = GPU_ARRAY(np.zeros((n, n, len(jXXp_list))), np.complex128)
        FXXP_ARRAY = np.zeros((n, n, len(jXXp_list)), dtype=np.complex128)
        if not deg:
            compute_raman = gpu_mod.get_function("compute_raman_nondeg")
        else:
            compute_raman = gpu_mod.get_function("compute_raman_deg")
        compute_raman(EIG.length(), NJX, cuda.In(INIT_INDEX.to_gpu()), cuda.In(SIZE_LIST.to_gpu()), cuda.In(JXXP_LIST.to_gpu()), cuda.In(EIG.to_gpu()),
                      cuda.In(FX.to_gpu()), cuda.Out(FXXP_ARRAY), block=gpu.block, grid=gpu.grid)
        # reshape array
        #gpu.create_index_list((n, n, len(JXXP_LIST)))
        #FXXp = gpu.reshape_array(jXXp_list, FXXP)
        FXXP.reshape_gpu_array(FXXP_ARRAY, jXXp_list, (n, n, 3*nat, 3*nat))
        return FXXP.cpu_array
    
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
    #  compute gqqp
    #
    def compute_gqqp(self, nat, iq, iqp, qgr, ph, Hsp, Fax, FXXp0, Faxby=None):
        # FXXp units -> eV / ang^2
        n = len(Hsp.basis_vectors)
        gqqp = np.zeros((n, n, ph.nmodes, ph.nmodes), dtype=np.complex128)
        print(gqqp.shape)
        print(mpi.rank, iq, iqp)
        # set modes list
        ql_list = []
        for il in range(ph.nmodes):
            ql_list.append((iq, il))
        # pre-compute ph. amplitudes
        A_ql = ph.compute_ph_amplitude_q(nat, ql_list)
        # set modes list
        qpl_list = []
        for il in range(ph.nmodes):
            qpl_list.append((iqp, il))
        # pre-compute ph. amplitudes
        A_qpl = ph.compute_ph_amplitude_q(nat, qpl_list)
        # compute raman term
        
        # set e^iqR
        q = qgr.qpts[iq]
        L = atoms.supercell_size
        eiqR = np.zeros(L, dtype=np.complex128)
        for i in range(L):
            Rn = atoms.supercell_grid[i]
            eiqR[i] = cmath.exp(1j*2.*np.pi*np.dot(q,Rn))
        # set e^iqpR
        qp = qgr.qpts[iqp]
        eiqpR = np.zeros(L, dtype=np.complex128)
        for i in range(L):
            Rn = atoms.supercell_grid[i]
            eiqpR[i] = cmath.exp(1j*2.*np.pi*np.dot(qp,Rn))
        # run over modes pairs (ql,q'l')
        for il in range(ph.nmodes):
            for ilp in range(ph.nmodes):
                # compute ph. resolved force
                # compute first order raman term
                # for this (q,q') pair
                Fr1 = np.zeros((n, n, 3*nat, 3*nat), dtype=np.complex128)
                F = Fr1 + FXXp
                # compute gqqp
                for jx in range(3*nat):
                    mx = atoms.atoms_mass[atoms.index_to_ia_map[jx]]
                    eq = ph.eql[iq][jx,il] / np.sqrt(mx)
                    for jxp in range(3*nat):
                        mxp = atoms.atoms_mass[atoms.index_to_ia_map[jxp]]
                        eqp = ph.eql[iqp][jxp,ilp] / np.sqrt(mxp)
                        # ang/eV^1/2 *ps^-1
                        for i in range(L):
                            for j in range(L):
                                gqqp[:,:,il,ilp] += A_ql[il] * eiqR[i] * eq * F[:,:,jx,jxp] * eqp * eiqpR[j] * A_qpl[ilp]
            print(il)
        gqqp = 0.5 * gqqp
        exit()
        for jax in range(3*nat):
            # effective force
            F_lq_lqp[0,jax,:] = eiqR[jax] * euq[jax,il] * F_lq_lqp[0,jax,:] / np.sqrt(m_ia)
            F_lq_lqp[2,jax,:] = eiqR[jax] * euq[jax,il] * F_lq_lqp[2,jax,:] / np.sqrt(m_ia)
            F_lq_lqp[1,jax,:] = np.conj(eiqR[jax]) * np.conj(euq[jax,il]) * F_lq_lqp[1,jax,:] / np.sqrt(m_ia)
            F_lq_lqp[3,jax,:] = np.conj(eiqR[jax]) * np.conj(euq[jax,il]) * F_lq_lqp[3,jax,:] / np.sqrt(m_ia)
            # [eV^0.5/ang/ps *ang/eV^0.5/ps] = 1/ps^2
        return gqqp
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
        print('FXXp ', np.max(FXXp.real))
        FXXp = mpi.collect_array(FXXp)
        nan_indices = np.isnan(FXXp)
        assert nan_indices.any() == False
        return FXXp