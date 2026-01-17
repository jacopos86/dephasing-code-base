import numpy as np
import cmath
import collections
from itertools import product
from pathlib import Path
from pydephasing.common.phys_constants import hbar, mp, THz_to_ev
from pydephasing.common.matrix_operations import compute_matr_elements
from pydephasing.parallelization.GPU_arrays_handler import GPU_ARRAY
from pydephasing.atomic_list_struct import atoms
from pydephasing.set_param_object import p
from pydephasing.global_params import GPU_ACTIVE
from pydephasing.spin_model.spin_ph_inter import SpinPhononClass
from pydephasing.utilities.log import log
from pydephasing.parallelization.mpi import mpi

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
    #  compute forces
    def compute_forces(self, nat, Hsp, interact_dict, sp_config=None):
        n = len(Hsp.basis_vectors)
        Fax = np.zeros((n,n,3*nat), dtype=np.complex128)
        Faxby = None
        # ZFS calc.
        if self.ZFS_CALC:
            gradZFS = interact_dict['gradZFS']
            Fax += self.set_gaxD_force(gradZFS, Hsp)
        # HFI calc
        if self.HFI_CALC:
            gradHFI = interact_dict['gradHFI']
            Fax += self.set_Fax_hfi(gradHFI, Hsp, sp_config)
        # compute Hessian
        if self.hessian:
            Faxby = self.compute_hessian_term(interact_dict, Hsp)
        return Fax, Faxby
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
        # build ql_list
        ql_list = mpi.split_ph_modes(qgr.nq, ph.nmodes)
        # central cell approximation
        # compute g_ql
        g_ql = self.compute_gql(nat, ql_list, qgr, ph, Hsp, Fax)
        nan_indices = np.isnan(g_ql)
        assert nan_indices.any() == False
        print("max Fx", np.max(Fax.real))
        # collect g_ql
        self.collect_gql_and_indices(g_ql, ql_list)
        self.gq_at_iq(iq_target=26)
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
    # compute single gqqp coefficient -> this is needed for comparison
    #
    def compute_gqqp_l12(self, nat, iq, iqp, il1, il2, qgr, ph, H, Fx, Fxy):
        n = len(H.basis_vectors)
        g12 = np.zeros((n,n), dtype=np.complex128)
        # modes
        ql_list = [(iq, il1)]
        Aq1 = ph.compute_ph_amplitude_q(nat, ql_list)
        ql_list = [(iqp, il2)]
        Aq2 = ph.compute_ph_amplitude_q(nat, ql_list)
        # phase
        eiq1r = np.zeros(atoms.supercell_size, dtype=np.complex128)
        eiq2r = np.zeros(atoms.supercell_size, dtype=np.complex128)
        qv = qgr.qpts[iq]
        qpv = qgr.qpts[iqp]
        for iL in range(atoms.supercell_size):
            Rn = atoms.supercell_grid[iL]
            eiq1r[iL] = cmath.exp(1j*2.*np.pi*np.dot(qv,Rn))
            eiq2r[iL] = cmath.exp(1j*2.*np.pi*np.dot(qpv,Rn))
        # eig
        eig = np.zeros(n)
        for a in range(n):
            eig[a] = H.qs[a]['eig']
        # ph. vectors -> eq
        wq1 = ph.uql[iq][il1]*THz_to_ev
        wq2 = ph.uql[iqp][il2]*THz_to_ev
        eq1 = ph.eql[iq]
        eq2 = ph.eql[iqp]
        for jax in range(3*nat):
            ia = atoms.index_to_ia_map[jax]
            m_ia = atoms.atoms_mass[ia]
            eq1[jax,:] = eq1[jax,:] / np.sqrt(m_ia)
            eq2[jax,:] = eq2[jax,:] / np.sqrt(m_ia)
            # ang/eV^1/2 ps^-1
        # run over states
        for a in range(n):
            for ap in range(n):
                for jax in range(3*nat):
                    for jby in range(3*nat):
                        for n1 in range(atoms.supercell_size):
                            for n2 in range(atoms.supercell_size):
                                F = 0.
                                for b in range(n):
                                    F += Fx[a,b,jby] * Fx[b,ap,jax] * (1./(-wq2-eig[b]+eig[a]) + 1./(-wq1-eig[b]+eig[ap]))
                                    F += Fx[a,b,jax] * Fx[b,ap,jby] * (1./(wq1-eig[b]+eig[a]) + 1./(wq2-eig[b]+eig[ap]))
                                if Fxy is not None:
                                    F += Fxy[a,ap,jax,jby]
                                g12[a,ap] += 0.5 * Aq1 * eq1[jax,il1] * eiq1r[n1] * F * eiq2r[n2] * eq2[jby,il2] * Aq2
        return g12
    #
    # compute g_{ab}(q,qp)
    # = \sum_{nn';ss'} Aq e^{iq Rn}e_q(s) <a|H^(2)|b> Aqp e^{iqp Rn'}e_qp(s')
    #
    def set_gqqp_calculation(self, nat, qgr, ph, Hsp, Fax, Faxby):
        if not self.hessian:
            assert Faxby == None
        # make list of q vector pairs for each proc.
        qqp_list = qgr.build_irred_qqp_pairs()
        # parallelize calculation over (q,q')
        qqp_list = mpi.split_list(qqp_list)
        # compute g_qqp
        for iq, iqp in qqp_list:
            file_name = 'G-iq-' + str(iq) + '-iqp-' + str(iqp) + '.npz'
            file_path = p.write_dir + '/restart/' + file_name
            file_path = "{}".format(file_path)
            fil = Path(file_path)
            if not fil.exists():
                if not GPU_ACTIVE:
                    gqqp = self.compute_gqqp(nat, iq, iqp, qgr, ph, Hsp, Fax, FXXp0, Faxby)
                else:
                    gqqp = self.compute_gqqp(nat, iq, iqp, qgr, ph, Hsp, Fax, Faxby)
                # save data
                np.savez(file_path, G=gqqp)
            else:
                pass
            log.debug("\t " + p.sep)
            log.debug("\t iq " + str(iq) + " - iqp " + str(iqp) + " -> calculation complete")
            log.debug("\t " + p.sep)
            gpu.cleanup()
            exit()
        mpi.comm.Barrier()
        if mpi.rank == mpi.root:
            log.info("\n")
            log.info("\t " + p.sep)
            log.info("\t CALCULATION OF GQQP COMPLETE")
            log.info("\t " + p.sep)
            log.info("\n")
    #
    # read gqqp from file
    def read_gqqp_from_file(self, file_path):
        npz_file = np.load(file_path)
        if 'G' not in npz_file:
            log.error("'G' NOT FOUND IN: " + npz_file)
        gqqp = npz_file['G']
        return gqqp

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
    #  -------------------------------------------------------------------------
    #
    #       driver function
    #
    # --------------------------------------------------------------------------
    def compute_gqqp(self, nat, iq, iqp, qgr, ph, Hsp, Fax, Faxby=None):
        # FXXp units -> eV / ang^2
        n = len(Hsp.basis_vectors)
        # g_qq'^\pm : 
        # \pm -> 2 - n -> n. spin states - nmodes
        gqqp = np.zeros((2, n, n, ph.nmodes, ph.nmodes), dtype=np.complex128)
        GQQP = GPU_ARRAY(gqqp, np.complex128)
        log.info("\t rank: " + str(mpi.rank) + " -> iq=" + str(iq) + " - iqp=" + str(iqp))
        # load file
        gpu_mod = gpu.get_device_module("compute_two_phonons_matr.cu")
        # prepare input quantities
        NAT = np.int32(nat)
        NMD = np.int32(ph.nmodes)
        # phonon energies
        WQL = GPU_ARRAY(np.array(ph.uql[iq]) * THz_to_ev)
        WQPL= GPU_ARRAY(np.array(ph.uql[iqp]) * THz_to_ev)
        print(WQL.length(), WQPL.length())
        # energy eigenvalues array -> EIG
        eig = np.zeros(n)
        for a in range(n):
            eig[a] = Hsp.qs[a]['eig']
        EIG = GPU_ARRAY(eig)
        NST = EIG.length()
        exit()
        # TODO we need to compute gq at q and qp
        # Hessian term
        if Faxby is not None:
            if mpi.rank == mpi.root:
                log.info("\t max Faxby: " + str(np.max(Faxby.real)))
            FXXp = GPU_ARRAY(Faxby, np.complex128)
            compute_gqqp = gpu_mod.get_function("compute_gqqp_2nd_Raman")
        else:
            compute_gqqp = gpu_mod.get_function("compute_gqqp")
        # -> GPU parallelized arrays
        illp_list = np.array(list(product(range(ph.nmodes), range(ph.nmodes))))
        INIT_INDEX, SIZE_LIST = gpu.distribute_data_on_grid(illp_list)
        MODES_LIST = GPU_ARRAY(illp_list, np.int32)
        # set e^iqR
        EIQR = GPU_ARRAY(qgr.compute_phase_factor(iq, nat), np.complex128)
        EIQPR= GPU_ARRAY(qgr.compute_phase_factor(iqp, nat), np.complex128)
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
        EQ = GPU_ARRAY(eq, np.complex128)
        EQP = GPU_ARRAY(eqp, np.complex128)
        # amplitudes
        ql_list = list(product([iq], range(ph.nmodes)))
        AQL = GPU_ARRAY(ph.compute_ph_amplitude_q(ql_list))
        qpl_list = list(product([iqp], range(ph.nmodes)))
        AQPL = GPU_ARRAY(ph.compute_ph_amplitude_q(qpl_list))
        #  call GPU function
        if Faxby is None:
            compute_gqqp(NAT, NL, NST, NMD, INIT_INDEX.to_gpu(), SIZE_LIST.to_gpu(), MODES_LIST.to_gpu(),
                AQL.to_gpu(), AQPL.to_gpu(), WQL.to_gpu(), WQPL.to_gpu(), EIG.to_gpu(), 
                FX.to_gpu(), EQ.to_gpu(), EQP.to_gpu(), EIQR.to_gpu(), EIQPR.to_gpu(), 
                GQQP.to_gpu(allocate_only=True), block=gpu.block, grid=gpu.grid)
        else:
            compute_gqqp(NAT, NL, NST, NMD, INIT_INDEX.to_gpu(), SIZE_LIST.to_gpu(), MODES_LIST.to_gpu(),
                AQL.to_gpu(), AQPL.to_gpu(), WQL.to_gpu(), WQPL.to_gpu(), EIG.to_gpu(), 
                FX.to_gpu(), FXXp.to_gpu(), EQ.to_gpu(), EQP.to_gpu(), EIQR.to_gpu(), 
                EIQPR.to_gpu(), GQQP.to_gpu(allocate_only=True), block=gpu.block, grid=gpu.grid)
        # synchronize
        cuda.Context.synchronize()
        return GQQP.from_gpu()
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