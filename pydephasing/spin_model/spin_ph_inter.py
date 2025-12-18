import numpy as np
import cmath
from abc import ABC
from pydephasing.atomic_list_struct import atoms
from pydephasing.set_param_object import p
from pydephasing.parallelization.mpi import mpi
from pydephasing.common.phys_constants import hbar, THz_to_ev
from pydephasing.common.matrix_operations import compute_matr_elements
from pydephasing.utilities.log import log
#
# set of routines 
# for spin phonon dephasing class
#
class SpinPhononClass(ABC):
    def __init__(self):
        # add HFI contribution to the spin-phonon coupl
        self.HFI_CALC = None
        # add ZFS contribution to the spin-phonon coupl.
        self.ZFS_CALC = None
    # set up < qs1 | S grad_ax D S | qs2 > coefficients
    # n X n matrix -> n: number states Hsp
    def set_gaxD_force(self, gradZFS, Hsp):
        n = len(Hsp.basis_vectors)
        nat = gradZFS.struct_0.nat
        # Hsp = S D S
        Fax = np.zeros((n, n, 3*nat), dtype=np.complex128)
        # jax list
        jax_list = mpi.split_list(range(3*nat))
        # run over jax index
        for jax in jax_list:
            SgDS = np.zeros((n,n), dtype=np.complex128)
            gD = np.zeros((3,3))
            gD[:,:] = gradZFS.U_gradD_U[jax,:,:]
            SgDS =  gD[0,0] * np.matmul(Hsp.Sx, Hsp.Sx)
            SgDS += gD[0,1] * np.matmul(Hsp.Sx, Hsp.Sy)
            SgDS += gD[1,0] * np.matmul(Hsp.Sy, Hsp.Sx)
            SgDS += gD[1,1] * np.matmul(Hsp.Sy, Hsp.Sy)
            SgDS += gD[0,2] * np.matmul(Hsp.Sx, Hsp.Sz)
            SgDS += gD[2,0] * np.matmul(Hsp.Sz, Hsp.Sx)
            SgDS += gD[1,2] * np.matmul(Hsp.Sy, Hsp.Sz)
            SgDS += gD[2,1] * np.matmul(Hsp.Sz, Hsp.Sy)
            SgDS += gD[2,2] * np.matmul(Hsp.Sz, Hsp.Sz)
            # compute matrix elements
            for i1 in range(len(Hsp.qs)):
                qs1 = Hsp.qs[i1]['eigv']
                for i2 in range(len(Hsp.qs)):
                    qs2 = Hsp.qs[i2]['eigv']
                    # <qs1|SgDS|qs2>
                    Fax[i1,i2,jax] = compute_matr_elements(SgDS, qs1, qs2)
        mpi.comm.Barrier()
        Fax = mpi.collect_array(Fax)
        # THz / Ang units
        Fax = Fax * 2.*np.pi * hbar
        # eV / Ang units (energy)
        # add 2pi factor
        return Fax
    # set < qs | I(aa) grad_ax Ahf(aa) S | qs > coefficients
    def set_gaxAhf(self, aa, gradHFI, Hsp, Iaa):
        # nat
        nat = gradHFI.struct_0.nat
        n = len(Hsp.basis_vectors)
        # eff. force
        Fax = np.zeros((n, n, 3*nat), dtype=np.complex128)
        # compute : gax_Ahf
        U_gahf_U = np.zeros((3*nat, 3, 3))
        U_gahf_U[:,:,:] = gradHFI.U_gradAhfi_U[:,aa,:,:]
        # THz / ang
        sx = np.zeros((n, n), dtype=np.complex128)
        sy = np.zeros((n, n), dtype=np.complex128)
        sz = np.zeros((n, n), dtype=np.complex128)
        for i1 in range(n):
            qs1 = Hsp.qs[i1]['eigv']
            for i2 in range(n):
                qs2 = Hsp.qs[i2]['eigv']
                sx[i1,i2] = compute_matr_elements(Hsp.Sx, qs1, qs2)
                sy[i1,i2] = compute_matr_elements(Hsp.Sy, qs1, qs2)
                sz[i1,i2] = compute_matr_elements(Hsp.Sz, qs1, qs2)
        # iterate jax : 0, 3*nat
        for jax in range(3*nat):
            gax_ahf = np.zeros((3,3))
            gax_ahf[:,:] = U_gahf_U[jax,:,:]
            I_gA = np.einsum("i,ij->j", Iaa, gax_ahf)
            Fax[:,:,jax] = I_gA[0] * sx[:,:] + I_gA[1] * sy[:,:] + I_gA[2] * sz[:,:]
        return Fax
    #
    #  set hyperfine effective force from
    #  nuclear spins
    def set_Fax_hfi(self, gradHFI, Hsp, sp_config):
        nat = gradHFI.struct_0.nat
        n = len(Hsp.basis_vectors)
        # effective : S gA S
        Fax = np.zeros((n, n, 3*nat), dtype=np.complex128)
        isp_list = mpi.split_list(range(sp_config.nsp))
        # run over spin index
        for isp in isp_list:
            # spin site
            aa = sp_config.nuclear_spins[isp]['site']
            Iaa= sp_config.nuclear_spins[isp]['I']
            # compute effective force
            # spin site number 1 - nat
            Fax += self.set_gaxAhf(aa, gradHFI, Hsp, Iaa)
        # collect data
        Fax = mpi.collect_array(Fax) * 2.*np.pi * hbar
        # eV / ang
        return Fax
    #
    # compute g_{ab}(q,l)
    # = \sum_{n;s} Aq e^{iq Rn}e_q(s) <a|g_(ns)H|b>
    #
    def compute_gql(self, nat, ql_list, qgr, ph, Hsp, Fax):
        n = len(Hsp.basis_vectors)
        g_ql = np.zeros((n, n, len(ql_list)), dtype=np.complex128)
        # ph. amplitude
        A_ql = ph.compute_ph_amplitude_q(nat, ql_list)
        iql = 0
        for iq, il in ql_list:
            qv = qgr.qpts[iq]
            for iL in range(atoms.supercell_size):
                Rn = atoms.supercell_grid[iL]
                for jax in range(3*nat):
                    ia = atoms.index_to_ia_map[jax]
                    m_ia = atoms.atoms_mass[ia]             
                    # eV ps^2/A^2
                    eq = ph.eql[iq][jax,il] / np.sqrt(m_ia)
                    eiqRn = cmath.exp(1j*2.*np.pi*np.dot(qv, Rn))
                    g_ql[:,:,iql] += A_ql[iql] * eiqRn * eq * Fax[:,:,jax]
                    # [eV/ang * ang/eV^1/2 *ps^-1 * eV^1/2 ps]
                    # = eV
            iql += 1
        return g_ql
        
#
# first order spin-phonon coupling class
#
class SpinPhononFirstOrder(SpinPhononClass):
    def __init__(self, ZFS_CALC, HFI_CALC):
        super(SpinPhononFirstOrder, self).__init__()
        self.ZFS_CALC = ZFS_CALC
        self.HFI_CALC = HFI_CALC
    #
    # compute spin phonon coupling
    # at first order
    # g_ql = <s1|gX Hsp|s2> e_ql(X)
    def compute_spin_ph_coupl(self, nat, Hsp, ph, qgr, interact_dict, sp_config=None):
        # n. spin states
        n = len(Hsp.basis_vectors)
        Fax = np.zeros((n, n, 3*nat), dtype=np.complex128)
        # ZFS call
        if self.ZFS_CALC:
            gradZFS = interact_dict['gradZFS']
            Fax += self.set_gaxD_force(gradZFS, Hsp)
        # HFI call
        if self.HFI_CALC:
            gradHFI = interact_dict['gradHFI']
            Fax += self.set_Fax_hfi(gradHFI, Hsp, sp_config)
        # build ql_list
        ql_list = mpi.split_ph_modes(qgr.nq, ph.nmodes)
        # compute g_ql
        self.g_ql = self.compute_gql(nat, ql_list, qgr, ph, Hsp, Fax)
        self.ql_list = ql_list
        nan_indices = np.isnan(self.g_ql)
        assert nan_indices.any() == False

#
# second order spin-phonon coupling class
#
class SpinPhononSecndOrder00(SpinPhononClass):
    def __init__(self):
        super(SpinPhononSecndOrder00, self).__init__()
        # <qs1| grad_ax grad_a'x' D S|qs2> -> (3*nat,3*nat) matrix
        self.Fzfs_axby = None
        # Hf coupling 2nd order gradient
        self.Fhf_axby = None
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
    # set < qs | I(aa) grad_axby Ahf(aa) S | qs > coefficients
    def set_gaxbyA_force(self, aa, grad2HFI, Hsp, Iaa, displ_structs):
        # nat
        nat = grad2HFI.struct_0.nat
        # eff. force
        Faxby = np.zeros((3*nat, 3*nat), dtype=np.complex128)
        # compute gaxby_Ahf
        U_grad2_ahf_U = grad2HFI.evaluate_at_resolved_U_grad2ahf_U(aa, nat, displ_structs)
        # spin vector
        sv = np.zeros(3, dtype=np.complex128)
        sv[0] = self.compute_matr_elements(Hsp.Sx, self.qs1, self.qs0)
        sv[1] = self.compute_matr_elements(Hsp.Sy, self.qs1, self.qs0)
        sv[2] = self.compute_matr_elements(Hsp.Sz, self.qs1, self.qs0)
        # iterate (iax,iby)
        for jax in range(3*nat):
            for jby in range(3*nat):
                gaxby_ahf = np.zeros((3,3))
                gaxby_ahf[:,:] = U_grad2_ahf_U[jax,jby,:,:]
                # THz / ang^2
                I_g2A = np.einsum("i,ij->j", Iaa, gaxby_ahf)
                Faxby[jax,jby] = np.dot(I_g2A, sv)
        return Faxby