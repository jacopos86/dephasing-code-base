import numpy as np
import cmath
import sys
from pydephasing.atomic_list_struct import atoms
from pydephasing.set_param_object import p
from pydephasing.mpi import mpi
from pydephasing.phys_constants import hbar
from pydephasing.log import log
#
# set of routines 
# for spin phonon dephasing class
#
class SpinPhononClass:
    def __init__(self):
        # <qs1|S grad_ax D S|qs2> -> (3,3,3*nat) matrix
        self.Fzfs_ax = None
        # <qs1| grad_ax grad_a'x' D S|qs2> -> (3*nat,3*nat) matrix
        self.Fzfs_axby = None
        # Hf coupling tensor force
        self.Fhf_ax = None
        # Hf coupling 2nd order gradient
        self.Fhf_axby = None
        # quantum basis spin triplet
        self.qs0 = np.zeros(3, dtype=np.complex128)
        # | 1 >
        self.qs1 = np.zeros(3, dtype=np.complex128)
    def generate_instance(self):
        if p.deph:
            return SpinPhononDephClass()
        elif p.relax:
            return SpinPhononRelaxClass()
        else:
            log.info("\n")
            log.error("\t either relax or deph calculation")
    def set_quantum_states(self, Hsp):
        expv = np.zeros(3)
        for j in range(3):
            expv[j] = self.compute_matr_elements(Hsp.Sz, Hsp.qs[:,j], Hsp.qs[:,j])
        index = np.argsort(expv)
        self.qs1[:] = Hsp.qs[:,index[2]]
        self.qs0[:] = Hsp.qs[:,index[1]]
        # index quantum states
        p.index_qs1 = index[2]
        p.index_qs0 = index[1]
    # compute matrix elements
    # <qs1|A|qs2>
    def compute_matr_elements(self, A, qs1, qs2):
        r = np.einsum("ij,j->i", A, qs2)
        expv = np.einsum("i,i", qs1.conjugate(), r)
        return expv
    # set up < qs | S grad_ax D S | qs > coefficients
    # 3 X 3 matrix
    def set_gaxD_force(self, gradZFS, Hsp):
        nat = gradZFS.struct_0.nat
        # Hsp = S D S
        Fax = np.zeros((3, 3, 3*nat), dtype=np.complex128)
        # jax list
        jax_list = mpi.split_list(range(3*nat))
        # run over jax index
        for jax in jax_list:
            SgDS = np.zeros((3,3), dtype=np.complex128)
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
            # <qs0|SgDS|qs0>
            Fax[0,0,jax] = self.compute_matr_elements(SgDS, Hsp.qs[:,0], Hsp.qs[:,0])
			# <qs1|SgDS|qs1>
            Fax[1,1,jax] = self.compute_matr_elements(SgDS, Hsp.qs[:,1], Hsp.qs[:,1])
            # <qs0|SgDS|qs1>
            Fax[0,1,jax] = self.compute_matr_elements(SgDS, Hsp.qs[:,0], Hsp.qs[:,1])
            Fax[1,0,jax] = self.compute_matr_elements(SgDS, Hsp.qs[:,0], Hsp.qs[:,1]).conjugate()
            # <qs0|SgDS|qs2>
            Fax[0,2,jax] = self.compute_matr_elements(SgDS, Hsp.qs[:,0], Hsp.qs[:,2])
            Fax[2,0,jax] = self.compute_matr_elements(SgDS, Hsp.qs[:,0], Hsp.qs[:,2]).conjugate()
            #
            Fax[1,2,jax] = self.compute_matr_elements(SgDS, Hsp.qs[:,1], Hsp.qs[:,2])
            Fax[2,1,jax] = self.compute_matr_elements(SgDS, Hsp.qs[:,1], Hsp.qs[:,2]).conjugate()
            #
            Fax[2,2,jax] = self.compute_matr_elements(SgDS, Hsp.qs[:,2], Hsp.qs[:,2])
        mpi.comm.Barrier()
        Fax = mpi.collect_array(Fax)
        # THz / Ang units
        return Fax
    # set < qs | I(aa) grad_ax Ahf(aa) S | qs > coefficients
    def set_gaxA_force(self, aa, gradHFI, Hsp, Iaa):
        # nat
        nat = gradHFI.struct_0.nat
        # eff. force
        Fax = np.zeros((3, 3, 3*nat), dtype=np.complex128)
        # compute : gax_Ahf
        U_gahf_U = np.zeros((3*nat,3,3))
        U_gahf_U[:,:,:] = gradHFI.gradAhfi[:,aa,:,:]
        # THz / ang
        # set Sv matrix elements
        sx = np.zeros((3,3), dtype=np.complex128)
        sy = np.zeros((3,3), dtype=np.complex128)
        sz = np.zeros((3,3), dtype=np.complex128)
        for i in range(3):
            for j in range(3):
                sx[i,j] = self.compute_matr_elements(Hsp.Sx, Hsp.qs[:,i], Hsp.qs[:,j])
                sy[i,j] = self.compute_matr_elements(Hsp.Sy, Hsp.qs[:,i], Hsp.qs[:,j])
                sz[i,j] = self.compute_matr_elements(Hsp.Sz, Hsp.qs[:,i], Hsp.qs[:,j])
        # iterate iax
        for iax in range(3*nat):
            gax_ahf = np.zeros((3,3))
            gax_ahf[:,:] = U_gahf_U[iax,:,:]
            I_gA = np.einsum("i,ij->j", Iaa, gax_ahf)
            Fax[:,:,iax] = I_gA[0] * sx[:,:] + I_gA[1] * sy[:,:] + I_gA[2] * sz[:,:]
        return Fax
    #
    # set ZFS energy gradient 1st order -> spin dephasing
    def set_Fax_zfs(self, gradZFS, Hsp):
        nat = gradZFS.struct_0.nat
        # compute grad delta Ezfs
        # gradient of the spin state energy difference
        self.Fzfs_ax = np.zeros((3, 3, 3*nat), dtype=np.complex128)
        #
        # compute : < qs1 | S gradD S | qs1 > - < qs2 | S gradD S | qs2 >
        #
        Fax = self.set_gaxD_force(gradZFS, Hsp)
        # eV / Ang units (energy)
        # add 2pi factor
        self.Fzfs_ax[:,:,:] = Fax[:,:,:] * 2.*np.pi * hbar
    # set hyperfine dephasing
    # effective force
    def set_Fax_hfi(self, gradHFI, Hsp, sp_config):
        nat = gradHFI.struct_0.nat
        # effective force
        self.Fhf_ax = np.zeros((3, 3, 3*nat), dtype=np.complex128)
        Fax = np.zeros((3, 3, 3*nat), dtype=np.complex128)
        isp_list = mpi.split_list(range(p.nsp))
        # run over spin index
        for isp in isp_list:
            # spin site
            aa = sp_config.nuclear_spins[isp]['site']-1
            Iaa= sp_config.nuclear_spins[isp]['I']
            # compute effective force
            # spin site number 1 - nat
            Fax += self.set_gaxA_force(aa, gradHFI, Hsp, Iaa)
        # collect data
        self.Fhf_ax = mpi.collect_array(Fax) * 2.*np.pi * hbar
        # eV / ang
    #
    # set ZFS 2nd order gradient force
    def set_Faxby_zfs(self, grad2ZFS, Hsp):
        nat = grad2ZFS.struct_0.nat
        # compute grad2 delta Ezfs
        # laplacian spin state energy difference
        self.Fzfs_axby = np.zeros((3*nat, 3*nat), dtype=np.complex128)
        #
        # compute : < qs1 | S gradD S | qs1 > - < qs2 | S gradD S | qs2 >
        #
        Faxby = self.set_gaxbyD_force(grad2ZFS, Hsp)
		# eV / ang^2 units
        self.Fzfs_axby[:,:] = Faxby[:,:] * 2.*np.pi * hbar
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
class SpinPhononRelaxClass(SpinPhononClass):
    def __init__(self):
        super(SpinPhononRelaxClass, self).__init__()
    # set up < 1 | S grad_axby D S | 0 > coefficients
    # 2nd order spin phonon relaxation matrix elements
    def set_gaxbyD_force(self, grad2ZFS, Hsp):
        nat = grad2ZFS.struct_0.nat
        # partition jax between proc.
        jax_list = mpi.random_split(range(3*nat))
        # S g^2D S matrix elements
        Faxby = np.zeros((3*nat, 3*nat), dtype=np.complex128)
        for jax in jax_list:
            for jby in range(jax, 3*nat):
                SggDS = np.zeros((3,3), dtype=np.complex128)
                g2D = np.zeros((3,3))
                g2D[:,:] = grad2ZFS.U_grad2D_U[jax,jby,:,:]
                # THz / ang^2
                SggDS =  g2D[0,0] * np.matmul(Hsp.Sx, Hsp.Sx)
                SggDS += g2D[0,1] * np.matmul(Hsp.Sx, Hsp.Sy)
                SggDS += g2D[1,0] * np.matmul(Hsp.Sy, Hsp.Sx)
                SggDS += g2D[1,1] * np.matmul(Hsp.Sy, Hsp.Sy)
                SggDS += g2D[0,2] * np.matmul(Hsp.Sx, Hsp.Sz)
                SggDS += g2D[2,0] * np.matmul(Hsp.Sz, Hsp.Sx)
                SggDS += g2D[1,2] * np.matmul(Hsp.Sy, Hsp.Sz)
                SggDS += g2D[2,1] * np.matmul(Hsp.Sz, Hsp.Sy)
                SggDS += g2D[2,2] * np.matmul(Hsp.Sz, Hsp.Sz)
                #
                Faxby[jax,jby] = self.compute_matr_elements(SggDS, self.qs1, self.qs0)
                if jby != jax:
                    Faxby[jby,jax] = Faxby[jax,jby]
        # THz / ang^2 units
        # collect data into single proc.
        mpi.comm.Barrier()
        Faxby =  mpi.collect_array(Faxby)
        return Faxby
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
#
class SpinPhononDephClass(SpinPhononClass):
    def __init__(self):
        super(SpinPhononDephClass, self).__init__()
    # compute <qs1|A|qs1> - <qs2|A|qs2>
    def compute_deph_matr_elem_difference(self, A, qs1, qs2):
        r1 = np.einsum("ij,j->i", A, qs1)
        expv1 = np.einsum("i,i", qs1.conjugate(), r1)
        #
        r2 = np.einsum("ij,j->i", A, qs2)
        expv2 = np.einsum("i,i", qs2.conjugate(), r2)
        r = expv1 - expv2
        return r
    # set up < 0 | S grad_axby D S | 0 > - < 1 | S grad_axby D S | 1 > coefficients
    # 2nd order spin phonon dephasing matrix elements
    def set_gaxbyD_force(self, grad2ZFS, Hsp):
        nat = grad2ZFS.struct_0.nat
        # partition jax between proc.
        jax_list = mpi.random_split(range(3*nat))
        # S g^2D S matrix elements
        Faxby = np.zeros((3*nat, 3*nat), dtype=np.complex128)
        for jax in jax_list:
            for jby in range(jax, 3*nat):
                SggDS = np.zeros((3,3), dtype=np.complex128)
                g2D = np.zeros((3,3))
                g2D[:,:] = grad2ZFS.U_grad2D_U[jax,jby,:,:]
                # THz / ang^2
                SggDS =  g2D[0,0] * np.matmul(Hsp.Sx, Hsp.Sx)
                SggDS += g2D[0,1] * np.matmul(Hsp.Sx, Hsp.Sy)
                SggDS += g2D[1,0] * np.matmul(Hsp.Sy, Hsp.Sx)
                SggDS += g2D[1,1] * np.matmul(Hsp.Sy, Hsp.Sy)
                SggDS += g2D[0,2] * np.matmul(Hsp.Sx, Hsp.Sz)
                SggDS += g2D[2,0] * np.matmul(Hsp.Sz, Hsp.Sx)
                SggDS += g2D[1,2] * np.matmul(Hsp.Sy, Hsp.Sz)
                SggDS += g2D[2,1] * np.matmul(Hsp.Sz, Hsp.Sy)
                SggDS += g2D[2,2] * np.matmul(Hsp.Sz, Hsp.Sz)
                #
                r = self.compute_deph_matr_elem_difference(SggDS, self.qs0, self.qs1)
                Faxby[jax,jby] = r
                if jby != jax:
                    Faxby[jby,jax] = Faxby[jax,jby]
        # THz / ang^2 units
        # collect into single proc.
        mpi.comm.Barrier()
        Faxby = mpi.collect_array(Faxby)
        return Faxby
    # set < 0 | I(aa) grad_axby Ahf(aa) S | 0 > - < 1 | I(aa) grad_axby Ahf(aa) S | 1 >
    def set_gaxbyA_force(self, aa, grad2HFI, Hsp, Iaa, displ_structs):
        # nat
        nat = grad2HFI.struct_0.nat
        # eff. force
        Faxby = np.zeros((3*nat, 3*nat), dtype=np.complex128)
        # compute gaxby_Ahf
        U_grad2_ahf_U = grad2HFI.evaluate_at_resolved_U_grad2ahf_U(aa, nat, displ_structs)
        # spin vector
        Dsv = np.zeros(3, dtype=np.complex128)
        Dsv[0] = self.compute_deph_matr_elem_difference(Hsp.Sx, self.qs0, self.qs1)
        Dsv[1] = self.compute_deph_matr_elem_difference(Hsp.Sy, self.qs0, self.qs1)
        Dsv[2] = self.compute_deph_matr_elem_difference(Hsp.Sz, self.qs0, self.qs1)
        # iterate (jax,jby)
        for jax in range(3*nat):
            for jby in range(3*nat):
                gaxby_ahf = np.zeros((3,3))
                gaxby_ahf[:,:] = U_grad2_ahf_U[jax,jby,:,:]
                # THz / ang^2
                I_g2A = np.einsum("i,ij->j", Iaa, gaxby_ahf)
                Faxby[jax,jby] = np.dot(I_g2A, Dsv)
        return Faxby