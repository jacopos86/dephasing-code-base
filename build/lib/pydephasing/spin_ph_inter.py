import numpy as np
import cmath
import sys
from pydephasing.atomic_list_struct import atoms
from pydephasing.input_parameters import p
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
        # <qs1| grad_ax grad_a'x' D S|qs2> -> (3,3,3*nat,3*nat) matrix
        self.Fzfs_axby = None
    # set up < qs | S grad_ax D S | qs > coefficients
    def set_gaxD_force(self, gradZFS, Hsp):
        nat = gradZFS.struct_0.nat
        # qs1 -> |1>
        qs1 = np.array([1.+0j,0j,0j])
        # qs2 -> |0>
        qs2 = np.array([0j,1.+0j,0j])
        # qs3 -> |-1>
        qs3 = np.array([0j,0j,1.+0j])
        # Hsp = S D S
        Fax = np.zeros((3,3,3*nat), dtype=np.complex128)
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
            r1 = np.dot(SgDS, qs1)
            expv11 = np.dot(qs1.conjugate(), r1)
            Fax[0,0,jax] = expv11
			#
            r2 = np.dot(SgDS, qs2)
            expv22 = np.dot(qs2.conjugate(), r2)
            Fax[1,1,jax] = expv22
            #
            expv12 = np.dot(qs1.conjugate(), r2)
            Fax[0,1,jax] = expv12
            Fax[1,0,jax] = expv12.conjugate()
            #
            r3 = np.dot(SgDS, qs3)
            expv13 = np.dot(qs1.conjugate(), r3)
            Fax[0,2,jax] = expv13
            Fax[2,0,jax] = expv13.conjugate()
            #
            expv23 = np.dot(qs2.conjugate(), r3)
            Fax[1,2,jax] = expv23
            Fax[2,1,jax] = expv23.conjugate()
            #
            expv33 = np.dot(qs3.conjugate(), r3)
            Fax[2,2,jax] = expv33
        mpi.comm.Barrier()
        Fax = mpi.collect_array(Fax)
        # THz / Ang units
        return Fax
    # set up < qs | S grad_axby D S | qs > coefficients
    def set_gaxbyD_force(self, grad2ZFS, Hsp):
        nat = grad2ZFS.struct_0.nat
        # qs1 -> |0>
        qs1 = p.qs1
        # qs2 -> |1>
        qs2 = p.qs2
        # partition jax between cores
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
                r1 = np.dot(SggDS, qs1)
                expv1 = np.dot(qs1.conjugate(), r1)
                #
                r2 = np.dot(SggDS, qs2)
                expv2 = np.dot(qs2.conjugate(), r2)
                #
                Faxby[jax,jby] = expv1 - expv2
                if jby != jax:
                    Faxby[jby,jax] = Faxby[jax,jby].conjugate()
        # THz / ang^2 units
        # collect data into single proc.
        mpi.comm.Barrier()
        Faxby =  mpi.collect_array(Faxby)
        return Faxby
    # set < qs | I(aa) grad_ax Ahf(aa) S | qs > coefficients
    def set_gaxA_force(self, aa, gradHFI, Hsp, Iaa):
        # nat
        nat = gradHFI.struct_0.nat
        # eff. force
        Fax = np.zeros(3*nat, dtype=np.complex128)
        # compute : gax_Ahf
        U_gahf_U = np.zeros((3*nat,3,3))
        U_gahf_U[:,:,:] = gradHFI.gradAhfi[:,aa,:,:]
        # THz / ang
        # <Delta S>
        dS = Hsp.set_DeltaS(p.qs1, p.qs2)
        # iterate iax
        for iax in range(3*nat):
            gax_ahf = np.zeros((3,3))
            gax_ahf[:,:] = U_gahf_U[iax,:,:]
            I_gA = np.einsum("i,ij->j", Iaa, gax_ahf)
            Fax[iax] = np.dot(I_gA, dS)
        return Fax
    # set < qs | I(aa) grad_axby Ahf(aa) S | qs > coefficients
    def set_gaxbyA_force(self, aa, grad2HFI, Hsp, Iaa, displ_structs):
        # nat
        nat = grad2HFI.struct_0.nat
        # eff. force
        Faxby = np.zeros((3*nat, 3*nat), dtype=np.complex128)
        # compute gaxby_Ahf
        U_grad2_ahf_U = grad2HFI.evaluate_at_resolved_U_grad2ahf_U(aa, nat, displ_structs)
        # <Delta S>
        dS = Hsp.set_DeltaS(p.qs1, p.qs2)
        # iterate (iax,iby)
        for iax in range(3*nat):
            for iby in range(3*nat):
                gaxby_ahf = np.zeros((3,3))
                gaxby_ahf[:,:] = U_grad2_ahf_U[iax,iby,:,:]
                # THz / ang^2
                I_g2A = np.einsum("i,ij->j", Iaa, gaxby_ahf)
                Faxby[iax,iby] = np.dot(I_g2A, dS)
        return Faxby
    #
    # set ZFS energy gradient 1st order -> spin dephasing
    def set_Fax_zfs(self, gradZFS, Hsp):
        nat = gradZFS.struct_0.nat
        # compute grad delta Ezfs
        # gradient of the spin state energy difference
        self.Fzfs_ax = np.zeros((3,3,3*nat), dtype=np.complex128)
        #
        # compute : < qs1 | S gradD S | qs1 > - < qs2 | S gradD S | qs2 >
        #
        Fax = self.set_gaxD_force(gradZFS, Hsp)
        # eV / Ang units (energy)
        # add 2pi factor
        self.Fzfs_ax[:,:,:] = Fax[:,:,:] * 2.*np.pi * hbar
    # set ZFS energy 2nd order gradient
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
    # set hyperfine dephasing
    # effective force
    def set_Fax_hfi(self, gradHFI, Hsp, sp_config):
        nat = gradHFI.struct_0.nat
        # effective force
        self.Fhf_ax = np.zeros(3*nat, dtype=np.complex128)
        Fax = np.zeros(3*nat, dtype=np.complex128)
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
class SpinPhononRelaxClass():
    def __init__(self):
        self.Fzfs_ax = None
        self.Fzfs_axby = None
    # set up < qs1=0 | S grad_ax D S | qs2=1 > matrix coefficients
    # spin triplet -> 3 X 3 matrix
    def set_gaxD_force(self, gradZFS, Hsp):
        nat = gradZFS.struct_0.nat
        # qs1 -> |0>
        qs1 = p.qs1
        # qs2 -> |1>
        qs2 = p.qs2
        # Hsp = S D S
