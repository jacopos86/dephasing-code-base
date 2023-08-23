# This is the main subroutine for
# the calculation of the ZFS+HFI dephasing
# it computes the energy autocorrelation function
# and return it for further processing
from pydephasing.input_parameters import p
from pydephasing.set_structs import DisplacedStructs, DisplacedStructures2ndOrder
from pydephasing.gradient_interactions import gradient_ZFS, gradient_HFI, gradient_2nd_ZFS, gradient_2nd_HFI
from pydephasing.spin_hamiltonian import spin_hamiltonian
from pydephasing.spin_ph_inter import SpinPhononClass
from pydephasing.atomic_list_struct import atoms
from pydephasing.extract_ph_data import extract_ph_data
from pydephasing.auto_correlation_driver import acf_ph
from pydephasing.mpi import mpi
from pydephasing.log import log
import sys
# function
def compute_full_dephas():
    # main driver code for the calculation of full dephasing time
    # homogeneous + inhomogeneous spin systems
    #
    # create displaced structures
    struct_list = []
    for i in range(len(p.displ_poscar_dir)):
        displ_struct = DisplacedStructs(p.displ_poscar_dir[i], p.displ_outcar_dir[i])
        # set atomic displ. in the structure
        displ_struct.atom_displ(p.atoms_displ[i])      # Ang
        # append to list
        struct_list.append(displ_struct)
    # 2nd order displ structs
    if p.order_2_correct:
        struct_list_2nd = []
        for i in range(len(p.displ_2nd_poscar_dir)):
            displ_struct = DisplacedStructures2ndOrder(p.displ_2nd_poscar_dir[i], p.displ_2nd_outcar_dir[i])
            # set atomic displ. in the structure
            displ_struct.atom_displ(p.atoms_2nd_displ[i]) # Ang
            # append to list
            struct_list_2nd.append(displ_struct)
    # ----------------------------------------------
    #    set ZFS gradient
    # ----------------------------------------------
    gradZFS = gradient_ZFS(p.work_dir, p.grad_info)
    gradZFS.set_gs_zfs_tensor()
    # compute tensor gradient
    gradZFS.compute_noise(struct_list)
    gradZFS.set_tensor_gradient(struct_list)
    # set ZFS gradient in quant. axis coordinates
    gradZFS.set_UgradDU_tensor()
    # ---------------------------------------------
    # set HFI gradient
    # ---------------------------------------------
    gradHFI = gradient_HFI(p.work_dir, p.grad_info, p.fc_core)
    # set hyperfine interaction
    gradHFI.set_gs_hfi_tensor()
    # set the gradient
    gradHFI.set_tensor_gradient(struct_list)
    # set gradient in quant. vector basis
    gradHFI.set_U_gradAhfi_U_tensor()
    # n. atoms
    nat = gradHFI.struct_0.nat
    # set index maps
    atoms.compute_index_to_ia_map(nat)
    atoms.compute_index_to_idx_map(nat)
    # extract atom positions
    atoms.extract_atoms_coords(nat)
    # 2nd order
    if p.order_2_correct:
        # --------------------------------------------------
        # set 2nd order ZFS tensor
        # --------------------------------------------------
        grad2ZFS = gradient_2nd_ZFS(p.work_dir, p.grad_info)
        grad2ZFS.set_gs_zfs_tensor()
        # set secon order grad
        grad2ZFS.compute_2nd_order_gradients(struct_list_2nd)
        # -------------------------------------------------
        # set 2nd HFI tensor 
        # -------------------------------------------------
        grad2HFI = gradient_2nd_HFI(p.work_dir, p.grad_info, p.fc_core)
        grad2HFI.set_gs_hfi_tensor()
        # compute 2nd order grad. HFI
        grad2HFI.extract_2nd_order_gradients()
    # set up the spin Hamiltonian
    Hsp = spin_hamiltonian()
    Hsp.set_zfs_levels(gradZFS.struct_0, p.B0)
    # set up spin phonon interaction class
    sp_ph_inter = SpinPhononClass()
    sp_ph_inter.generate_instance()
    sp_ph_inter.set_quantum_states(Hsp)
    #
    # extract phonon data
    #
    u, wu, nq, qpts, wq, mesh = extract_ph_data()
    #
    if mpi.rank == 0:
        log.info("nq: " + str(nq))
        log.info("mesh: " + str(mesh))
        log.info("wq: " + str(wq))
    assert len(qpts) == nq
    assert len(u) == nq
    mpi.comm.Barrier()
    # if w_resolved define freq. grid
    if p.w_resolved:
        p.set_w_grid(wu)
    # set the effective phonon forces
    # F_ax = <1|S Grad_ax D S|1> - <0|S Grad_ax D S|0>
    # F should be in eV/ang units
    sp_ph_inter.set_Fax_zfs(gradZFS, Hsp)
    Fzfs_ax = sp_ph_inter.Fzfs_ax
    # 2nd order forces
    if p.order_2_correct:
        # F_axby = <1|S Grad_ax,by D S|1> - <0|S Grad_ax,by D S|0> (dephasing)
        # F_axby = <1|S Grad_ax,by D S|0> (relax)
        # F should be in eV/ang^2 units
        sp_ph_inter.set_Faxby_zfs(grad2ZFS, Hsp)
        Fzfs_axby = sp_ph_inter.Fzfs_axby
        # eV / ang^2
    # set q pts. grid
    if p.ph_resolved:
        p.set_wql_grid(wu, nq, nat)
    #
    # prepare calculation over q pts.
    # and ph. modes
    #
    ql_list = mpi.split_ph_modes(nq, 3*nat)
    #
    # set ACF 
    acf = acf_ph().generate_instance()
    sys.exit()