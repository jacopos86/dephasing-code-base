# This is the main subroutine
# for non Markovian calculation of decoherence
# times
# -> evolve delta\rho(1)q
# -> if 2nd order correction add lindblad for two-phonons term
# if evolve coherently the additional terms
import os
from pydephasing.mpi import mpi
from pydephasing.set_param_object import p
from pydephasing.set_structs import DisplacedStructs, DisplacedStructures2ndOrder
from pydephasing.gradient_interactions import gradient_ZFS
from pydephasing.atomic_list_struct import atoms
#
def compute_nmark_dephas():
    # main driver of the calculation
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
    # check restart exist otherwise create
    if not os.path.isdir(p.work_dir+'/restart'):
        if mpi.rank == mpi.root:
            os.mkdir(p.work_dir+'/restart')
        mpi.comm.Barrier()
    # set ZFS gradient
    gradZFS = gradient_ZFS(p.work_dir, p.grad_info)
    gradZFS.set_gs_zfs_tensor()
    # compute tensor gradient
    gradZFS.compute_noise(struct_list)
    gradZFS.set_tensor_gradient(struct_list)
    # set ZFS gradient in quant. axis coordinates
    gradZFS.set_UgradDU_tensor()
    mpi.comm.Barrier()
    # save data to restart
    if mpi.rank == mpi.root:
        gradZFS.write_gradDtensor_to_file(p.work_dir+'/restart')
    mpi.comm.Barrier()
    # n. atoms
    nat = gradZFS.struct_0.nat
    # compute index maps
    atoms.compute_index_to_ia_map(nat)
    atoms.compute_index_to_idx_map(nat)
    # set atoms dict
    atoms.extract_atoms_coords(nat)
    atoms.set_supercell_coords(nat)