#
#  call this function
#  to compute interaction gradients
import logging
from pydephasing.set_param_object import p
from pydephasing.mpi import mpi
from pydephasing.log import log
from pydephasing.set_structs import DisplacedStructs, DisplacedStructures2ndOrder
from pydephasing.gradient_interactions import gradient_ZFS, generate_2nd_order_grad_instance

def calc_interaction_grad(ZFS_CALC, HFI_CALC):
    gradZFS = None
    grad2ZFS = None
    gradHFI = None
    grad2HFI = None
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
    if ZFS_CALC:
        # set ZFS gradient
        gradZFS = gradient_ZFS(p.work_dir, p.grad_info)
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
        # zfs 2nd order
        if p.order_2_correct:
            # set 2nd order tensor
            grad2ZFS = generate_2nd_order_grad_instance(p.work_dir, p.grad_info)
            grad2ZFS.set_gs_zfs_tensor()
            # set secon order grad
            grad2ZFS.compute_2nd_order_gradients(struct_list_2nd)
            mpi.comm.Barrier()
            # save data to restart
            if mpi.rank == mpi.root:
                grad2ZFS.write_grad2Dtensor_to_file(p.work_dir+'/restart')
        mpi.comm.Barrier()
        # debug mode
        if mpi.rank == mpi.root:
            if log.level <= logging.DEBUG:
                log.debug(" checking ZFS gradients")
                gradZFS.plot_tensor_grad_component(struct_list)
                if p.order_2_correct:
                    grad2ZFS.check_tensor_coefficients()
    #
    # build interaction dictionary
    dict = {'gradZFS': gradZFS, 'grad2ZFS': grad2ZFS, 'gradHFI': gradHFI, 'grad2HFI':grad2HFI}
    return dict