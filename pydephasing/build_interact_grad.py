#
#  call this function
#  to compute interaction gradients
import logging
from pydephasing.set_param_object import p
from pydephasing.mpi import mpi
from pydephasing.log import log
from pydephasing.set_structs import DisplacedStructs, DisplacedStructures2ndOrder
from pydephasing.gradient_interactions import gradient_ZFS, gradient_HFI, gradient_2nd_HFI
from pydephasing.gradient_interactions import generate_2nd_orderZFS_grad_instance

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
    if mpi.rank == mpi.root:
        log.info("\n")
        log.info("\t " + p.sep)
        log.info("\t HESSIAN CALCULATION: " + str(p.hessian))
        log.info("\t " + p.sep)
        log.info("\n")
    # 2nd order displ structs
    if p.hessian:
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
            gradZFS.write_gradDtensor_to_file(p.write_dir+'/restart')
        mpi.comm.Barrier()
        # zfs 2nd order
        if p.hessian:
            # set 2nd order tensor
            grad2ZFS = generate_2nd_orderZFS_grad_instance(p.work_dir, p.grad_info)
            # set secon order grad
            grad2ZFS.compute_2nd_order_gradients(struct_list_2nd)
            mpi.comm.Barrier()
            # save data to restart
            if mpi.rank == mpi.root:
                grad2ZFS.write_hessDtensor_to_file(p.write_dir+'/restart')
        mpi.comm.Barrier()
        # debug mode
        if mpi.rank == mpi.root:
            if log.level <= logging.DEBUG:
                log.debug(" checking ZFS gradients")
                gradZFS.plot_tensor_grad_component(struct_list)
                if p.hessian:
                    log.debug(" checking ZFS hessian")
                    grad2ZFS.check_tensor_coefficients()
    if HFI_CALC:
        if mpi.rank == mpi.root:
            log.info("\n")
            log.info("\t " + p.sep)
            log.info("\t HF fc core: " + str(p.fc_core))
            log.info("\t " + p.sep)
            log.info("\n")
        gradHFI = gradient_HFI(p.work_dir, p.grad_info, p.fc_core)
        # compute tensor gradient
        gradHFI.compute_noise(struct_list)
        gradHFI.set_tensor_gradient(struct_list)
        # set gradient in quant. vector basis
        gradHFI.set_U_gradAhfi_U_tensor()
        mpi.comm.Barrier()
        # save data to restart
        if mpi.rank == mpi.root:
            gradHFI.write_gradHtensor_to_file(p.write_dir+'/restart')
        mpi.comm.Barrier()
        # hfi 2nd order
        if p.hessian:
            # set 2nd order tensor
            grad2HFI = gradient_2nd_HFI(p.work_dir, p.grad_info, p.fc_core)
    #
    # build interaction dictionary
    dict = {'gradZFS': gradZFS, 'grad2ZFS': grad2ZFS, 'gradHFI': gradHFI, 'grad2HFI':grad2HFI}
    return dict