# This is the main subroutine for
# the calculation of the ZFS dephasing
# it computes the energy autocorrelation function
# and return it for further processing
import logging
import numpy as np
from pydephasing.mpi import mpi
from pydephasing.log import log
from pydephasing.input_parameters import p
from pydephasing.set_structs import DisplacedStructs, DisplacedStructures2ndOrder
from pydephasing.gradient_interactions import gradient_ZFS, gradient_2nd_ZFS
from pydephasing.atomic_list_struct import atoms
from pydephasing.spin_hamiltonian import spin_hamiltonian
from pydephasing.spin_ph_inter import SpinPhononDephClass, SpinPhononRelaxClass
from pydephasing.extract_ph_data import extract_ph_data
from pydephasing.ph_ampl_module import PhononAmplitude
from pydephasing.auto_correlation_module import acf_ph_deph
from pydephasing.T2_classes import T2i_ofT, Delta_ofT, tauc_ofT
#
def compute_homo_dephas():
    # main driver code for the calculation of dephasing time
    # in homogeneous spin systems
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
    # set ZFS gradient
    gradZFS = gradient_ZFS(p.work_dir, p.grad_info)
    gradZFS.set_gs_zfs_tensor()
    # compute tensor gradient
    gradZFS.compute_noise(struct_list)
    gradZFS.set_tensor_gradient(struct_list)
    # set ZFS gradient in quant. axis coordinates
    gradZFS.set_UgradDU_tensor()
    # n. atoms
    nat = gradZFS.struct_0.nat
    # compute index maps
    atoms.compute_index_to_ia_map(nat)
    atoms.compute_index_to_idx_map(nat)
    # set atoms dict
    atoms.extract_atoms_coords(nat)
    # zfs 2nd order
    if p.order_2_correct:
        # set 2nd order tensor
        grad2ZFS = gradient_2nd_ZFS(p.work_dir, p.grad_info)
        grad2ZFS.set_gs_zfs_tensor()
        # set secon order grad
        grad2ZFS.compute_2nd_order_gradients(struct_list_2nd)
    # debug mode
    if mpi.rank == mpi.root:
        if log.level <= logging.DEBUG:
            log.debug(" checking ZFS gradients")
            gradZFS.plot_tensor_grad_component(struct_list)
        # print data
        if log.level <= logging.INFO:
            gradZFS.write_gradDtensor_to_file(p.write_dir)
            if p.order_2_correct:
                grad2ZFS.write_grad2Dtensor_to_file(p.write_dir)
        if log.level <= logging.DEBUG and p.order_2_correct:
            grad2ZFS.check_tensor_coefficients()
    mpi.comm.Barrier()
    # set up spin phonon interaction class
    if p.deph:
        sp_ph_inter = SpinPhononDephClass()
        # set up quantum states
        p.qs1 = np.array([1.0+0j,0j,0j])
        p.qs2 = np.array([0j,1.0+0j,0j])
    elif p.relax:
        sp_ph_inter = SpinPhononRelaxClass()
    else:
        log.error("type of calculation must be deph or relax")
    # normalization
    p.qs1[:] = p.qs1[:] / np.sqrt(np.dot(p.qs1.conjugate(), p.qs1))
    p.qs2[:] = p.qs2[:] / np.sqrt(np.dot(p.qs2.conjugate(), p.qs2))
    # set up the spin Hamiltonian
    Hsp = spin_hamiltonian()
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
    # set the effective phonon forces
    # F_ax = <1|S Grad_ax D S|1> - <0|S Grad_ax D S|0>
    # F should be in eV/ang units
    sp_ph_inter.set_Fax_zfs(gradZFS, Hsp)
    Fax = sp_ph_inter.Fzfs_ax
    if p.order_2_correct:
        # F_axby = <1|S Grad_ax,by D S|1> - <0|S Grad_ax,by D S|0>
        # F should be in eV/ang^2 units
        sp_ph_inter.set_Faxby_zfs(grad2ZFS, Hsp)
        Faxby = sp_ph_inter.Fzfs_axby
        # eV / ang^2
    else:
        Faxby = None
    #
    # prepare calculation over q pts.
    # and ph. modes
    #
    ql_list = mpi.split_ph_modes(nq, 3*nat)
    #
    # compute acf over local (q,l) list
    acf = acf_ph_deph().generate_instance()
    acf.compute_acf(wq, wu, u, qpts, nat, Fax, Faxby, ql_list)
    #
    # collect data from processes
    if mpi.size > 1:
        acf.collect_acf_from_processes(nat)
    #
    # print average atom displ
    if log.level <= logging.DEBUG:
        #
        ph_ampl = PhononAmplitude(nat)
        # run over q pts
        #
        iq_list = mpi.split_list(range(nq))
        for iq in iq_list:
            # w(q) data (THz)
            wuq = wu[iq]
            # eu(q) data
            euq = u[iq]
            # q vector
            qv = qpts[iq]
            # compute atoms displ. first
            ph_ampl.compute_ph_amplq(euq, wuq, nat)
            ph_ampl.update_atom_displ(wq[iq], qv, wuq, nat)
            #
        if mpi.size > 1:
            ph_ampl.collect_displ_between_proc(nat)
        # save data on file
        ph_ampl.print_atom_displ(p.write_dir)
    mpi.comm.Barrier()
    # prepare data arrays
    T2_obj = T2i_ofT(nat)
    Delt_obj= Delta_ofT(nat)
    tauc_obj= tauc_ofT(nat)
    # run over temperature list
    for iT in range(p.ntmp):
        # extract dephasing data
        ft = acf.extract_dephas_data(T2_obj, Delt_obj, tauc_obj, iT)
        ft_inp = np.zeros((p.nt,2))
        ft_inp[:,0] = ft[0][:]
        ft_inp[:,1] = ft[1][:]
        #
        # if atom resolved calc.
        #
        ft_atr = None
        if p.at_resolved:
            # make local list of atoms
            atr_list = mpi.split_list(range(nat))
            # run over separated atoms
            ft_atr = np.zeros((p.nt2,nat,2))
            for ia in atr_list:
                # compute acf + T2 times + print acf data
                ft_ia = acf.extract_dephas_data_atr(T2_obj, Delt_obj, tauc_obj, ia, iT)
                if ft_ia is not None:
                    ft_atr[:,ia,0] = ft_ia[0][:]
                    ft_atr[:,ia,1] = ft_ia[1][:]
            ft_atr = mpi.collect_array(ft_atr)
            # collect data in single proc.
            T2_obj.collect_atr_from_other_proc(iT)
            Delt_obj.collect_atr_from_other_proc(iT)
            tauc_obj.collect_atr_from_other_proc(iT)
        #
        # if ph. resolved calc.
        #
        ft_phr = None
        if p.ph_resolved:
            # make local list of modes
            local_ph_list = mpi.split_list(p.phm_list)
            # run over modes
            ft_phr = np.zeros((p.nt2,p.nphr,2))
            for im in local_ph_list:
                iph = p.phm_list.index(im)
                # compute acf + T2 times + print acf data
                ft_iph = acf.extract_dephas_data_phr(T2_obj, Delt_obj, tauc_obj, iph, iT)
                if ft_iph is not None:
                    ft_phr[:,iph,0] = ft_iph[0][:]
                    ft_phr[:,iph,1] = ft_iph[1][:]
            ft_phr = mpi.collect_array(ft_phr)
            # collect data into single proc.
            T2_obj.collect_phr_from_other_proc(iT)
            Delt_obj.collect_phr_from_other_proc(iT)
            tauc_obj.collect_phr_from_other_proc(iT)
        #
        # print acf
        if mpi.rank == mpi.root:
            acf.print_autocorrel_data(ft_inp, ft_atr, ft_phr, iT)
        # wait all processes
        mpi.comm.Barrier()
    return T2_obj, Delt_obj, tauc_obj