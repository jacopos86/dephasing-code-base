# This is the main subroutine used
# for the calculation of the HFI component
# of the dephasing time
# it computes the relative autocorrelation function
# and returns it for further processing
import numpy as np
import os
from pydephasing.input_parameters import p
from pydephasing.set_structs import DisplacedStructs, DisplacedStructures2ndOrder
from pydephasing.spin_hamiltonian import spin_hamiltonian
from pydephasing.atomic_list_struct import atoms
from pydephasing.gradient_interactions import gradient_HFI, gradient_2nd_HFI
from pydephasing.nuclear_spin_config import nuclear_spins_config
from pydephasing.extract_ph_data import extract_ph_data
from pydephasing.spin_ph_inter import SpinPhononClass, SpinPhononRelaxClass
from pydephasing.auto_correlation_module import acf_ph_deph
from pydephasing.T2_classes import T2i_ofT, Delta_ofT, tauc_ofT
from pydephasing.log import log
from pydephasing.mpi import mpi
from pydephasing.restart import restart_calculation, save_data
#
def compute_hfi_dephas():
    # driving code for the calculation of
    # inhomogeneous dephasing
    #
    # create perturbed atomic structures
    struct_list = []
    for i in range(len(p.displ_poscar_dir)):
        displ_struct = DisplacedStructs(p.displ_poscar_dir[i], p.displ_outcar_dir[i])
        # set atomic displacements
        displ_struct.atom_displ(p.atoms_displ[i])    # ang
        # append to list
        struct_list.append(displ_struct)
    # 2nd order displ structs
    if p.order_2_correct:
        struct_list_2nd = []
        for i in range(len(p.displ_2nd_poscar_dir)):
            displ_struct = DisplacedStructures2ndOrder(p.displ_2nd_poscar_dir[i], p.displ_2nd_outcar_dir[i])
            # set atomic displ. in the structure
            displ_struct.atom_displ(p.atoms_2nd_displ[i]) # ang
            # append to list
            struct_list_2nd.append(displ_struct)
    # set HFI gradient
    if mpi.rank == mpi.root:
        log.info("HF fc core: " + str(p.fc_core))
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
    # HFI 2nd order
    if p.order_2_correct:
        # set 2nd order tensor
        grad2HFI = gradient_2nd_HFI(p.work_dir, p.grad_info, p.fc_core)
        grad2HFI.set_gs_hfi_tensor()
        # compute 2nd order grad. HFI
        grad2HFI.extract_2nd_order_gradients()
    #
    # set up the spin
    # configurations
    #
    if mpi.rank == mpi.root:
        log.info("n. config: " + str(p.nconf))
        log.info("n. spins: " + str(p.nsp))
    #
    # set spin ph. interaction object
    #
    if p.deph:
        sp_ph_inter = SpinPhononDephClass()
        # set states
        p.qs1 = np.array([1.0+0j,0j,0j])
        p.qs2 = np.array([0j,1.0+0j,0j])
    elif p.relax:
        sp_ph_inter = SpinPhononRelaxClass()
    else:
        log.error("type of calculation must be deph or relax")
    # normalization
    p.qs1[:] = p.qs1[:] / np.sqrt(np.dot(p.qs1.conjugate(), p.qs1))
    p.qs2[:] = p.qs2[:] / np.sqrt(np.dot(p.qs2.conjugate(), p.qs2))
    # set spin Hamiltonian
    Hsp = spin_hamiltonian()
    #
    # extract phonon data
    #
    u, wu, nq, qpts, wq, mesh = extract_ph_data()
    if mpi.rank == 0:
        log.info("nq: " + str(nq))
        log.info("mesh: " + str(mesh))
        log.info("wq: " + str(wq))
    assert len(qpts) == nq
    assert len(u) == nq
    mpi.comm.Barrier()
    # set auto correlation function
    ql_list = mpi.split_ph_modes(nq, 3*nat)
    # set wql grid
    if p.ph_resolved:
        p.set_wql_grid(wu, nq, nat)
    # auto correlation function
    acf = acf_ph_deph().generate_instance()
    # set average T2 / D^2 / tau_c
    T2_obj_aver = T2i_ofT(nat)
    Delt_obj_aver = Delta_ofT(nat)
    tauc_obj_aver = tauc_ofT(nat)
    # deph. data lists
    T2_list = []
    Delt_list = []
    tauc_list = []
    # restart calculation
    restart_file = p.write_dir + "/restart_calculation.yml"
    isExist = os.path.exists(restart_file)
    if not isExist:
        ic0 = 0
        acf_aver = np.zeros((p.nt,p.ntmp), dtype=np.complex128)
        if p.at_resolved:
            acf_atr_aver = np.zeros((p.nt2,nat,p.ntmp), dtype=np.complex128)
        if p.ph_resolved:
            acf_wql_aver = np.zeros((p.nt2,p.nwbn,p.ntmp), dtype=np.complex128)
            if p.nphr > 0:
                acf_phr_aver = np.zeros((p.nt2,p.nphr,p.ntmp), dtype=np.complex128)
    else:
        ic0, T2_list, Delt_list, tauc_list, acf_data = restart_calculation(restart_file)
        acf_aver = acf_data[0]
        if p.at_resolved:
            acf_atr_aver = acf_data[1]
            if p.ph_resolved and len(acf_data) == 3:
                acf_wql_aver = acf_data[2]
            elif p.ph_resolved and len(acf_data) == 4:
                acf_wql_aver = acf_data[2]
                acf_phr_aver = acf_data[3]
        else:
            if p.ph_resolved and len(acf_data) == 2:
                acf_wql_aver = acf_data[1]
            elif p.ph_resolved and len(acf_data) == 3:
                acf_wql_aver = acf_data[1]
                acf_phr_aver = acf_data[2]
    #
    # run over different configurations
    for ic in range(ic0, p.nconf):
        # set spin config.
        config = nuclear_spins_config(p.nsp, p.B0)
        config.set_nuclear_spins(nat, ic)
        # compute the forces
        sp_ph_inter.set_Fax_hfi(gradHFI, Hsp, config)
        Fax = sp_ph_inter.Fhf_ax
        # eV / ang
        # 2nd order
        if p.order_2_correct:
            sp_ph_inter.set_Faxby_hfi(grad2HFI, Hsp, struct_list_2nd, config)
            # eV / ang^2
            Faxby = sp_ph_inter.Fhf_axby
        else:
            Faxby = None
        # compute over (q,l) list
        acf.compute_acf(wq, wu, u, qpts, nat, Fax, Faxby, ql_list)
        # parallelization sec.
        if mpi.size > 1:
            acf.collect_acf_from_processes(nat)
        mpi.comm.Barrier()
        # <acf>
        acf_aver[:,:] += acf.acf[:,:]
        if p.at_resolved:
            acf_atr_aver[:,:,:] += acf.acf_atr[:,:,:]
        if p.ph_resolved:
            acf_wql_aver[:,:,:] += acf.acf_wql[:,:,:]
            if p.nphr > 0:
                acf_phr_aver[:,:,:] += acf.acf_phr[:,:,:]
        # prepare data arrays
        T2_obj = T2i_ofT(nat)
        Delt_obj = Delta_ofT(nat)
        tauc_obj = tauc_ofT(nat)
        #
        # run over temperature
        for iT in range(p.ntmp):
            # extract dephasing data
            ft = acf.extract_dephas_data(T2_obj, Delt_obj, tauc_obj, iT)
            ft_inp = np.zeros((p.nt,2))
            ft_inp[:,0] = ft[0][:]
            ft_inp[:,1] = ft[1][:]
            #
            # if atom resolved
            #
            ft_atr = None
            if p.at_resolved:
                # make local atoms list
                atr_list = mpi.split_list(range(nat))
                # run over atoms
                ft_atr = np.zeros((p.nt2,nat,2))
                for ia in atr_list:
                    # compute acf + T2 times
                    ft_ia = acf.extract_dephas_data_atr(T2_obj, Delt_obj, tauc_obj, ia, iT)
                    if ft_ia is not None:
                        ft_atr[:,ia,0] = ft_ia[0][:]
                        ft_atr[:,ia,1] = ft_ia[1][:]
                ft_atr = mpi.collect_array(ft_atr)
                # collect into single proc.
                T2_obj.collect_atr_from_other_proc(iT)
                Delt_obj.collect_atr_from_other_proc(iT)
                tauc_obj.collect_atr_from_other_proc(iT)
            #
            # if ph. resolved calc.
            #
            ft_phr = None
            ft_wql = None
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
                # local wql grid list
                local_wql_list = mpi.split_list(np.arange(0, p.nwbn, 1))
                # run over modes
                ft_wql = np.zeros((p.nt2,p.nwbn,2))
                for iwb in local_wql_list:
                    # compute acf + T2 times + print acf data
                    ft_ii = acf.extract_dephas_data_wql(T2_obj, Delt_obj, tauc_obj, iwb, iT)
                    if ft_ii is not None:
                        ft_wql[:,iwb,0] = ft_ii[0][:]
                        ft_wql[:,iwb,1] = ft_ii[1][:]
                ft_wql = mpi.collect_array(ft_wql)
                # collect data into single proc.
                T2_obj.collect_phr_from_other_proc(iT)
                Delt_obj.collect_phr_from_other_proc(iT)
                tauc_obj.collect_phr_from_other_proc(iT)
            #
            # print acf
            if mpi.rank == mpi.root:
                acf.print_autocorrel_data_spinconf(ft_inp, ft_atr, ft_wql, ft_phr, ic+1, iT)
        # wait
        mpi.comm.Barrier()
        T2_list.append(T2_obj)
        Delt_list.append(Delt_obj)
        tauc_list.append(tauc_obj)
        if mpi.rank == mpi.root:
            log.warning("ic: " + str(ic+1) + " -> completed")
        # save temp. data
        if mpi.rank == mpi.root:
            acf_data = [acf_aver]
            if p.at_resolved:
                acf_data.append(acf_atr_aver)
            if p.ph_resolved:
                acf_data.append(acf_wql_aver)
                acf_data.append(acf_phr_aver)
            save_data(ic, T2_list, Delt_list, tauc_list, acf_data)
        mpi.comm.Barrier()
    #
    # compute average data
    #
    acf_aver[:,:] = acf_aver[:,:] / p.nconf
    acf.acf[:,:] = acf_aver[:,:]
    if p.at_resolved:
        acf_atr_aver[:,:,:] = acf_atr_aver[:,:,:] / p.nconf
        acf.acf_atr[:,:,:] = acf_atr_aver[:,:,:]
    if p.ph_resolved:
        acf_phr_aver[:,:,:] = acf_phr_aver[:,:,:] / p.nconf
        acf.acf_phr[:,:,:] = acf_phr_aver[:,:,:]
        acf_wql_aver[:,:,:] = acf_wql_aver[:,:,:] / p.nconf
        acf.acf_wql[:,:,:] = acf_wql_aver[:,:,:]
    # run over T
    for iT in range(p.ntmp):
        # extract dephasing data
        ft = acf.extract_dephas_data(T2_obj_aver, Delt_obj_aver, tauc_obj_aver, iT)
        ft_inp = np.zeros((p.nt,2))
        ft_inp[:,0] = ft[0][:]
        ft_inp[:,1] = ft[1][:]
        #
        # if atom resolved
        #
        ft_atr = None
        if p.at_resolved:
            # run over atoms
            ft_atr = np.zeros((p.nt2,nat,2))
            for ia in atr_list:
                # compute acf + T2 times
                ft_ia = acf.extract_dephas_data_atr(T2_obj_aver, Delt_obj_aver, tauc_obj_aver, ia, iT)
                if ft_ia is not None:
                    ft_atr[:,ia,0] = ft_ia[0][:]
                    ft_atr[:,ia,1] = ft_ia[1][:]
            ft_atr = mpi.collect_array(ft_atr)
            # collect into single proc.
            T2_obj_aver.collect_atr_from_other_proc(iT)
            Delt_obj_aver.collect_atr_from_other_proc(iT)
            tauc_obj_aver.collect_atr_from_other_proc(iT)
        #
        # if ph. resolved calc.
        #
        ft_phr = None
        ft_wql = None
        if p.ph_resolved:
            # run over modes
            ft_phr = np.zeros((p.nt2,p.nphr,2))
            for im in local_ph_list:
                iph = p.phm_list.index(im)
                # compute acf + T2 times + print acf data
                ft_iph = acf.extract_dephas_data_phr(T2_obj_aver, Delt_obj_aver, tauc_obj_aver, iph, iT)
                if ft_iph is not None:
                    ft_phr[:,iph,0] = ft_iph[0][:]
                    ft_phr[:,iph,1] = ft_iph[1][:]
            ft_phr = mpi.collect_array(ft_phr)
            # local wql grid list
            local_wql_list = mpi.split_list(np.arange(0, p.nwbn, 1))
            # run over modes
            ft_wql = np.zeros((p.nt2,p.nwbn,2))
            for iwb in local_wql_list:
                # compute acf + T2 times + print acf data
                ft_ii = acf.extract_dephas_data_wql(T2_obj, Delt_obj, tauc_obj, iwb, iT)
                if ft_ii is not None:
                    ft_wql[:,iwb,0] = ft_ii[0][:]
                    ft_wql[:,iwb,1] = ft_ii[1][:]
            ft_wql = mpi.collect_array(ft_wql)
            # collect data into single proc.
            T2_obj_aver.collect_phr_from_other_proc(iT)
            Delt_obj_aver.collect_phr_from_other_proc(iT)
            tauc_obj_aver.collect_phr_from_other_proc(iT)
        #
        # print acf
        if mpi.rank == mpi.root:
            acf.print_autocorrel_data_spinconf(ft_inp, ft_atr, ft_wql, ft_phr, 0, iT)
        mpi.comm.Barrier()
    # append average data
    T2_list.append(T2_obj_aver)
    Delt_list.append(Delt_obj_aver)
    tauc_list.append(tauc_obj_aver)
    # wait
    mpi.comm.Barrier()
    return T2_list, Delt_list, tauc_list