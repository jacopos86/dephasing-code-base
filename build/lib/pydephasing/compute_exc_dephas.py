# This is the main subroutine
# for the homogeneous exciton dephasing calculation
# it computes the energy auto-correlation function
# and it returns it for further processing
import numpy as np
import logging
from pydephasing.mpi import mpi
from pydephasing.log import log
from pydephasing.gradient_interactions import gradient_Eg
from pydephasing.utility_functions import print_zpl_fluct
from pydephasing.extract_ph_data import extract_ph_data
from pydephasing.auto_correlation_module import acf_ph_deph
from pydephasing.T2_classes import T2i_ofT, Delta_ofT, tauc_ofT, lw_ofT
from pydephasing.input_parameters import p
from pydephasing.atomic_list_struct import atoms
#
def compute_homo_exc_dephas():
    # input_params -> input parameters data object
    # at_resolved -> atom resolved calculation
    # ph_resolved -> phonon resolved calculation
    #
    #
    # compute GS config. forces
    #
    gradEGS = gradient_Eg(p.grad_info, 'GS_DIR')
    # set struct 0
    gradEGS.set_unpert_struct()
    # set atoms info
    nat = gradEGS.struct_0.nat
    atoms.compute_index_to_ia_map(nat)
    atoms.compute_index_to_idx_map(nat)
    # set GS forces
    gradEGS.set_forces()
    # set force constants
    gradEGS.set_force_constants()
    #
    # set EXC config. forces
    #
    # second: extract unperturbed structure (excited state)
    gradEXC = gradient_Eg(p.grad_info, 'EXC_DIR')
    # set struct 0
    gradEXC.set_unpert_struct()
    # set GS forces
    gradEXC.set_forces()
    # set force constants
    gradEXC.set_force_constants()
    # ZPL
    if mpi.rank == mpi.root:
        log.info("exciton energy= " + str(gradEXC.struct_0.E-gradEGS.struct_0.E) + " eV")
    # gradZPL -> gradEXC - gradEGS
    gradZPL = np.zeros(3*nat)
    gradZPL[:] = gradEXC.forces[:] - gradEGS.forces[:]
    # eV / Ang units
    #
    # define force constants -> HessZPL
    #
    hessZPL = np.zeros((3*nat,3*nat))
    hessZPL[:,:] = gradEXC.force_const[:,:] - gradEGS.force_const[:,:]
    # eV / Ang^2 units
    # print data
    if mpi.rank == mpi.root:
        if log.level <= logging.INFO:
            print_zpl_fluct(gradZPL, hessZPL, p.write_dir)
    mpi.comm.Barrier()
    # extract atom positions
    atoms.extract_atoms_coords(nat)
    # extract phonon data
    u, wu, nq, qpts, wq, mesh = extract_ph_data()
    if mpi.rank == mpi.root:
        log.info("nq: " + str(nq))
        log.info("mesh: " + str(mesh))
        log.info("wq: " + str(wq))
    assert len(qpts) == nq
    assert len(u) == nq
    mpi.comm.Barrier()
    # set auto correlation function
    ql_list = mpi.split_ph_modes(nq, 3*nat)
    #
    # compute acf over local list
    acf = acf_ph_deph().generate_instance()
    acf.compute_acf(wq, wu, u, qpts, nat, gradZPL, hessZPL, ql_list)
    # collect data from processes
    if mpi.size > 1:
        acf.collect_acf_from_processes(nat)
    mpi.comm.Barrier()
    # prepare data arrays
    T2_obj = T2i_ofT(nat)
    Delt_obj = Delta_ofT(nat)
    tauc_obj = tauc_ofT(nat)
    lw_obj = lw_ofT(nat)
    # run over the temperature list
    for iT in range(p.ntmp):
        # extract dephasing data
        ft = acf.extract_dephas_data(T2_obj, Delt_obj, tauc_obj, iT, lw_obj)
        ft_inp = np.zeros((p.nt,2))
        ft_inp[:,0] = ft[0][:]
        ft_inp[:,1] = ft[1][:]
        #
        # if atom resolved
        #
        ft_atr = None
        if p.at_resolved:
            # local list of atoms
            atr_list = mpi.split_list(range(nat))
            # run over each atom
            ft_atr = np.zeros((p.nt2,nat,2))
            for ia in atr_list:
                # compute T2 atoms + print
                ft_ia = acf.extract_dephas_data_atr(T2_obj, Delt_obj, tauc_obj, ia, iT, lw_obj)
                if ft_ia is not None:
                    ft_atr[:,ia,0] = ft_ia[0][:]
                    ft_atr[:,ia,1] = ft_ia[1][:]
            ft_atr = mpi.collect_array(ft_atr)
            # collect in single proc.
            T2_obj.collect_atr_from_other_proc(iT)
            Delt_obj.collect_atr_from_other_proc(iT)
            tauc_obj.collect_atr_from_other_proc(iT)
            lw_obj.collect_atr_from_other_proc(iT)
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
                ft_iph = acf.extract_dephas_data_phr(T2_obj, Delt_obj, tauc_obj, iph, iT, lw_obj)
                if ft_iph is not None:
                    ft_phr[:,iph,0] = ft_iph[0][:]
                    ft_phr[:,iph,1] = ft_iph[1][:]
            ft_phr = mpi.collect_array(ft_phr)
            # collect data into singl. proc.
            T2_obj.collect_phr_from_other_proc(iT)
            Delt_obj.collect_phr_from_other_proc(iT)
            tauc_obj.collect_phr_from_other_proc(iT)
            lw_obj.collect_phr_from_other_proc(iT)
        #
        # print acf
        if mpi.rank == mpi.root:
            acf.print_autocorrel_data(ft_inp, ft_atr, ft_phr, iT)
        # wait processes
        mpi.comm.Barrier()
    return T2_obj, Delt_obj, tauc_obj, lw_obj