import sys
from pydephasing.create_displ_struct_files import gen_poscars, gen_2ndorder_poscar
from pydephasing.input_parameters import p
from pydephasing.compute_zfs_hfi_dephas import compute_full_dephas
from pydephasing.compute_zfs_dephas import compute_homo_dephas
from pydephasing.compute_dyndec_dephas import compute_homo_dyndec_dephas
from pydephasing.compute_exc_dephas import compute_homo_exc_dephas
from pydephasing.compute_hfi_dephas import compute_hfi_dephas
from pydephasing.compute_hfi_dephas_stat import compute_hfi_stat_dephas
from pydephasing.T2_classes import print_dephas_data, print_dephas_data_phr, print_dephas_data_atr, print_dephas_data_hfi, print_dephas_data_atr_hfi, print_dephas_data_phr_hfi, print_dephas_data_stat, print_dephas_data_dyndec
from pydephasing.mpi import mpi
from pydephasing.log import log
from pydephasing.timer import timer
#
# set up parallelization
#
if mpi.rank == mpi.root:
    log.info("start pydephasing                              ")
#
if len(sys.argv) < 5:
    if sys.argv[1] == "--init":
        pass
    else:
        if mpi.rank == mpi.root:
            log.warning("code usage: \n")
            log.warning("python pydephasing --[energy/spin] --[homo/inhomo] --[deph/relax] input.yml")
            log.error("wrong execution parameters: pydephasing stops")
timer.start_execution()
calc_type = sys.argv[1]
if calc_type == "--energy":
    if mpi.rank == mpi.root:
        log.info("energy level dephasing calculation         ")
    # prepare energy dephasing calculation
    calc_type2 = sys.argv[2]
    calc_type3 = sys.argv[3]
    if calc_type2 == "--homo":
        if calc_type3 == "--deph":
            p.deph = True
            p.relax= False
            if mpi.rank == mpi.root:
                log.info("homogeneous - dephasing calculation                ")
        elif calc_type3 == "--relax":
            p.relax = True
            p.deph  = False
            if mpi.rank == mpi.root:
                log.info("homogeneous - relaxation calculation               ")
        else:
            log.warning("code usage: \n")
            log.warning("python pydephasing --[energy/spin] --[homo/inhomo] --[deph/relax] input.yml")
            log.error("--deph or --relax notspecified")
        # read input file
        input_file = sys.argv[4]
        p.read_yml_data(input_file)
        # compute auto correl. function first
        T2_obj, Delt_obj, tauc_obj, lw_obj = compute_homo_exc_dephas()
        # finalize calculation
        if mpi.rank == mpi.root:
            log.info("    print results on file    ")
            # write T2 yaml files
            print_dephas_data(T2_obj, tauc_obj, Delt_obj, lw_obj)
            # if atom resolved
            if p.at_resolved:
                print_dephas_data_atr(T2_obj, tauc_obj, Delt_obj, lw_obj)
            # if phonon resolved
            if p.ph_resolved:
                print_dephas_data_phr(T2_obj, tauc_obj, Delt_obj, lw_obj)
elif calc_type == "--spin":
    if mpi.rank == mpi.root:
        log.info("spin-phonon calculation                    ")
    # prepare spin dephasing calculation
    calc_type2 = sys.argv[2]
    calc_type3 = sys.argv[3]
    # --------------------------------------------------------------
    # 
    #    SIMPLE HOMOGENEOUS CALC. (ZFS ONLY)
    #
    # --------------------------------------------------------------
    if calc_type2 == "--homo":
        if calc_type3 == "--deph":
            p.deph = True
            p.relax= False
            if mpi.rank == mpi.root:
                log.info("homogeneous spin dephasing calculation                 ")
                log.info("setting up T2 calculation                              ")
        elif calc_type3 == "--relax":
            p.deph = False
            p.relax= True
            if mpi.rank == mpi.root:
                log.info("homogeneous spin relaxation calculation                ")
                log.info("setting up T1 calculation                              ")
        elif calc_type3 == "--dyndec":
            p.deph = True
            p.relax = False
            p.dyndec = True
            if mpi.rank == mpi.root:
                log.info("homogeneous spin dephasing calculation + dynamical decoupling   ")
                log.info("setting up T2 calculation                                       ")
        # read input file
        input_file = sys.argv[4]
        p.read_yml_data(input_file)
        # compute auto correl. function first
        if p.dyndec:
            T2_obj, Delt_obj, tauc_obj = compute_homo_dyndec_dephas()
        else:
            T2_obj, Delt_obj, tauc_obj = compute_homo_dephas()
        # finalize calculation
        if mpi.rank == mpi.root:
            log.info("    print results on file    ")
            # write T2 yaml files
            if p.dyndec:
                print_dephas_data_dyndec(T2_obj, tauc_obj, Delt_obj)
            else:
                print_dephas_data(T2_obj, tauc_obj, Delt_obj)
            # if atom resolved
            if p.at_resolved:
                print_dephas_data_atr(T2_obj, tauc_obj, Delt_obj)
            # if phonon resolved
            if p.ph_resolved:
                print_dephas_data_phr(T2_obj, tauc_obj, Delt_obj)
        mpi.comm.Barrier()
    # --------------------------------------------------------------
    # 
    #    FULL CALC. (HFI + ZFS)
    #
    # --------------------------------------------------------------
    elif calc_type2 == "--full":
        if calc_type3 == "--deph":
            p.deph = True
            p.relax= False
            if mpi.rank == mpi.root:
                log.info("full spin dephasing calculation                        ")
                log.info("setting up T2 calculation                              ")
        elif calc_type3 == "--relax":
            p.deph = False
            p.relax= True
            if mpi.rank == mpi.root:
                log.info("full spin relaxation calculation                       ")
                log.info("setting up T1 calculation                              ")
        # read input file
        input_file = sys.argv[4]
        p.read_yml_data(input_file)
        # compute auto correl. function first
        T2_obj, Delt_obj, tauc_obj = compute_full_dephas()
        # finalize calculation
        if mpi.rank == mpi.root:
            log.info("    print results on file    ")
            # write T2 yaml files
            print_dephas_data(T2_obj, tauc_obj, Delt_obj)
            # if atom resolved
            if p.at_resolved:
                print_dephas_data_atr(T2_obj, tauc_obj, Delt_obj)
            # if phonon resolved
            if p.ph_resolved:
                print_dephas_data_phr(T2_obj, tauc_obj, Delt_obj)
        mpi.comm.Barrier()
    # --------------------------------------------------------------
    # 
    #    SIMPLE INHOMOGENEOUS CALC. (HFI ONLY)
    #
    # --------------------------------------------------------------
    elif calc_type2 == "--inhomo":
        # read input file
        input_file = sys.argv[4]
        # check calc type
        if calc_type3 != "--stat":
            # read file
            p.read_yml_data(input_file)
            if calc_type3 == "--deph":
                p.deph = True
                p.relax= False
                if mpi.rank == 0:
                    log.info("inhomogeneous spin dephasing calculation              ")
                    log.info("setting up T2* calculation                            ")
            elif calc_type3 == "--relax":
                p.deph = False
                p.relax= True
                if mpi.rank == 0:
                    log.info("inhomogeneous spin relaxation calculation              ")
                    log.info("setting up T1 calculation                              ")
            # compute the dephas. time
            T2_obj_lst, Delt_obj_lst, tauc_obj_lst = compute_hfi_dephas()
            # finalize calculation
            if mpi.rank == mpi.root:
                log.info("    print results on file    ")
                # write T2 yaml files
                print_dephas_data_hfi(T2_obj_lst, tauc_obj_lst, Delt_obj_lst)
                # if atom resolved
                if p.at_resolved:
                    print_dephas_data_atr_hfi(T2_obj_lst, tauc_obj_lst, Delt_obj_lst)
                # if phonon resolved
                if p.ph_resolved:
                    print_dephas_data_phr_hfi(T2_obj_lst, tauc_obj_lst, Delt_obj_lst)
            mpi.comm.Barrier()
        elif calc_type3 == "--stat":
            # read file
            p.read_inhomo_stat(input_file)
            # static HFI calculation
            if mpi.rank == 0:
                log.info("inhomogeneous static spin dephasing              ")
            # compute dephasing time
            T2s_obj, Delt_obj, tauc_obj = compute_hfi_stat_dephas()
            # finalize calculation
            if mpi.rank == mpi.root:
                log.info("    print results on file    ")
                # write T2 yaml files
                print_dephas_data_stat(T2s_obj, tauc_obj, Delt_obj)
            mpi.comm.Barrier()
        else:
            if mpi.rank == 0:
                log.warning("wrong action flag usage:                  ")
                log.warning("-dyn -> dynamic inhomogeneous calculation ")
                log.warning("-stat -> static inhomogeneous calculation ")
            log.error("Wrong action type flag")
elif calc_type == "--init":
    # read data file
    order = sys.argv[2]
    input_file = sys.argv[3]
    # read data
    p.read_yml_data_pre(input_file)
    if mpi.rank == mpi.root:
        log.info("BUILD DISPLACED STRUCTURES             ")
    if int(order) == 1:
        if mpi.rank == mpi.root:
            gen_poscars(p.max_dist_defect, p.defect_index)
    elif int(order) == 2:
        if mpi.rank == mpi.root:
            gen_2ndorder_poscar(p.max_dist_defect, p.defect_index, p.max_dab)
    else:
        if mpi.rank == mpi.root:
            log.warning("wrong order flag                ")
            log.warning("order=1 or 2                    ")
        log.error("wrong displacement order flag     ")
elif calc_type == "--post":
    # post process output data from VASP
    pass
else:
    if mpi.rank == mpi.root:
        log.warning("CALC_TYPE FLAG NOT RECOGNIZED         ")
        log.warning("EXIT PROGRAM                          ")
    log.error("wrong calculation flag                      ")
# end execution
timer.end_execution()
if mpi.rank == mpi.root:
    log.info("PROCEDURE SUCCESSFULLY COMPLETED")
mpi.finalize_procedure()