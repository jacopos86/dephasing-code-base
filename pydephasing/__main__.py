from pydephasing.create_displ_struct_files import gen_poscars, gen_2ndorder_poscar
from pydephasing.input_parameters import p
from pydephasing.compute_zfs_hfi_dephas import compute_full_dephas
from pydephasing.compute_zfs_dephas import compute_homo_dephas
from pydephasing.compute_exc_dephas import compute_homo_exc_dephas
from pydephasing.compute_hfi_dephas import compute_hfi_dephas
from pydephasing.compute_hfi_dephas_stat import compute_hfi_stat_dephas
from pydephasing.T2_classes import print_dephas_data, print_dephas_data_phr, print_dephas_data_atr, print_dephas_data_hfi, print_dephas_data_atr_hfi, print_dephas_data_phr_hfi, print_dephas_data_stat, print_dephas_data_dyndec
from pydephasing.mpi import mpi
from pydephasing.log import log
from pydephasing.timer import timer
from pydephasing.input_parser import parser
#
# set up parallelization
#
if mpi.rank == mpi.root:
    log.info("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    log.info("++++++                                                                                  ++++++")
    log.info("++++++                           PYDEPHASING   CODE                                     ++++++")
    log.info("++++++                                                                                  ++++++")
    log.info("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
#
yml_file = parser.parse_args().yml_inp[0]
if yml_file is None:
    if mpi.rank == mpi.root:
        log.error("-> yml file name missing")
nargs = 2
if parser.parse_args().ct2 is not None:
    nargs += 1
if parser.parse_args().typ is not None:
    nargs += 1
if nargs < 4:
    if parser.parse_args().ct1[0] == "init":
        pass
    else:
        if mpi.rank == mpi.root:
            log.warning("->           code usage: \n")
            log.warning("->           python pydephasing [energy/spin] [homo/inhomo] [deph/relax/stat/statdd] input.yml")
            log.error("-----          wrong execution parameters: pydephasing stops                -------")
timer.start_execution()
calc_type1 = parser.parse_args().ct1[0]
if calc_type1 == "energy":
    if mpi.rank == mpi.root:
        log.info("-----------                   energy level dephasing calculation         ---------------")
    # prepare energy dephasing calculation
    calc_type2 = parser.parse_args().ct2
    deph_type  = parser.parse_args().typ
    if calc_type2 == "homo":
        if deph_type == "deph":
            p.deph = True
            p.relax= False
            if mpi.rank == mpi.root:
                log.info("-----------                   homogeneous - dephasing calculation         ---------------")
        elif deph_type == "relax":
            p.relax = True
            p.deph  = False
            if mpi.rank == mpi.root:
                log.info("-----------                   homogeneous - relaxation calculation         --------------")
        else:
            log.warning("->           code usage: \n")
            log.warning("->           python pydephasing [energy/spin] [homo/inhomo] [deph/relax/stat/statdd] input.yml")
            log.error("-----          deph or --relax notspecified                                  -------")
        # read input file
        p.read_yml_data(yml_file)
        # compute auto correl. function first
        T2_obj, Delt_obj, tauc_obj, lw_obj = compute_homo_exc_dephas()
        # finalize calculation
        if mpi.rank == mpi.root:
            log.info("-----------                   PRINT DATA ON FILES         --------------")
            # write T2 yaml files
            print_dephas_data(T2_obj, tauc_obj, Delt_obj, lw_obj)
            # if atom resolved
            if p.at_resolved:
                print_dephas_data_atr(T2_obj, tauc_obj, Delt_obj, lw_obj)
            # if phonon resolved
            if p.ph_resolved:
                print_dephas_data_phr(T2_obj, tauc_obj, Delt_obj, lw_obj)
elif calc_type1 == "spin":
    if mpi.rank == mpi.root:
        log.info("------------------                   SPIN - PHONON CALCULATION        --------------------")
    # prepare spin dephasing calculation
    calc_type2 = parser.parse_args().ct2
    deph_type = parser.parse_args().typ
    # --------------------------------------------------------------
    # 
    #    SIMPLE HOMOGENEOUS CALC. (ZFS ONLY)
    #
    # --------------------------------------------------------------
    if calc_type2 == "homo":
        if deph_type == "deph":
            p.deph = True
            p.relax= False
            if mpi.rank == mpi.root:
                log.info("----------------                  T2 CALCULATION -> STARTING        --------------")
                log.info("------------                  HOMOGENEOUS SPIN - DEPHASING           -------------")
                log.info("----------------------------------------------------------------------------------")
        elif deph_type == "relax":
            p.deph = False
            p.relax= True
            if mpi.rank == mpi.root:
                log.info("-----------                  T1 CALCULATION -> STARTING        -------------")
                log.info("--------                  HOMOGENEOUS SPIN - RELAXATION           ----------")
        # read input file
        p.read_yml_data(yml_file)
        # compute auto correl. function first
        T2_obj, Delt_obj, tauc_obj, lw_obj = compute_homo_dephas()
        # finalize calculation
        if mpi.rank == mpi.root:
            log.info("-----------                   PRINT DATA ON FILES         --------------")
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
    #    FULL CALC. (HFI + ZFS)
    #
    # --------------------------------------------------------------
    elif calc_type2 == "full":
        if deph_type == "deph":
            p.deph = True
            p.relax= False
            if mpi.rank == mpi.root:
                log.info("-----------                  T2 CALCULATION -> STARTING        -------------")
                log.info("-------                  FULL HOMOGENEOUS SPIN - DEPHASING           -------") 
        elif deph_type == "relax":
            p.deph = False
            p.relax= True
            if mpi.rank == mpi.root:
                log.info("--------                 FULL T1 CALCULATION -> STARTING        ------------")
                log.info("--------                  HOMOGENEOUS SPIN - RELAXATION           ----------")
        # read input file
        p.read_yml_data(yml_file)
        # compute auto correl. function first
        T2_obj, Delt_obj, tauc_obj, lw_obj = compute_full_dephas()
        # finalize calculation
        if mpi.rank == mpi.root:
            log.info("-----------                   PRINT DATA ON FILES         --------------")
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
    elif calc_type2 == "inhomo":
        # check calc type
        if deph_type != "stat" or deph_type != "statdd":
            # read file
            p.read_yml_data(yml_file)
            if deph_type == "deph":
                p.deph = True
                p.relax= False
                if mpi.rank == 0:
                    log.info("-----------                T2* CALCULATION -> STARTING        -------------")
                    log.info("---------                  INHOMOGENEOUS SPIN - DEPHASING        ----------")
            elif deph_type == "relax":
                p.deph = False
                p.relax= True
                if mpi.rank == 0:
                    log.info("-----------                T1* CALCULATION -> STARTING        -------------")
                    log.info("---------                  INHOMOGENEOUS SPIN - RELAXATION       ----------")
            # compute the dephas. time
            T2_obj, Delt_obj, tauc_obj, lw_obj = compute_hfi_dephas()
            # finalize calculation
            if mpi.rank == mpi.root:
                log.info("-----------                   PRINT DATA ON FILES         --------------")
                # write T2 yaml files
                print_dephas_data_hfi(T2_obj, tauc_obj, Delt_obj, lw_obj)
                # if atom resolved
                if p.at_resolved:
                    print_dephas_data_atr_hfi(T2_obj, tauc_obj, Delt_obj, lw_obj)
                # if phonon resolved
                if p.ph_resolved:
                    print_dephas_data_phr_hfi(T2_obj, tauc_obj, Delt_obj, lw_obj)
            mpi.comm.Barrier()
        elif deph_type == "stat" or deph_type == "statdd":
            if deph_type == "statdd":
                p.dyndec = True
            # read file
            p.read_inhomo_stat(yml_file)
            # static HFI calculation
            if mpi.rank == 0:
                log.info("-----------                T2* CALCULATION -> STARTING        ------------------")
                log.info("-----------              INHOMOGENEOUS STATIC SPIN - DEPHASING        ----------")
            # compute dephasing time
            T2_obj, Delt_obj, tauc_obj, lw_obj = compute_hfi_stat_dephas()
            # finalize calculation
            if mpi.rank == mpi.root:
                log.info("-----------                   PRINT DATA ON FILES         --------------")
                # write T2 yaml files
                print_dephas_data_stat(T2_obj, tauc_obj, Delt_obj, lw_obj)
            mpi.comm.Barrier()
        else:
            if mpi.rank == 0:
                log.warning("->           code usage: \n")
                log.warning("->           python pydephasing [energy/spin] [homo/inhomo] [deph/relax/stat/statdd] input.yml")
                log.error("-----          Wrong action type flag: pydephasing stops                -------")
elif calc_type1 == "init":
    # read data file
    order = parser.parse_args().o
    # read data
    p.read_yml_data_pre(yml_file)
    if mpi.rank == mpi.root:
        log.info("-------           BUILD DISPLACED STRUCTS.           ---------")
    if int(order) == 1:
        if mpi.rank == mpi.root:
            gen_poscars(p.max_dist_defect, p.defect_index)
    elif int(order) == 2:
        if mpi.rank == mpi.root:
            gen_2ndorder_poscar(p.max_dist_defect, p.defect_index, p.max_dab)
    else:
        if mpi.rank == mpi.root:
            log.warning("-------           Wrong order flag            ---------")
            log.warning("-------           order=1 or 2                ---------")
            log.error("--------       Wrong displacement order flag    ---------")
elif calc_type1 == "--post":
    # post process output data from VASP
    pass
else:
    if mpi.rank == mpi.root:
        log.warning("-------           CALC. TYPE NOT RECOGNIZED       ---------")
        log.warning("-------           QUIT PROGRAM                    ---------")
        log.error("-------            WRONG CALC. FLAG                 ---------")
# end execution
timer.end_execution()
if mpi.rank == mpi.root:
    log.info("-------           PROCEDURE SUCCESSFULLY COMPLETED       ---------")
mpi.finalize_procedure()