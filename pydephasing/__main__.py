from pydephasing.set_param_object import p
from pydephasing.create_displ_struct_files import gen_poscars, gen_2ndorder_poscar
from pydephasing.compute_zfs_hfi_dephas import compute_full_dephas
from pydephasing.compute_zfs_dephas import compute_homo_dephas
from pydephasing.compute_exc_dephas import compute_homo_exc_dephas
from pydephasing.compute_hfi_dephas import compute_hfi_dephas
from pydephasing.compute_hfi_dephas_stat import compute_hfi_stat_dephas
from pydephasing.mpi import mpi
from pydephasing.log import log
from pydephasing.timer import timer
from pydephasing.input_parser import parser
#
# set up parallelization
#
if mpi.rank == mpi.root:
    log.info("\t ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    log.info("\t ++++++                                                                                  ++++++")
    log.info("\t ++++++                           PYDEPHASING   CODE                                     ++++++")
    log.info("\t ++++++                                                                                  ++++++")
    log.info("\t ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
#
yml_file = parser.parse_args().yml_inp[0]
if yml_file is None:
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
            log.info("\n")
            log.info("\t " + p.sep)
            log.warning("\t CODE USAGE: \n")
            log.warning("\t -> python pydephasing [energy/spin] [homo/inhomo] [deph/relax/stat/statdd] input.yml")
        log.error("\t WRONG EXECUTION PARAMETERS: PYDEPHASING STOPS")
timer.start_execution()
calc_type1 = parser.parse_args().ct1[0]
if calc_type1 == "energy":
    if mpi.rank == mpi.root:
        log.info("\n")
        log.info("\t " + p.sep)
        log.info("\t ENERGY LEVELS T2 CALCULATION")
        log.info("\n")
    # prepare energy dephasing calculation
    calc_type2 = parser.parse_args().ct2
    deph_type  = parser.parse_args().typ
    if calc_type2 == "homo":
        if deph_type == "deph":
            p.deph = True
            p.relax= False
            if mpi.rank == mpi.root:
                log.info("\t T2 CALCULATION -> STARTING")
                log.info("\t HOMOGENEOUS - DEPHASING")
                log.info("\n")
                log.info("\t " + p.sep)
        elif deph_type == "relax":
            p.relax = True
            p.deph  = False
            if mpi.rank == mpi.root:
                log.info("\t T1 CALCULATION -> STARTING")
                log.info("\t HOMOGENEOUS - RELAXATION")
                log.info("\n")
                log.info("\t " + p.sep)
        else:
            if mpi.rank == mpi.root:
                log.info("\n")
                log.info("\t " + p.sep)
                log.warning("\t CODE USAGE: \n")
                log.warning("\t -> python pydephasing [energy/spin] [homo/inhomo] [deph/relax/stat/statdd] input.yml")
                log.info("\t " + p.sep)
            log.error("\t deph_type : (1) deph or (2) relax")
        # read input file
        p.read_yml_data(yml_file)
        # compute auto correl. function first
        data = compute_homo_exc_dephas()
        # finalize calculation
        if mpi.rank == mpi.root:
            log.info("\n")
            log.info("\t " + p.sep)
            log.info("\t PRINT DATA ON FILES")
            # write T2 yaml files
            print_decoher_data(data)
            log.info("\t " + p.sep)
        mpi.comm.Barrier()
        # energy branch -> END
elif calc_type1 == "spin":
    # prepare spin dephasing calculation
    calc_type2 = parser.parse_args().ct2
    deph_type = parser.parse_args().typ
    if mpi.rank == mpi.root:
        if deph_type == "stat" or deph_type == "statdd":
            log.info("\t " + p.sep)
            log.info("\n")
            log.info("\t SPIN - STATIC CALCULATION")
            log.info("\n")
        else:
            log.info("\t " + p.sep)
            log.info("\n")
            log.info("\t SPIN - PHONON CALCULATION")
            log.info("\n")
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
                log.info("\t T2 CALCULATION -> STARTING")
                log.info("\t HOMOGENEOUS SPIN - DEPHASING")
                log.info("\n")
                log.info("\t " + p.sep)
        elif deph_type == "relax":
            p.deph = False
            p.relax= True
            if mpi.rank == mpi.root:
                log.info("\t T1 CALCULATION -> STARTING")
                log.info("\t HOMOGENEOUS SPIN - RELAXATION")
                log.info("\n")
                log.info("\t " + p.sep)
        else:
            if mpi.rank == mpi.root:
                log.info("\n")
                log.info("\t " + p.sep)
                log.warning("\t CODE USAGE: \n")
                log.warning("\t -> python pydephasing [energy/spin] [homo/inhomo] [deph/relax/stat/statdd] input.yml")
                log.info("\t " + p.sep)
            log.error("\t deph_type : (1) deph or (2) relax")
        # read input file
        p.read_yml_data(yml_file)
        # compute auto correl. function first
        T2_calc_handler = compute_homo_dephas()
        # finalize calculation
        if mpi.rank == mpi.root:
            log.info("-----------                   PRINT DATA ON FILES         --------------")
            # write T2 yaml files
            T2_calc_handler.print_decoherence_times()
        mpi.comm.Barrier()
        # homo spin branch -> END
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
                log.info("\t T2 CALCULATION -> STARTING")
                log.info("\t FULL HOMOGENEOUS SPIN - DEPHASING")
                log.info("\n")
                log.info("\t " + p.sep)
        elif deph_type == "relax":
            p.deph = False
            p.relax= True
            if mpi.rank == mpi.root:
                log.info("\t T2 CALCULATION -> STARTING")
                log.info("\t FULL HOMOGENEOUS SPIN - RELAXATION")
                log.info("\n")
                log.info("\t " + p.sep)
        else:
            if mpi.rank == mpi.root:
                log.info("\n")
                log.info("\t " + p.sep)
                log.warning("\t CODE USAGE: \n")
                log.warning("\t -> python pydephasing [energy/spin] [homo/inhomo] [deph/relax/stat/statdd] input.yml")
                log.info("\t " + p.sep)
            log.error("\t deph_type : (1) deph or (2) relax")
        # read input file
        p.read_yml_data(yml_file)
        # compute auto correl. function first
        T2_calc_handler = compute_full_dephas()
        # finalize calculation
        if mpi.rank == mpi.root:
            log.info("\n")
            log.info("\t" + p.sep)
            log.info("\t PRINT DATA ON FILES")
            # write T2 yaml files
            T2_calc_handler.print_decoherence_times()
            log.info("\t" + p.sep)
            log.info("\n")
        mpi.comm.Barrier()
        # full spin branch -> END
    # --------------------------------------------------------------
    # 
    #    SIMPLE INHOMOGENEOUS CALC. (HFI ONLY)
    #
    # --------------------------------------------------------------
    elif calc_type2 == "inhomo":
        # check calc type
        if deph_type != "stat" and deph_type != "statdd":
            # read file
            p.read_yml_data(yml_file)
            if deph_type == "deph":
                p.deph = True
                p.relax= False
                if mpi.rank == 0:
                    log.info("\t T2* CALCULATION -> STARTING")
                    log.info("\t INHOMOGENEOUS SPIN - DEPHASING")
                    log.info("\n")
                    log.info("\t " + p.sep)
            elif deph_type == "relax":
                p.deph = False
                p.relax= True
                if mpi.rank == 0:
                    log.info("\t T1* CALCULATION -> STARTING")
                    log.info("\t INHOMOGENEOUS SPIN - RELAXATION")
                    log.info("\n")
                    log.info("\t " + p.sep)
            else:
                if mpi.rank == mpi.root:
                    log.info("\n")
                    log.info("\t " + p.sep)
                    log.warning("\t CODE USAGE: \n")
                    log.warning("\t -> python pydephasing [energy/spin] [homo/inhomo] [deph/relax/stat/statdd] input.yml")
                    log.info("\t " + p.sep)
                log.error("\t deph_type : (1) deph or (2) relax")
            # compute the dephas. time
            T2_calc_handler = compute_hfi_dephas()
            # finalize calculation
            if mpi.rank == mpi.root:
                log.info("\n")
                log.info("\t" + p.sep)
                log.info("\t PRINT DATA ON FILES")
                # write T2 yaml files
                T2_calc_handler.print_decoherence_times()
                log.info("\t" + p.sep)
                log.info("\n")
            mpi.comm.Barrier()
            # inhomo spin branch -> END
        elif deph_type == "stat" or deph_type == "statdd":
            if deph_type == "statdd":
                p.dyndec = True
            # read file
            p.read_yml_data(yml_file)
            # static HFI calculation
            if mpi.rank == 0:
                log.info("\t T2* CALCULATION -> STARTING")
                log.info("\t INHOMOGENEOUS STATIC SPIN - DEPHASING")
                log.info("\n")
                log.info("\t " + p.sep)
            # compute dephasing time
            T2_calc_handler = compute_hfi_stat_dephas()
            # finalize calculation
            if mpi.rank == mpi.root:
                log.info("\n")
                log.info("\t" + p.sep)
                log.info("\t PRINT DATA ON FILES")
                # write T2 yaml files
                T2_calc_handler.print_decoherence_times()
                log.info("\t" + p.sep)
                log.info("\n")
            mpi.comm.Barrier()
            # inhomo static branch -> END
        else:
            if mpi.rank == 0:
                log.info("\n")
                log.info("\t " + p.sep)
                log.warning("\t CODE USAGE: \n")
                log.warning("\t -> python pydephasing [energy/spin] [homo/inhomo] [deph/relax/stat/statdd] input.yml")
                log.info("\t " + p.sep)
            log.error("\t WRONG ACTION FLAG TYPE: PYDEPHASING STOPS HERE")
elif calc_type1 == "init":
    # read data file
    order = parser.parse_args().o
    # read data
    p.read_yml_data(yml_file)
    if mpi.rank == mpi.root:
        log.info("\n")
        log.info("\t " + p.sep)
        log.info("\t BUILD DISPLACED STRUCTS.")
    if int(order) == 1:
        if mpi.rank == mpi.root:
            gen_poscars(p.max_dist_defect, p.defect_index)
    elif int(order) == 2:
        if mpi.rank == mpi.root:
            gen_2ndorder_poscar(p.max_dist_defect, p.defect_index, p.max_dab)
    else:
        if mpi.rank == mpi.root:
            log.info("\n")
            log.info("\t " + p.sep)
            log.warning("\t WRONG ORDER FLAG")
            log.warning("\t ORDER = 1 or 2")
        log.error("\t WRONG DISPLACEMENT ORDER FLAG")
    if mpi.rank == mpi.root:
        log.info("\t " + p.sep)
        log.info("\n")
elif calc_type1 == "--post":
    # post process output data from VASP
    pass
else:
    if mpi.rank == mpi.root:
        log.info("\n")
        log.info("\t " + p.sep)
        log.warning("\t CALC. TYPE NOT RECOGNIZED")
        log.warning("\t QUIT PROGRAM")
        log.info("\t " + p.sep)
    log.error("\t WRONG CALC. FLAG")
# end execution
timer.end_execution()
if mpi.rank == mpi.root:
    log.info("\t ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    log.info("\t ++++++                                                                                  ++++++")
    log.info("\t ++++++                    CALCULATION SUCCESSFULLY COMPLETED                            ++++++")
    log.info("\t ++++++                                                                                  ++++++")
    log.info("\t ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
mpi.finalize_procedure()