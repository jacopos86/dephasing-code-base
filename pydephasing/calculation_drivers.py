from parallelization.mpi import mpi
from utilities.log import log
from utilities.input_parser import parser
from pydephasing.set_param_object import p
from pydephasing.compute_LR_spin_decoher import compute_spin_dephas
from pydephasing.compute_hfi_dephas_stat import compute_hfi_stat_dephas
from pydephasing.non_markov_dephas import compute_nmark_dephas
from pydephasing.compute_LR_elec_decoher import compute_elec_dephas
from quantum.compute_QA_spin_decoher import compute_dephas_QA

#
#   different calculation drivers
#   functions called from __main__
#

def energy_linewidth_driver(yml_file):
    if mpi.rank == mpi.root:
        log.info("\n")
        log.info("\t " + p.sep)
        log.info("\t ENERGY LEVELS T2 CALCULATION")
        log.info("\n")
    # prepare energy dephasing calculation
    '''
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
    '''

#
#  spin qubit driver
#

def spin_qubit_driver(yml_file):
    # prepare spin dephasing calculation
    calc_type1 = parser.parse_args().ct1[0]
    calc_type2 = parser.parse_args().ct2
    if calc_type1 == "LR":
        if mpi.rank == mpi.root:
            log.info("\t " + p.sep)
            log.info("\n")
            log.info("\t SPIN - FERMI GOLDEN RULE CALCULATION")
            log.info("\n")
        # ------------------------------------------------------------
        #
        #    CHECK calc_type2 variable
        #
        # ------------------------------------------------------------
        if calc_type2 == "homo":
            if mpi.rank == mpi.root:
                log.info("\t " + p.sep)
                log.info("\n")
                log.info("\t SPIN - PHONON HOMOGENEOUS CALCULATION")
                log.info("\n")
            # --------------------------------------------------------------
            # 
            #    SIMPLE HOMOGENEOUS CALC. (ZFS ONLY)
            #
            # --------------------------------------------------------------
            ZFS_CALC = True
            HFI_CALC = False
            if mpi.rank == mpi.root:
                log.info("\t T2 CALCULATION -> STARTING")
                log.info("\t HOMOGENEOUS SPIN - DEPHASING")
                log.info("\t ZFS_CALC: " + str(ZFS_CALC))
                log.info("\t HFI_CALC: " + str(HFI_CALC))
                log.info("\n")
                log.info("\t " + p.sep)
        elif calc_type2 == "inhomo":
            # --------------------------------------------------------------
            # 
            #    SIMPLE INHOMOGENEOUS CALC. (HFI ONLY)
            #
            # --------------------------------------------------------------
            ZFS_CALC = False
            HFI_CALC = True
            if mpi.rank == mpi.root:
                log.info("\t " + p.sep)
                log.info("\n")
                log.info("\t SPIN - PHONON INHOMOGENEOUS CALCULATION")
                log.info("\n")
                log.info("\t T2 CALCULATION -> STARTING")
                log.info("\t INHOMOGENEOUS SPIN - DEPHASING")
                log.info("\t ZFS_CALC: " + str(ZFS_CALC))
                log.info("\t HFI_CALC: " + str(HFI_CALC))
                log.info("\n")
                log.info("\t " + p.sep)
        elif calc_type2 == "full":
            ZFS_CALC = True
            HFI_CALC = True
            if mpi.rank == mpi.root:
                log.info("\t " + p.sep)
                log.info("\n")
                log.info("\t SPIN - PHONON INHOMOGENEOUS CALCULATION")
                log.info("\n")
                log.info("\t T2 CALCULATION -> STARTING")
                log.info("\t INHOMOGENEOUS SPIN - DEPHASING")
                log.info("\t ZFS_CALC: " + str(ZFS_CALC))
                log.info("\t HFI_CALC: " + str(HFI_CALC))
                log.info("\n")
                log.info("\t " + p.sep)
        elif calc_type2 == "stat" or calc_type2 == "statdd":
            if mpi.rank == mpi.root:
                log.info("\t " + p.sep)
                log.info("\n")
                log.info("\t SPIN - STATIC CALCULATION")
                log.info("\n")
            if calc_type2 == "statdd":
                p.dyndec = True
            # static HFI calculation
            if mpi.rank == mpi.root:
                log.info("\t T2* CALCULATION -> STARTING")
                log.info("\t INHOMOGENEOUS STATIC SPIN - DEPHASING")
                log.info("\n")
                log.info("\t " + p.sep)
        elif calc_type2 == "electronic":
            if mpi.rank == mpi.root:
                log.info("\t " + p.sep)
                log.info("\n")
                log.info("\t ELECTRONIC DEPHASING CALCULATION")
                log.info("\n")
        else:
            if mpi.rank == mpi.root:
                log.info("\n")
                log.info("\t " + p.sep)
                log.warning("\t CODE USAGE: \n")
                log.warning("\t -> python pydephasing -ct1 [LR, LBLD, NMARK, init, postproc] -co [spin, energy] -ct2 [inhomo,stat,statdd,homo,full] - yml_inp [input]")
                log.info("\t " + p.sep)
            log.error("\t calc_type2 wrong: " + calc_type2)
        #
        #    read input file
        #
        p.read_yml_data(yml_file)
        # compute auto correl. function first
        if calc_type2 == "stat" or calc_type2 == "statdd":
            T2_calc_handler = compute_hfi_stat_dephas()
        elif calc_type2 == "electronic":
            T2_calc_handler = compute_elec_dephas()
        else:
            T2_calc_handler = compute_spin_dephas(ZFS_CALC, HFI_CALC)
        #
        # finalize calculation
        #
        if mpi.rank == mpi.root:
            log.info("\n")
            log.info("\t" + p.sep)
            log.info("\t PRINT DATA ON FILES")
            # write T2 yaml files
            T2_calc_handler.print_decoherence_times()
            log.info("\t" + p.sep)
            log.info("\n")
        mpi.comm.Barrier()
    # --------------------------------------------------------------
    # 
    #    FULL CALC. -> REAL TIME
    #
    # --------------------------------------------------------------
    elif calc_type1 == "RT":
        if mpi.rank == mpi.root:
            log.info("\t T2 CALCULATION -> STARTING")
            log.info("\t REAL TIME DYNAMICS")
            log.info("\n")
            log.info("\t " + p.sep)
        # read input file
        p.read_yml_data(yml_file)
        p.check_consistency()
        # compute dephas
        if calc_type2 == "homo":
            if mpi.rank == mpi.root:
                log.info("\t " + p.sep)
                log.info("\n")
                log.info("\t SPIN - PHONON HOMOGENEOUS CALCULATION")
                log.info("\n")
            # --------------------------------------------------------------
            # 
            #    SIMPLE HOMOGENEOUS CALC. (ZFS ONLY)
            #
            # --------------------------------------------------------------
            ZFS_CALC = True
            HFI_CALC = False
        elif calc_type2 == "inhomo":
            # --------------------------------------------------------------
            # 
            #    SIMPLE INHOMOGENEOUS CALC. (HFI ONLY)
            #
            # --------------------------------------------------------------
            ZFS_CALC = False
            HFI_CALC = True
            if mpi.rank == mpi.root:
                log.info("\t " + p.sep)
                log.info("\n")
                log.info("\t SPIN - PHONON INHOMOGENEOUS CALCULATION")
                log.info("\n")
        elif calc_type2 == "full":
            ZFS_CALC = True
            HFI_CALC = True
            if mpi.rank == mpi.root:
                log.info("\t " + p.sep)
                log.info("\n")
                log.info("\t SPIN - PHONON INHOMOGENEOUS CALCULATION")
                log.info("\n")
        else:
            if mpi.rank == mpi.root:
                log.warning("\t REAL TIME DYNAMICS -> calc_type2 : homo/inhomo/full")
            log.error("\t WRONG ACTION FLAG TYPE: PYDEPHASING STOPS HERE")
        #
        #  START NON MARKOVIAN CALCULATION
        #
        T2_calc_handler = compute_nmark_dephas(ZFS_CALC, HFI_CALC)
    elif calc_type1 == "QUANTUM":
        if mpi.rank == mpi.root:
            log.info("\n")
            log.info("\t " + p.sep)
            log.info("\n")
            log.info("\t " + " PERFORM QUANTUM CALCULATION")
            log.info("\n")
            log.info("\t " + p.sep)
            log.info("\n")
        # read input file
        p.read_yml_data(yml_file)
        # compute auto correl. function first
        if calc_type2 == "homo":
            ZFS_CALC = True
            HFI_CALC = False
        elif calc_type2 == "inhomo":
            ZFS_CALC = False
            HFI_CALC = True
        elif calc_type2 == "full":
            ZFS_CALC = True
            HFI_CALC = True
        else:
            if mpi.rank == mpi.root:
                log.info("\n")
                log.info("\t " + p.sep)
                log.warning("\t CODE USAGE: \n")
                log.warning("\t -> python pydephasing -ct1 [LR, LBLD, NMARK, init, postproc] -co [spin, energy] -ct2 [inhomo,stat,statdd,homo,full] - yml_inp [input]")
                log.info("\t " + p.sep)
            log.error("\t calc_type2 wrong: " + calc_type2)
        if mpi.rank == mpi.root:
            log.info("\n")
            log.info("\t T2 CALCULATION -> STARTING")
            log.info("\t ZFS_CALC: " + str(ZFS_CALC))
            log.info("\t HFI_CALC: " + str(HFI_CALC))
            log.info("\n")
            log.info("\t " + p.sep)
        T2_calc_handler = compute_dephas_QA(ZFS_CALC, HFI_CALC)
    else:
        if mpi.rank == mpi.root:
            log.info("\n")
            log.info("\t " + p.sep)
            log.warning("\t CODE USAGE: \n")
            log.warning("\t -> python pydephasing -ct1 [LR, RT, init, postproc] -co [spin, energy] -ct2 [inhomo,stat,statdd,homo,full] - yml_inp [input]")
            log.info("\t " + p.sep)
        log.error("\t WRONG ACTION FLAG TYPE: PYDEPHASING STOPS HERE")