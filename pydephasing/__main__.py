from pydephasing.set_param_object import p
from pydephasing.create_displ_struct_files import gen_poscars, gen_2ndorder_poscar
from pydephasing.parallelization.mpi import mpi
from pydephasing.utilities.log import log
from pydephasing.utilities.timer import timer
from pydephasing.utilities.input_parser import parser
from pydephasing.calculation_drivers import energy_linewidth_driver, spin_qubit_driver, elec_system_driver

def run():
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
    #  input structure :
    #  pydephasing -ct1 [LR, LBLD, NMARK, init, postproc] -co [spin, energy] -ct2 [stat,statdd,homo,inhomo,full] - yml_inp [input]
    # 
    yml_file = parser.parse_args().yml_inp[0]
    if yml_file is None:
        log.error("-> yml file name missing")
    nargs = 2
    if parser.parse_args().co is not None:
        nargs += 1
    if parser.parse_args().ct2 is not None:
        nargs += 1
    if nargs < 4:
        if parser.parse_args().ct1[0] == "init" or parser.parse_args().ct1[0] == "postproc":
            log.debug("\t observable: " + str(parser.parse_args().ct1[0]))
        else:
            if mpi.rank == mpi.root:
                log.info("\n")
                log.info("\t " + p.sep)
                log.warning("\t CODE USAGE: \n")
                log.warning("\t -> python pydephasing -ct1 [LR, RT, init, postproc] -co [spin, energy] -ct2 [inhomo,stat,statdd,homo,full] - yml_inp [input]")
            log.error("\t WRONG EXECUTION PARAMETERS: PYDEPHASING STOPS")
    else:
        if mpi.rank == mpi.root:
            log.debug("\t observable: " + str(parser.parse_args().co[0]))
            log.debug("\t calculation type (1): " + str(parser.parse_args().ct1[0]))
            log.debug("\t calculation type (2): " + str(parser.parse_args().ct2))
    timer.start_execution()

    #
    #  call different drivers
    #

    calc_type1 = parser.parse_args().ct1[0]
    if calc_type1 == "init":
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
    elif calc_type1 == "init_RT":
        # parse input file
        p.read_yml_data(yml_file)
    elif calc_type1 == "postproc":
        # post process output data from VASP
        pass
    elif calc_type1 == "LR" or calc_type1 == "RT" or calc_type1 == "QUANTUM":
        co = parser.parse_args().co[0]
        if co == "energy-lw":
            energy_linewidth_driver(yml_file)
        elif co == "spin-qubit":
            spin_qubit_driver(yml_file)
        elif co == "elec-sys":
            elec_system_driver(yml_file)
        else:
            if mpi.rank == mpi.root:
                log.info("\n")
                log.info("\t " + p.sep)
                log.warning("\t CALC. TYPE 1 NOT RECOGNIZED")
                log.warning("\t QUIT PROGRAM")
                log.info("\t " + p.sep)
            log.error("\t WRONG CALC. FLAG")
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

if __name__ == "__main__":
    run()