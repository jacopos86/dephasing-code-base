from pydephasing.input_parser import parser
from pydephasing.mpi import mpi
from pydephasing.log import log
from pydephasing.set_param_object import p
from pydephasing.Liouville_solver import LiouvilleSolverSpin, LiouvilleSolverHFI
from pydephasing.oneph_process_solver import OnephSolver

#
#   This module set the real time solver
#

def set_real_time_solver(HFI_CALC):
    calc_type1 = parser.parse_args().ct1[0]
    # assert calc_type1 is set to RT
    assert calc_type1 == "RT"
    if mpi.rank == mpi.root:
        log.info("\n")
        log.info("\t " + p.sep)
        log.info("\t SETTING UP REAL TIME SOLVER")
        log.info("\t " + p.sep)
        log.info("\n")
    # read dynamical mode
    # dynamical[i] -> i=0 : one-phonon process - i=1 : two-phonon process 
    if p.dynamical_mode[0] == 0 and p.dynamical_mode[1] == 0:
        if mpi.rank == mpi.root:
            log.info("\t FULLY COHERENT EVOLUTION ")
            log.info("\t " + p.sep)
            log.info("\n")
        if HFI_CALC:
            return LiouvilleSolverHFI()
        else:
            return LiouvilleSolverSpin()
    elif p.dynamical_mode[0] == 1 and p.dynamical_mode[1] == 0:
        if mpi.rank == mpi.root:
            log.info("\t LINDBLAD SOLVER - ELEC-PH ORDER 1")
            log.info("\t " + p.sep)
            log.info("\n")
        return OnephSolver(p.dynamical_mode[0])
    elif p.dynamical_mode[0] == 2 and p.dynamical_mode[1] == 0:
        if mpi.rank == mpi.root:
            log.info("\t COMPLETE NON MARKOVIAN SOLVER - ELEC-PH ORDER 1")
            log.info("\t " + p.sep)
            log.info("\n")