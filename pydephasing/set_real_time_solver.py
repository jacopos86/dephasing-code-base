from pydephasing.input_parser import parser
from pydephasing.mpi import mpi
from pydephasing.log import log
from pydephasing.set_param_object import p
from pydephasing.Liouville_solver import LiouvilleSolver

#
#   This module set the real time solver
#

def set_real_time_solver():
    calc_type1 = parser.parse_args().ct1[0]
    # no real time solver
    if calc_type1 == "LR":
        return None
    elif calc_type1 == "LBLD":
        if mpi.rank == mpi.root:
            log.info("\n")
            log.info("\t " + p.sep)
            log.info("\t SETTING LINDBLAD SOLVER")
            log.info("\t " + p.sep)
            log.info("\n")
        return None
    elif calc_type1 == "NMARK":
        if mpi.rank == mpi.root:
            log.info("\n")
            log.info("\t " + p.sep)
            log.info("\t SETTING NON MARKOVIAN SOLVER")
            log.info("\t " + p.sep)
            log.info("\n")
        return LiouvilleSolver()