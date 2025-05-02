from pydephasing.real_time_solver_base import RealTimeSolver
from pydephasing.log import log

#
#   set one-phonon processes solver
#

class OnephSolver(RealTimeSolver):
    def __init__(self, mode):
        self.LDBLD = None
        self.NMARK = None
        super(OnephSolver, self).__init__()
        if mode == 1:
            self.LDBLD = True
            self.NMARK = False
        elif mode == 2:
            self.LDBLD = False
            self.NMARK = True
        else:
            log.error("\t incorrect mode : mode = [1,2]")