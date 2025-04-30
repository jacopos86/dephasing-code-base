from pydephasing.real_time_solver_base import RealTimeSolver

#
#    This module implements
#    the Liouville solver for DM dynamics
#

class LiouvilleSolver(RealTimeSolver):
    def __init__(self):
        super(LiouvilleSolver, self).__init__()