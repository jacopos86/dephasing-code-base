from abc import ABC
import numpy as np
from pathlib import Path
from petsc4py import PETSc
import yaml
from pydephasing.parallelization.mpi import mpi
from pydephasing.utilities.log import log
from pydephasing.set_param_object import p

#  TODO MUST BE REPLACED WITH SPIN DYNAMICS ?
#    This module implements
#    the real time solver class
#    Takes Hamiltonian + density matrix object -> real time dynamics
#    ABSTRACT class

class RealTimeSolver(ABC):
    def __init__(self):
        pass
    # set time arrays
    def set_time_array(self, dt, T):
        # n. time steps
        nt = int(T / dt)
        time = np.linspace(0., T, nt)
        # build dense array
        #nt2 = int(T / (dt/2.))
        #self.time_dense = np.linspace(0., T, nt2)
        # everything must be in ps
        #self.T = T
        #self.dt = dt
        return time
    # write time object to file
    def write_obj_to_file(self, t, var_oft, units, file_name):
        # write data on file
        fil = Path(file_name)
        if not fil.exists():
            data = {'time': t, 'quantity': var_oft, 'units': units}
            with open(file_name, 'w') as out_file:
                yaml.dump(data, out_file)

# ========================================================================
# 
#
#    GENERAL ELECTRON DYNAMICS SOLVER
#          IT MUST SUPPORT:
#    1)  LINDBLAD DYNAMICS
#    2) FULL NON MARKOVIAN
#
#    FOR DIFFERENT INTERACTIONS
#    ELEC-ELEC + ELEC-PH + ELEC-RAD
#
#
# ========================================================================

class ElecPhDynamicSolverBase:
    """
    Generic PETSc TS driver.
    Subclasses define:
      - _build_initial_state
      - _rhs
      - monitor
    """
    TITLE = None
    def __init__(self, evol_params):
        # input parameters
        self.dt = evol_params.get("time_step")                # ps
        self.nsteps = evol_params.get("num_steps")
        self.save_every = evol_params.get("save_every")
        self.solver_type = evol_params.get("solver_type", "RK4").upper()
        # PETSc ts object
        self.ts = PETSc.TS().create(comm=PETSc.COMM_SELF)
        self._configure_ts()
        # runtime context
        self._ik = None
        self._isp = None
        # internal global variables
        self._nbnd = None
        self._rho_e = None
    # --------------------------------------------------
    # PETSc configuration
    # --------------------------------------------------
    def _configure_ts(self):
        """ Configure PETSc TS based on solver type """
        if self.solver_type == "RK4":
            self.ts.setType(PETSc.TS.Type.RK)
            self.ts.setRKType(PETSc.TS.RKType.RK4)
        elif self.solver_type == "RK45":
            self.ts.setType(PETSc.TS.Type.RK)
            self.ts.setRKType(PETSc.TS.RKType.RK45)
        elif self.solver_type == "EULER":
            self.ts.setType(PETSc.TS.Type.EULER)
        elif self.solver_type == "CN":              # Crank-Nicholson
            self.ts.setType(PETSc.TS.Type.BEULER)   # backward Euler
            self.ts.setExactFinalTime(PETSc.TS.ExactFinalTime.MATCHSTEP)
        elif self.solver_type == "ARKIMEX":
            self.ts.setType(PETSc.TS.Type.ARKIMEX)
            self.ts.getKSP().getPC().setType('jacobi')
            self.ts.setExactFinalTime(PETSc.TS.ExactFinalTime.MATCHSTEP)
        else:
            log.error(f"Unknown solver type {self.solver_type}")
    # reset solver
    def _reset_ts(self, y):
        """ Reset the PETSc TS object to start a new propagation."""
        self.ts.reset()                   # clear previous solver state
        self.ts.setTime(0.0)              # reset time
        self.ts.setStepNumber(0)          # step number
        self.ts.setTimeStep(self.dt)
        self.ts.setMaxSteps(self.nsteps)
        self.ts.setSolution(y)            # set initial solution
    # summary mode
    def summary(self):
        if mpi.rank == mpi.root:
            log.info("\n")
            log.info("\t " + p.sep)
            log.info("\t " + self.TITLE)
            log.info(f"\t TIME STEP: {self.dt} ps")
            log.info(f"\t NUMBER OF STEPS: {self.nsteps}")
            log.info("\t REAL TIME SOLVER: " + self.solver_type.upper())
            log.info("\t " + p.sep)
            log.info("\n")
    # --------------------------------------------------
    # overridden methods
    # --------------------------------------------------
    def _set_initial_state(self):
        raise NotImplementedError
    def _unpack_state(self, y):
        raise NotImplementedError
    def _rhs(self, ts, t, y, ydot):
        raise NotImplementedError
    def _rhs_linear(self, ts, t, y, ydot):
        raise NotImplementedError
    def _initialize_linear_ODE_solver(self, y):
        raise NotImplementedError
    def _initialize_ODE_solver(self, y):
        raise NotImplementedError
    def monitor(self, ts, step, t, y):
        raise NotImplementedError
    def propagate(self, *args, **kwargs):
        raise NotImplementedError