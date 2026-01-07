from abc import ABC
import numpy as np
from pathlib import Path
import yaml

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

class ElecPhDynamicsSolver:
    """
    Skeleton for real-time evolution of electronic and electron-phonon density matrices.
    """

    def __init__(self, H_elec, e_ph):
        """
        H_elec : instance of electronic_hamiltonian
        e_ph   : instance of ElectronPhononClass (or deformation potential model)
        """
        self.H_elec = H_elec
        self.e_ph = e_ph

        # Number of bands, phonon modes
        self.nbnd = H_elec.nbnd
        self.n_ph = getattr(e_ph, "nm", 1)

        # PETSc objects
        self.rho = None       # electronic density matrix (PETSc Mat)
        self.rho_ep = None    # e-ph density matrices (could be numpy or PETSc, depends on GPU)
    
    def initialize_density_matrices(self):
        """Initialize electronic and electron-phonon density matrices."""
        # Electronic DM
        self.rho = PETSc.Mat().createDense(size=(self.nbnd, self.nbnd))
        self.rho.setUp()
        self.rho.set(0.0)
        self.rho.assemble()

        # Electron-phonon DM
        self.rho_ep = np.zeros((self.nbnd, self.nbnd, self.n_ph), dtype=np.complex128)

        if mpi.rank == mpi.root:
            log.info("Density matrices initialized")

    def compute_drho_dt(self):
        """Compute time derivative of density matrices. Replace with actual physics."""
        # Example placeholders
        drho_dt = PETSc.Mat().createDense(size=(self.nbnd, self.nbnd))
        drho_dt.setUp()
        drho_dt.set(0.0)
        drho_dt.assemble()

        drho_ep_dt = np.zeros_like(self.rho_ep)

        return drho_dt, drho_ep_dt

    def evolve(self, t_final, dt):
        """Simple Euler integration loop (replace with PETSc TS for serious use)."""
        n_steps = int(t_final / dt)
        for step in range(n_steps):
            drho_dt, drho_ep_dt = self.compute_drho_dt()
            
            # Update density matrices (Euler step)
            self.rho.axpy(dt, drho_dt)       # rho = rho + dt * drho_dt
            self.rho_ep += dt * drho_ep_dt   # rho_ep = rho_ep + dt * drho_ep_dt

            if mpi.rank == mpi.root and step % 10 == 0:
                log.info(f"Step {step}/{n_steps} completed")

    # Optional: GPU support can be added in separate subclass
    # class GPU_ModelDynamicsSolver(ModelDynamicsSolver): ...