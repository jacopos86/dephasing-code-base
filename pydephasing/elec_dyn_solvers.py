# This is the main subroutine
# for non Markovian calculation of electronic systems
#
from pydephasing.set_param_object import p
from pydephasing.parallelization.mpi import mpi
from pydephasing.utilities.log import log
from pydephasing.build_unpert_struct import build_jdftx_gs_elec_struct
from pydephasing.wannier_interface.wannier import Wannier
from pydephasing.phonons_module import JDFTxPhonons
from pydephasing.q_grid import jdftx_qgridClass
from pydephasing.atomic_list_struct import atoms
from pydephasing.electronic_hamiltonian import electronic_hamiltonian

#
def solve_elec_model_dyn():
    if mpi.rank == mpi.root:
        log.info("\t " + p.sep)
        log.info("\n")
        log.info("\t PERFORM REAL TIME DYNAMICS")
        log.info("\t MODEL SYSTEM WITH DEFORMATION POTENTIAL")
        log.info("\n")




def solve_elec_dyn_VASP_data():
    '''
    use VASP data to perform RT dynamics
    '''






def solve_elec_dyn_JDFTx_data():
    '''
    use JDFTx data to perform RT dynamics
    '''
    atoms.set_atoms_data(p.work_dir)
    if mpi.rank == mpi.root:
        log.info("\t " + p.sep)
        log.info("\n")
        log.info("\t PERFORM REAL TIME DYNAMICS")
        log.info("\t COLLECT DATA FROM JDFTx CALCULATION")
        log.info("\n")
        log.info("\t " + p.sep)
        atoms.print_atoms_info()
        log.info("\t " + p.sep)
    # set electronic structure
    elec_struct = build_jdftx_gs_elec_struct(p.work_dir)
    He = electronic_hamiltonian(Ewin_Ha=p.elec_win, wann=p.wannier_interp)
    He.set_energy_spectrum(elec_struct)
    He.plot_band_structure()
    He.set_H0_matr()
    # set phonon structure
    if p.dynamical_mode[1] > 0:
        # read phonons data
        # set q grid
        qgr = jdftx_qgridClass(p.work_dir)
        qgr.set_qgrid()
        # if we need to compute e-ph interactions
        if mpi.rank == mpi.root:
            log.info("\t " + p.sep)
            log.info("\t COLLECT PHONONS INFORMATION")
            log.info("\t " + p.sep)
            log.info("\n")
        ph = JDFTxPhonons(p.work_dir, p.TR_SYM)
        ph.read_ph_hamilt(qgr=qgr)
        ph.get_ph_supercell()
        ph.compute_eq_ph_angular_momentum_dispersion(qgr)
        ph.compute_full_ph_angular_momentum_matrix(qgr)
    # clean up
    He.clean_up()