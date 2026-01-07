# This is the main subroutine
# for non Markovian calculation of electronic systems
#
from pydephasing.set_param_object import p
from pydephasing.parallelization.mpi import mpi
from pydephasing.utilities.log import log
from pydephasing.build_unpert_struct import build_jdftx_gs_elec_struct
from pydephasing.wannier_interface.wannier import Wannier
from pydephasing.phonons_module import JDFTxPhonons, PhononsStructModel
from pydephasing.q_grid import jdftx_qgridClass
from pydephasing.atomic_list_struct import atoms
from pydephasing.electronic_hamiltonian import electronic_hamiltonian, model_electronic_hamiltonian
from pydephasing.elec_dens_matr import elec_dmatr

#
def solve_elec_model_dyn():
    if mpi.rank == mpi.root:
        log.info("\t " + p.sep)
        log.info("\n")
        log.info("\t PERFORM REAL TIME DYNAMICS")
        log.info("\t MODEL SYSTEM WITH DEFORMATION POTENTIAL")
        log.info("\n")
        log.info("\t " + p.sep)
    # ============================================================
    # 1. Build MODEL electronic Hamiltonian
    # ============================================================
    He = model_electronic_hamiltonian(
        elec_bands=p.elec_bands,                     # 1 or 2
        nkpt=p.nkpt,
        eff_mass=p.elec_eff_mass,                    # list
        band_offset=p.elec_band_offset,              # list
        chem_pot=p.chem_pot,
        Ewin_Ha=p.elec_win
    )
    He.set_energy_spectrum()
    He.plot_band_structure()
    He.set_H0_matr()
    if mpi.rank == mpi.root:
        log.info("\t MODEL ELECTRONIC HAMILTONIAN initialized")
        log.info("\t " + p.sep)
        log.info("\n")
    # set MODEL density matrix
    rho_e = elec_dmatr(p.smearing, p.Te)
    # ============================================================
    # 2. Q grid && Model phonons (acoustic branch)
    # ============================================================
    ph = PhononsStructModel()
    print(ph.nmodes)
    exit()
    nq = p.model_nq
    qvals = np.linspace(-p.qmax, p.qmax, nq)
    q_vectors = np.zeros((nq, 3))
    q_vectors[:, 0] = qvals
    phonon_freqs = np.zeros((nq, 1))
    phonon_freqs[:, 0] = p.model_sound_velocity * np.abs(qvals)
    phonon_polarizations = np.zeros((nq, 1, 3))
    phonon_polarizations[:, 0, 0] = 1.0  # longitudinal
    # ============================================================
    # 3. Deformation potential electron–phonon coupling
    # ============================================================
    ep = DeformationPotentialElectronPhonon(
        deformation_potentials=p.model_deformation_potential,
        density=p.model_density,
        volume=p.model_volume
    )
    g_ql = ep.compute_gql(
        q_vectors=q_vectors,
        phonon_freqs=phonon_freqs,
        phonon_polarizations=phonon_polarizations
    )
    if mpi.rank == mpi.root:
        log.info("\t Deformation potential coupling computed")
    # ============================================================
    # 4. Ready for non-Markovian propagation
    # ============================================================

    # TODO:
    # dynamics_solver(He, g_ql)

    He.clean_up()

# ====================================================
#
#     VASP dynamic solver
#
# ====================================================

def solve_elec_dyn_VASP_data():
    '''
    use VASP data to perform RT dynamics
    '''


# ====================================================
#
#     JDFTX dynamic solver
#
# ====================================================

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
    He.print_info()
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