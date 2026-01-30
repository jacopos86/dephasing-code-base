# This is the main subroutine
# for non Markovian calculation of electronic systems
#
import numpy as np
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
from pydephasing.elec_ph_inter import DeformationPotentialElectronPhonon
from pydephasing.observables import ObservablesElectronicModel
from pydephasing.elec_light_inter import ElectronLightCouplModel
from pydephasing.real_time.set_real_time_solver import set_real_time_electronic_solver

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
        Ewin_Ha=p.elec_win,
        sys_size=p.L
    )
    He.set_energy_spectrum()
    He.set_H0_matr()
    if mpi.rank == mpi.root:
        log.info("\t MODEL ELECTRONIC HAMILTONIAN initialized")
        log.info("\t " + p.sep)
        log.info("\n")
    # set MODEL density matrix
    rho_e = elec_dmatr(p.smearing, p.Te, p.Nel)
    rho_e.initialize_dmatr(He)
    rho_e.summary()
    # plot band structure
    He.plot_band_structure()
    # phonons section
    if p.dynamical_mode[1] > 0:
        # ============================================================
        # 2. Q grid && Model phonons (acoustic branch)
        # ============================================================
        qgr = np.copy(He.kgr)
        if mpi.rank == mpi.root:
            log.info("\t " + p.sep)
            log.info(f"\t q grid shape: {qgr.shape}")
            log.info("\t " + p.sep)
        ph = PhononsStructModel(p.sound_velocity)
        ph.compute_phonon_DOS(qgr, p.ph_DOS_plot)
        ph.compute_energy_dispersion(qgr)
        ph.summary()
        # ============================================================
        # 3. Deformation potential electron–phonon coupling
        # ============================================================
        eph = DeformationPotentialElectronPhonon(
            eph_params=p.eph_params,
            nBands=p.elec_bands,
            sys_size=p.L
        )
        eph.summary()
        if mpi.rank == mpi.root:
            log.info("\t Deformation potential coupling computed")
    # ============================================================
    # 4. Spin operators + SO interaction
    # ============================================================
    Observ = ObservablesElectronicModel(basis_set=None)
    Observ.set_Spin_operators(p.elec_bands, p.nkpt)
    # ============================================================
    # 5. Electric dipole
    # ============================================================
    elc = ElectronLightCouplModel(
        dipole_strengths=p.dipole_coeff
    )
    # ============================================================
    # 6. Ready for non-Markovian propagation
    # ============================================================
    RT_solver = set_real_time_electronic_solver()
    RT_solver.summary()
    rho_e = RT_solver.propagate(He, rho_e)
    print(rho_e.traces[50,0,:])
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
    elec_struct = build_jdftx_gs_elec_struct(p.work_dir, p.gamma_point)
    He = electronic_hamiltonian(Ewin_Ha=p.elec_win, wann=p.wannier_interp)
    He.set_energy_spectrum(elec_struct)
    He.plot_band_structure()
    He.set_H0_matr()
    He.print_info()
    # set phonon structure
    if p.dynamical_mode[1] > 0:
        # read phonons data
        # set q grid
        qgr = jdftx_qgridClass(p.work_dir, p.gamma_point)
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
        # define e-ph coupling
        
    # clean up
    He.clean_up()