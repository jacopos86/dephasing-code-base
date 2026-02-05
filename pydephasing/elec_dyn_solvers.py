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
from pydephasing.q_grid import jdftx_qgridClass, QGrid_1D
from pydephasing.k_grid import KGrid_1D
from pydephasing.atomic_list_struct import atoms
from pydephasing.electronic_hamiltonian import electronic_hamiltonian, model_electronic_hamiltonian
from pydephasing.elec_dens_matr import elec_dmatr
from pydephasing.elec_ph_inter import DeformationPotentialElectronPhonon
from pydephasing.observables import ObservablesElectronicModel
from pydephasing.p_mtxel import TwoBandsLinearMomentum
from pydephasing.elec_light_inter import ElectronLightCouplTwoBandsModel
from pydephasing.Ehrenfest_field import ehr_field
from pydephasing.real_time.set_real_time_solver import set_real_time_electronic_solver
from pydephasing.common.phys_constants import THz_to_ev
from pydephasing.utilities.plot_functions import plot_td_Bq, plot_td_trace, plot_td_occup, plot_total_energy
from pydephasing.real_time.external_ph_fields import set_up_phonon_drive
from pydephasing.real_time.external_elec_fields import set_up_vector_potential

#
def solve_elec_model_dyn():
    if mpi.rank == mpi.root:
        log.info("\t " + p.sep)
        log.info("\n")
        log.info("\t PERFORM REAL TIME DYNAMICS")
        log.info("\t MODEL SYSTEM WITH DEFORMATION POTENTIAL")
        log.info("\n")
        log.info("\t " + p.sep)
    # first set the k grid
    kgr = KGrid_1D(p.nkpt)
    kgr.set_kgrid(p.L)
    kgr.set_wk()
    if mpi.rank == mpi.root:
        log.info("\t " + p.sep)
        log.info(f"\t k grid shape: {kgr.kpts.shape}")
        kgr.show_kgr()
        log.info("\t k grid weights: ")
        kgr.show_kw()
        log.info("\t " + p.sep)
    # ============================================================
    # 1. Build MODEL electronic Hamiltonian
    # ============================================================
    He = model_electronic_hamiltonian(
        elec_bands=p.elec_bands,                     # 1 or 2
        kgr=kgr,
        eff_mass=p.elec_eff_mass,                    # list
        band_offset=p.elec_band_offset,              # list
        Ewin_Ha=p.elec_win,
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
    # ============================================================
    # 2. Build MODEL electronic linear momentum
    # ============================================================
    elec_p = TwoBandsLinearMomentum(p.elec_lm_params)
    # ============================================================
    # 2A. Electric dipole interaction
    # ============================================================
    ext_Apot = set_up_vector_potential(p.ext_Apot_params)
    print(ext_Apot)
    exit()
    # ============================================================
    # 2B. Electric dipole interaction
    # ============================================================
    elc = ElectronLightCouplTwoBandsModel(
        pe_k=elec_p.set_p_matrix(kgr),
        ext_Apot = ext_Apot
    )
    exit()
    # phonons section
    if p.dynamical_mode[1] > 0:
        # ============================================================
        # 3. Q grid && Model phonons (acoustic branch)
        # ============================================================
        qgr = QGrid_1D(kgr)
        if mpi.rank == mpi.root:
            log.info("\t " + p.sep)
            log.info(f"\t q grid size: {qgr.qpts.shape}")
            qgr.show_qgr()
            log.info("\t q grid weights: ")
            qgr.show_qw()
            log.info("\t " + p.sep)
        ph = PhononsStructModel(p.sound_velocity)
        ph.compute_phonon_DOS(qgr, p.ph_DOS_plot)
        ph.compute_energy_dispersion(qgr)
        ph.set_ph_energies(qgr)
        ph.summary()
        # ============================================================
        # 3A. Deformation potential electron–phonon coupling
        # ============================================================
        eph = DeformationPotentialElectronPhonon(
            eph_params=p.eph_params,
            nBands=p.elec_bands,
            sys_size=p.L
        )
        eph.summary()
        if mpi.rank == mpi.root:
            log.info("\t Deformation potential coupling computed")
        gql = eph.compute_gql(qgr, kgr, ph)
        exit()
        # ============================================================
        # 3B. Ehrenfest / phonon DM initialization
        # ============================================================
        ehr = ehr_field()
        ehr.initialize_Bfield(p.nkpt, ph.nmodes)
        ehr.set_map_qtomq(qgr)
        # ============================================================
        # 3C. PH drive object
        # ============================================================
        phdr = set_up_phonon_drive(p.ph_drive, ph, qgr)
    else:
        gql = None
        ehr = None
        omega_q = None
    # ============================================================
    # 4. Spin operators + SO interaction
    # ============================================================
    Observ = ObservablesElectronicModel(basis_set=None)
    Observ.set_Spin_operators(p.elec_bands, p.nkpt)
    # ============================================================
    # 5. Ready for non-Markovian propagation
    # ============================================================
    RT_solver = set_real_time_electronic_solver()
    RT_solver.summary()
    out = RT_solver.propagate(
        He=He,
        rho_e=rho_e,
        ehr_field=ehr,
        gql=gql,
        omega_q = omega_q,
        ph_drive = phdr
    )
    # ============================================================
    # 6. print output + plot observables
    # ============================================================
    rho_e = out[0]
    ehr = out[1]
    print(rho_e.traces[1,0,:], rho_e.traces[0,0,:])
    print(ehr.sum_Bqt[:])
    # compute energy
    Eph_t, Ee_t, Eeph_t = Observ.compute_system_energies(p.evol_params, rho_e, ehr, He, omega_q, gql)
    if mpi.rank == mpi.root:
        plot_td_Bq(p.evol_params, ehr.Bq_t)
        plot_td_trace(p.evol_params, rho_e.traces)
        plot_td_occup(p.evol_params, rho_e.rho_t)
        plot_total_energy(p.evol_params, Eph_t, Ee_t, Eeph_t)
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
