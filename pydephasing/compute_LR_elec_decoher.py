import os
from pydephasing.atomic_list_struct import atoms
from pydephasing.parallelization.mpi import mpi
from pydephasing.utilities.log import log
from pydephasing.set_param_object import p
from pydephasing.phonons_module import PhonopyPhonons
from pydephasing.q_grid import phonopy_qgridClass
from pydephasing.read_wfc_interface import read_wfc
from pydephasing.electronic_hamiltonian import electronic_hamiltonian
from pydephasing.q_grid import jdftx_qgridClass
from pydephasing.phonons_module import JDFTxPhonons
from pydephasing.observables import ObservablesElectronicSystem
from pydephasing.electron_density import ElectronDensity
from pydephasing.build_unpert_struct import build_vasp_gs_elec_struct, build_jdftx_gs_elec_struct
from pydephasing.build_interact_grad import calc_elec_hamilt_gradient
from pydephasing.elec_ph_inter import eph_initialization
#
def compute_VASP_elec_dephas():
    # main driver code for the calculation of dephasing time
    # in electronic systems
    #
    # first set up atoms
    # compute index maps
    atoms.set_atoms_data()
    # check whether restart dir exists
    if not os.path.isdir(p.write_dir+'/restart'):
        if mpi.rank == mpi.root:
            os.mkdir(p.write_dir+'/restart')
    mpi.comm.Barrier()
    # extract unperturbed struct.
    if mpi.rank == mpi.root:
        log.info("\t GS DATA DIR: " + p.gs_data_dir)
    elec_struct = build_vasp_gs_elec_struct(p.gs_data_dir)
    # set electronic Hamiltonian
    He = electronic_hamiltonian(Ewin_Ha=p.elec_win)
    He.set_energy_spectrum(elec_struct)
    He.plot_band_structure()
    # set up phonon object
    if mpi.rank == mpi.root:
        log.info("\t SETTING UP PHONON OBJECT")
        log.info("\t " + p.sep)
    ph = PhonopyPhonons()
    ph.set_phonopy_calc(p.yaml_pos_file, p.force_sets_file)
    # setting up Q grid
    if mpi.rank == mpi.root:
        log.info("\t SET UP Q GRID")
        log.info("\t " + p.sep)
    qgr = phonopy_qgridClass(ph_obj=ph, grid_size=p.qgr_mesh)
    qgr.set_qgrid()
    if mpi.rank == 0:
        log.info("\n")
        log.info("\t " + p.sep)
        log.info("\t Q MESH INFORMATION")
        log.info("\n")
        log.info("\n")
        log.info("\t nq: " + str(qgr.nq))
        log.info("\t grid size: " + str(qgr.grid_size))
        if qgr.nq > 10:
            for iq in range(10):
                log.info("\t wq[" + str(iq+1) + "]: " + str(qgr.wq[iq]))
            log.info("\t ...")
        else:
            for iq in range(qgr.nq):
                log.info("\t wq[" + str(iq+1) + "]: " + str(qgr.wq[iq]))
        log.info("\n")
        log.info("\t " + p.sep)
        log.info("\n")
    mpi.comm.Barrier()
    # define phonon structure on grid
    ph.plot_phonon_DOS()
    exit(0)
    # set up density matrix object
    wfc = read_wfc(p.gs_data_dir, struct_0, True)
    wfc.extract_header()
    wfc.set_PAW_projectors()
    if mpi.rank == mpi.root:
        log.info("\t " + p.sep)
        log.info("\n")
        wfc.print_info()
        log.info("\n")
        log.info("\t " + p.sep)
    rho = ElectronDensity()
    rho.compute_nelec(wfc)
    exit(0)
    # compute spin operators
    observ = ObservablesElectronicSystem(wfc)
    observ.set_Spin_operators(struct_0)
    # calc He gradient
    gradHe = calc_elec_hamilt_gradient(p.eph_matr_file)
    # set q grid
    exit()
    # compute e-ph coupling
    eph = electron_phonon(p.eph_matr_file)
    eph.read_eph_matrix()
    mpi.comm.Barrier()
    exit()

def compute_JDFTx_elec_dephas():
    '''
    use JDFTx data to perform LR calculation
    '''
    atoms.set_atoms_data(p.work_dir)
    if mpi.rank == mpi.root:
        log.info("\t " + p.sep)
        log.info("\n")
        log.info("\t COLLECTING DATA FROM JDFTx CALCULATION")
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
    # q grid
    qgr = jdftx_qgridClass(p.work_dir,p.gamma_point)
    qgr.set_qgrid(mesh_size = p.qmesh_size)
    # if we need to compute e-ph interactions
    if mpi.rank == mpi.root:
        log.info("\t " + p.sep)
        log.info("\t COLLECT PHONONS INFORMATION")
        log.info("\t " + p.sep)
        log.info("\n")
    ph = JDFTxPhonons(p.work_dir, PREFIX=p.phonon_calc_prefix, gamma_point_only=p.gamma_point)
    ph.read_ph_hamilt(qgr=qgr)
    ph.compute_phonon_DOS(qgr=qgr, n_bins=p.num_bins, sigma=p.sigma)
    if mpi.rank == mpi.root:
        log.info("\t " + p.sep)
        log.info("\t START E-PH CALCULATION")
        log.info("\t" + p.sep)
        log.info("\t E-PH APPROXIMATION: " + p.EPH_APPROX)
        log.info("\t Reading dH from " + p.eph_matr_file)
    # e-ph calculation
    gradH = calc_elec_hamilt_gradient(p.eph_matr_file)
    eph = eph_initialization(p.EPH_APPROX, p.band_range_idx, ph.nmodes, elec_struct.nbnd)
    eph.compute_gql(gradH)
    exit()