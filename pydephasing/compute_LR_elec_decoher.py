import os
from pydephasing.atomic_list_struct import atoms
from parallelization.mpi import mpi
from utilities.log import log
from pydephasing.set_param_object import p
from pydephasing.phonopy_interface import atomic_displacements_from_file
from pydephasing.q_grid import qgridClass
from pydephasing.read_wfc_interface import read_wfc
from pydephasing.build_unpert_struct import build_gs_elec_struct
from pydephasing.electronic_hamiltonian import electronic_hamiltonian
from pydephasing.observables import ObservablesElectronicSystem
from pydephasing.electron_density import ElectronDensity
from pydephasing.build_interact_grad import calc_elec_hamilt_gradient
#
def compute_elec_dephas():
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
    struct_0 = build_gs_elec_struct(p.gs_data_dir)
    # atomic displacements
    atomic_displacements_from_file(p.yaml_pos_file)
    # set up density matrix object
    wfc = read_wfc(p.gs_data_dir, struct_0)
    wfc.extract_header()
    if mpi.rank == mpi.root:
        log.info("\t " + p.sep)
        log.info("\n")
        wfc.print_info()
        log.info("\n")
        log.info("\t " + p.sep)
    rho = ElectronDensity()
    rho.compute_elec_density_ofG(wfc)
    # set electronic Hamiltonian
    He = electronic_hamiltonian()
    He.set_energy_spectrum(wfc)
    # compute spin operators
    observ = ObservablesElectronicSystem(wfc)
    observ.set_Spin_operators(struct_0)
    # calc He gradient
    gradHe = calc_elec_hamilt_gradient(p.eph_matr_file)
    # set q grid
    exit()
    qgr = qgridClass()
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
    # compute e-ph coupling
    eph = electron_phonon(p.eph_matr_file)
    eph.read_eph_matrix()
    mpi.comm.Barrier()
    exit()