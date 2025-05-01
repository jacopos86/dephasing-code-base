# This is the main subroutine
# for non Markovian calculation of decoherence
# times
# -> evolve delta\rho(1)q
# -> if 2nd order correction add lindblad for two-phonons term
# if evolve coherently the additional terms
import os
from pydephasing.mpi import mpi
from pydephasing.log import log
from pydephasing.set_param_object import p
from pydephasing.atomic_list_struct import atoms
from pydephasing.build_interact_grad import calc_interaction_grad
from pydephasing.build_unpert_struct import build_gs_spin_struct
from pydephasing.electr_ph_dens_matr import elec_ph_dmatr
from pydephasing.spin_dens_matr import spin_dmatr
from pydephasing.q_grid import qgridClass
from pydephasing.nuclear_spin_config import nuclear_spins_config
from pydephasing.spin_hamiltonian import set_spin_hamiltonian
from pydephasing.set_real_time_solver import set_real_time_solver
#
def compute_nmark_dephas(ZFS_CALC, HFI_CALC, config_index=0):
    # main driver of the calculation non markovian
    # dephasing time
    #
    # first set up atoms
    # compute index maps
    atoms.set_atoms_data()
    # check restart exist otherwise create
    if not os.path.isdir(p.work_dir+'/restart'):
        if mpi.rank == mpi.root:
            os.mkdir(p.work_dir+'/restart')
    mpi.comm.Barrier()
    # extract interaction gradients
    interact_dict = calc_interaction_grad(ZFS_CALC, HFI_CALC)
    mpi.comm.Barrier()
    # n. atoms
    nat = atoms.nat
    # extract unperturbed struct.
    if mpi.rank == mpi.root:
        log.info("\t GS DATA DIR: " + p.gs_data_dir)
    struct_0 = build_gs_spin_struct(p.gs_data_dir, HFI_CALC)
    # set nuclear spin configuration
    nuclear_config = None
    if HFI_CALC:
        if mpi.rank == mpi.root:
            log.info("\n")
            log.info("\t " + p.sep)
            log.info("\t number nuclear spins: " + str(p.nsp))
            log.info("\t nuclear config. index: " + str(config_index))
            log.info("\t " + p.sep)
            log.info("\n")
        # set spin config.
        nuclear_config = nuclear_spins_config(p.nsp, p.B0)
        nuclear_config.set_nuclear_spins(nat, config_index)
    # set up the spin Hamiltonian
    Hsp = set_spin_hamiltonian(struct_0, p.B0, nuclear_config)
    # spin density matrix
    rho = spin_dmatr()
    rho.initialize(p.psi0)
    # set q grid
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
    # set real time solver
    solver = set_real_time_solver()
    solver.evolve(dt, T, rho, Hsp)
    exit()
    # spin phonon
    # density matrix
    rho_q = elec_ph_dmatr().generate_instance()
    rho_q.set_modes_list(qgr)