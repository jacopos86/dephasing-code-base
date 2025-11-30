# This is the main subroutine
# for non Markovian calculation of decoherence
# times
# -> evolve delta\rho(1)q
# -> if 2nd order correction add lindblad for two-phonons term
# if evolve coherently the additional terms
import os
import numpy as np
from pydephasing.parallelization.mpi import mpi
from pydephasing.utilities.log import log
from pydephasing.set_param_object import p
from pydephasing.atomic_list_struct import atoms
from pydephasing.build_interact_grad import calc_interaction_grad
from pydephasing.build_unpert_struct import build_gs_spin_struct
from pydephasing.spin_dens_matr import spin_dmatr
from pydephasing.q_grid import qgridClass
from pydephasing.nuclear_spin_config import nuclear_spins_config
from pydephasing.spin_model.spin_hamiltonian import set_spin_hamiltonian
from pydephasing.real_time.set_real_time_solver import set_real_time_solver
from pydephasing.magnetic_field import magnetic_field
from pydephasing.phonons_module import PhononsClass
from pydephasing.spin_model.spin_ph_handler import spin_ph_handler
#
def compute_RT_spin_dephas(ZFS_CALC, HFI_CALC, config_index=0):
    # main driver of the calculation dephasing time
    # spin model
    #
    # first set up atoms
    # compute index maps
    atoms.set_atoms_data()
    # check restart exist otherwise create
    if not os.path.isdir(p.write_dir+'/restart'):
        if mpi.rank == mpi.root:
            os.mkdir(p.write_dir+'/restart')
    mpi.comm.Barrier()
    # extract interaction gradients
    if np.sum(p.dynamical_mode) > 0:
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
    # set up spin phonon interaction
    if np.sum(p.dynamical_mode) > 0:
        sp_ph_inter = spin_ph_handler(p.order_2_correct, ZFS_CALC, HFI_CALC, p.hessian)
        if mpi.rank == mpi.root:
            log.debug("\t ZFS_CALC: " + str(sp_ph_inter.ZFS_CALC))
            log.debug("\t HFI_CALC: " + str(sp_ph_inter.HFI_CALC))
    # spin density matrix
    rho = spin_dmatr()
    rho.initialize(p.psi0)
    # set q grid
    if np.sum(p.dynamical_mode) > 0:
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
        # extract phonon data
        ph = PhononsClass()
        ph.set_ph_data(qgr)
        # compute spin phonon matrix
        if mpi.rank == mpi.root:
            log.info("\n")
            log.info("\t " + p.sep)
            log.info("\t START SPIN-PHONON COUPLING CALCULATION")
        sp_ph_inter.compute_spin_ph_coupl(nat, Hsp, ph, qgr, interact_dict, nuclear_config)
        if mpi.rank == mpi.root:
            log.info("\n")
            log.info("\t END SPIN-PHONON COUPLING CALCULATION")
            log.info("\t " + p.sep)
        print('g_ql', np.max(sp_ph_inter.g_ql.real))
    # set real time solver (time in ps)
    Bfield = magnetic_field(p.Bt)
    solver = set_real_time_solver(HFI_CALC)
    if np.sum(p.dynamical_mode) == 0:
        solver.evolve(p.dt, p.T, rho, Hsp, Bfield)
    else:
        solver.evolve(p.dt, p.T, rho, Hsp, Bfield, ph, sp_ph_inter, qgr)