# This is the main subroutine for
# the calculation of the ZFS dephasing
# it computes the energy autocorrelation function
# and return it for further processing
import logging
import numpy as np
import os
from pydephasing.mpi import mpi
from pydephasing.log import log
from pydephasing.set_param_object import p
from pydephasing.atomic_list_struct import atoms
from pydephasing.spin_hamiltonian import set_spin_hamiltonian
from pydephasing.spin_ph_handler import spin_ph_handler
from pydephasing.ph_amplitude_module import PhononAmplitude
from pydephasing.auto_correlation_spph_mod import acf_sp_ph
from pydephasing.T2_calc_handler import set_T2_calc_handler
from pydephasing.energy_fluct_mod import ZFS_ph_fluctuations
from pydephasing.phonons_module import PhononsClass
from pydephasing.q_grid import qgridClass
from pydephasing.build_interact_grad import calc_interaction_grad
from pydephasing.build_unpert_struct import build_gs_spin_struct
from pydephasing.nuclear_spin_config import nuclear_spins_config
from pydephasing.fermi_golden_rule import GeneralizedFermiGoldenRule
#
def compute_spin_dephas(ZFS_CALC, HFI_CALC, config_index=0):
    # main driver code for the calculation of dephasing time
    # in homogeneous spin systems
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
    # set up spin phonon interaction class
    sp_ph_inter = spin_ph_handler(p.order_2_correct, ZFS_CALC, HFI_CALC, p.hessian)
    if mpi.rank == mpi.root:
        log.debug("\t ZFS_CALC: " + str(sp_ph_inter.ZFS_CALC))
        log.debug("\t HFI_CALC: " + str(sp_ph_inter.HFI_CALC))
    # set Fermi Golden Rule object
    if mpi.rank == mpi.root:
        log.info("\t " + p.sep)
        log.info("\t TIME RESOLVED: " + str(p.time_resolved))
        log.info("\t FREQ. RESOLVED: " + str(p.w_resolved))
        log.info("\t " + p.sep)
    FGR = GeneralizedFermiGoldenRule().generate_instance(p.time_resolved, p.w_resolved)
    FGR.set_grids()
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
    #
    # extract phonon data
    #
    ph = PhononsClass()
    ph.set_ph_data(qgr)
    #
    # set spin-phonon matrix
    # in eV/ang units
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
    FGR.compute_relax_time_one_ph(Hsp, sp_ph_inter, ph, qgr, p.temperatures)
    exit()
    #
    # compute ZFS fluctuations
    if mpi.rank == mpi.root:
        log.info("\n")
        log.info("\t " + p.sep)
        log.info("\t START ENERGY FLUCTUATIONS CALC.")
    ZFS_fluct = ZFS_ph_fluctuations()
    ZFS_fluct.compute_fluctuations(wq, qpts, nat, wu, u)
    ZFS_fluct.collect_acf_from_processes()
    if mpi.rank == mpi.root:
        log.info("\n")
        log.info("\t ENERGY FLUCTUATIONS CALC. CONCLUDED")
        log.info("\t " + p.sep)
    #
    # if w_resolved define freq. grid
    if p.w_resolved:
        p.set_w_grid(wu)
    Fax = sp_ph_inter.Fzfs_ax
    print(max(Fax))
    import sys
    sys.exit()
    # 2nd order calculation
    if p.order_2_correct:
        # F_axby = <1|S Grad_ax,by D S|1> - <0|S Grad_ax,by D S|0>
        # F should be in eV/ang^2 units
        # TODO : uncomment here 
        #'''
        sp_ph_inter.set_Faxby_zfs(grad2ZFS, Hsp)
        Faxby = sp_ph_inter.Fzfs_axby
        print(np.max(Faxby))
        #'''
        Faxby = np.zeros((3*nat, 3*nat), dtype=np.complex128)
        # eV / ang^2
    else:
        Faxby = None
    # set q pts. grid
    if p.ph_resolved:
        p.set_wql_grid(wu, nq, nat)
    #
    # compute acf over local (q,l) list
    acf = acf_sp_ph().generate_instance()
    acf.allocate_acf_arrays(nat)
    acf.compute_acf(wq, wu, u, qpts, nat, Fax, Faxby, Hsp)
    #
    # collect data from processes
    acf.collect_acf_from_processes(nat)
    # test acf -> check t=0 / w=0
    if log.level <= logging.INFO:
        acf.auto_correl_test()
    #
    # print average atom displ
    if log.level <= logging.DEBUG:
        #
        ph_ampl = PhononAmplitude(nat)
        # run over q pts
        #
        iq_list = mpi.split_list(range(nq))
        for iq in iq_list:
            # w(q) data (THz)
            wuq = wu[iq]
            # eu(q) data
            euq = u[iq]
            # q vector
            qv = qpts[iq]
            # compute atoms displ. first
            ph_ampl.compute_ph_amplq(euq, wuq, nat)
            ph_ampl.update_atom_displ(wq[iq], qv, wuq, nat)
            #
        if mpi.size > 1:
            ph_ampl.collect_displ_between_proc(nat)
        # save data on file
        ph_ampl.print_atom_displ(p.write_dir)
    mpi.comm.Barrier()
    # set up T2 calculation
    T2_calc_handler = set_T2_calc_handler()
    # prepare data arrays
    T2_calc_handler.set_up_param_objects_from_scratch(nat)
    #
    # compute T2_inv + other phys. parameters
    T2_calc_handler.extract_physical_quantities(acf, nat)
    # wait all processes
    mpi.comm.Barrier()
    return T2_calc_handler