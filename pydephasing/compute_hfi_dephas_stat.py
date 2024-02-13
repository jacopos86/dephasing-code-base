# This is the driving subroutine for
# the calculation of the static dephasing
# it computes the energy auto-correlation
# and it returns it for further processing
import numpy as np
import logging
from pydephasing.input_parameters import p
from pydephasing.spin_hamiltonian import spin_hamiltonian
from pydephasing.nuclear_spin_config import nuclear_spins_config
from pydephasing.energy_fluct_mod import spin_level_static_fluctuations
from pydephasing.mpi import mpi
from pydephasing.log import log
from pydephasing.hf_stat_struct import perturbation_HFI_stat
from pydephasing.T2_calc_handler import set_T2_calc_handler
#
def compute_hfi_stat_dephas():
    # input_params -> input parameters object
    # at_resolved : true -> compute atom res. auto-correlation
    # ph_resolved : true -> compute ph. resolved auto-correlation
    #
    # first : set calculation list
    calc_list = range(p.nconf)
    # parallelization section
    calc_loc_list = mpi.split_list(calc_list)
    # print calc. data
    if mpi.rank == 0:
        log.info("\t " + p.sep)
        log.info("\n")
        log.info("\t n. config: " + str(p.nconf))
        log.info("\t n. spins: " + str(p.nsp))
    mpi.comm.Barrier()
    # unpert. structure (HFI+ZFS)
    if mpi.rank == mpi.root:
        log.info("\t HF fc core: " + str(p.fc_core))
    HFI0 = perturbation_HFI_stat(p.work_dir, p.grad_info, p.fc_core)
    # set HFI and ZFS
    HFI0.set_gs_struct()
    # nat
    nat = HFI0.struct_0.nat
    if mpi.rank == mpi.root:
        log.info("\t n. atoms= " + str(nat))
    # align the applied B field
    # to the spin quantization axis
    Bfield = np.array([0., 0., p.B0])
    if mpi.rank == mpi.root:
        log.info("\t applied B field : " + str(Bfield))
        log.info("\n")
        log.info("\t " + p.sep)
    # set spin hamiltonian
    Hss = spin_hamiltonian()
    # set time spinor evol.
    Hss.set_time(p.dt, p.T)
    # set up the spin vector evolution
    Hss.compute_spin_vector_evol(HFI0.struct_0, p.psi0, Bfield)
    # write data on file
    if mpi.rank == mpi.root:
        Hss.write_spin_vector_on_file(p.write_dir)
    mpi.comm.Barrier()
    # set up the calculation
    #     handler
    T2_calc_handler = set_T2_calc_handler()
    T2_calc_handler.set_up_param_objects_from_scratch(p.nconf)
    #
    # set up nuclear spins
    #
    for ic in calc_loc_list:
        if mpi.rank == mpi.root:
            log.info("\n")
            log.info("\t " + p.sep)
        log.info("\t compute It - conf: " + str(ic+1))
        # set up spin config.
        config = nuclear_spins_config(p.nsp, Bfield)
        # set up time in (mu sec)
        config.set_time(p.dt_mus, p.T_mus)
        # set initial spins
        config.set_nuclear_spins(nat, ic)
        # compute dynamical evolution
        config.set_nuclear_spin_evol(Hss, HFI0.struct_0)
        # compute dephas. matrix
        # D^~(u) -> T2i - lw_eV
        T2_calc_handler.parameter_eval_driver(ic, config, Hss, HFI0.struct_0)
        # write data on file
        if log.level <= logging.INFO:
            config.write_It_on_file(p.write_dir, ic)
        log.info("end It calculation - conf: " + str(ic+1))
    if mpi.rank == mpi.root:
        log.info("\t " + p.sep)
        log.info("\n")
    # wait processes
    mpi.comm.Barrier()
    import sys
    sys.exit()
    #
    # gather arrays on a single
    # processor
    #
    deltaE_aver_oft = mpi.collect_array(deltaE_aver_oft) / p.nconf
    E_fluct_aver = spin_level_static_fluctuations(p.nt2)
    E_fluct_aver.deltaE_oft = deltaE_aver_oft
    #
    if mpi.rank == mpi.root:
        # init. ACF
        acf = autocorrel_func_hfi_stat(E_fluct_aver)
        # compute ACF
        D2, Ct = acf.compute_acf()
        # extract T2 data
        acf.extract_dephas_data(D2, Ct, T2_obj, Delt_obj, tauc_obj, p.nconf)
    mpi.comm.Barrier()
    #
    # gather data into lists
    T2_obj.collect_from_other_proc()
    Delt_obj.collect_from_other_proc()
    tauc_obj.collect_from_other_proc()
    #
    return T2_obj, Delt_obj, tauc_obj