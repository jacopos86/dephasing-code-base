# This is the driving subroutine for
# the calculation of the static dephasing
# it computes the energy auto-correlation
# and it returns it for further processing
import numpy as np
from pydephasing.input_parameters import p
from pydephasing.spin_hamiltonian import spin_hamiltonian
from pydephasing.nuclear_spin_config import nuclear_spins_config
from pydephasing.energy_fluct_mod import spin_level_static_fluctuations
from pydephasing.mpi import mpi
from pydephasing.log import log
from pydephasing.hf_stat_struct import perturbation_HFI_stat
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
        log.info("n. config: " + str(p.nconf))
        log.info("n. spins: " + str(p.nsp))
    mpi.comm.Barrier()
    # unpert. structure (HFI+ZFS)
    if mpi.rank == mpi.root:
        log.info("HF fc core: " + str(p.fc_core))
    HFI0 = perturbation_HFI_stat(p.work_dir, p.grad_info, p.fc_core)
    # set HFI and ZFS
    HFI0.set_gs_struct()
    # nat
    nat = HFI0.struct_0.nat
    if mpi.rank == mpi.root:
        log.info("n. atoms= " + str(nat))
    # set spin hamiltonian
    Hss = spin_hamiltonian()
    # set time spinor evol.
    Hss.set_time(p.dt, p.T)
    # set up the spin vector evolution
    Hss.compute_spin_vector_evol(HFI0.struct_0, p.psi0, p.B0)
    # write data on file
    if mpi.rank == mpi.root:
        Hss.write_spin_vector_on_file(p.write_dir)
    mpi.comm.Barrier()
    # quantum spin states
    p.qs1 = np.array([1.+0j,0j,0j])
    p.qs2 = np.array([0j,1.+0j,0j])
    # normalization
    p.qs1[:] = p.qs1[:] / np.sqrt(np.dot(p.qs1.conjugate(), p.qs1))
    p.qs2[:] = p.qs2[:] / np.sqrt(np.dot(p.qs2.conjugate(), p.qs2))
    #
    # set up nuclear spins
    #
    for ic in calc_loc_list:
        log.info("compute It - conf: " + str(ic+1))
        # set up spin config.
        config = nuclear_spins_config(p.nsp, p.B0)
        # set up time in (mu sec)
        config.set_time(p.dt_mus, p.T_mus)
        # set initial spins
        config.set_nuclear_spins(nat, ic)
        # compute dynamical evolution
        config.set_nuclear_spin_evol(Hss, HFI0.struct_0)
        # set temporal fluctuations
        config.set_nuclear_spin_time_fluct()
        # write data on file
        config.write_It_on_file(p.write_dir, ic)
        # compute eff. forces 
        # for each spin
        HFI0.compute_stat_force_HFS(Hss, config)
        # initialize energy fluct.
        E_fluct = spin_level_static_fluctuations(p.nt2)
        # set spin fluct. energy
        # extract delta I(t)
        E_fluct.compute_deltaE_oft(config)
        # compute average fluct.
        deltaE_aver_oft[:] += E_fluct.deltaE_oft[:]
        # eV units
        # init. ACF
        acf = autocorrel_func_hfi_stat(E_fluct)
        # compute ACF
        D2, Ct = acf.compute_acf()
        # extract T2 data
        acf.extract_dephas_data(D2, Ct, T2_obj, Delt_obj, tauc_obj, ic)
        log.info("end It calculation - conf: " + str(ic+1))
    # wait processes
    mpi.comm.Barrier()
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