# This is the main subroutine used
# for the calculation of the HFI component
# of the dephasing time
# it computes the relative autocorrelation function
# and returns it for further processing
from pydephasing.input_parameters import p
from pydephasing.set_structs import DisplacedStructs, DisplacedStructures2ndOrder
from pydephasing.spin_hamiltonian import spin_hamiltonian
from pydephasing.atomic_list_struct import atoms
from pydephasing.gradient_interactions import gradient_HFI, gradient_2nd_HFI
from pydephasing.nuclear_spin_config import nuclear_spins_config
from pydephasing.spin_ph_inter import SpinPhononClass
from pydephasing.auto_correl_inhom_driver import acf_sp_ph_inhom
from pydephasing.extract_ph_data import extract_ph_data
from pydephasing.log import log
from pydephasing.mpi import mpi
import logging
#
def compute_hfi_dephas():
    # driving code for the calculation of
    # inhomogeneous dephasing
    #
    # create perturbed atomic structures
    struct_list = []
    for i in range(len(p.displ_poscar_dir)):
        displ_struct = DisplacedStructs(p.displ_poscar_dir[i], p.displ_outcar_dir[i])
        # set atomic displacements
        displ_struct.atom_displ(p.atoms_displ[i])    # ang
        # append to list
        struct_list.append(displ_struct)
    # 2nd order displ structs
    if p.order_2_correct:
        struct_list_2nd = []
        for i in range(len(p.displ_2nd_poscar_dir)):
            displ_struct = DisplacedStructures2ndOrder(p.displ_2nd_poscar_dir[i], p.displ_2nd_outcar_dir[i])
            # set atomic displ. in the structure
            displ_struct.atom_displ(p.atoms_2nd_displ[i]) # ang
            # append to list
            struct_list_2nd.append(displ_struct)
    # set HFI gradient
    if mpi.rank == mpi.root:
        log.info("\n")
        log.info("\t " + p.sep)
        log.info("\t HF fc core: " + str(p.fc_core))
    gradHFI = gradient_HFI(p.work_dir, p.grad_info, p.fc_core)
    # set hyperfine interaction
    gradHFI.set_gs_hfi_tensor()
    # set the gradient
    gradHFI.set_tensor_gradient(struct_list)
    # set gradient in quant. vector basis
    gradHFI.set_U_gradAhfi_U_tensor()
    # n. atoms
    nat = gradHFI.struct_0.nat
    # set index maps
    atoms.compute_index_to_ia_map(nat)
    atoms.compute_index_to_idx_map(nat)
    # extract atom positions
    atoms.extract_atoms_coords(nat)
    # HFI 2nd order
    if p.order_2_correct:
        # set 2nd order tensor
        grad2HFI = gradient_2nd_HFI(p.work_dir, p.grad_info, p.fc_core)
        grad2HFI.set_gs_hfi_tensor()
        # compute 2nd order grad. HFI
        grad2HFI.extract_2nd_order_gradients()
    #
    # set up the spin
    # configurations
    #
    if mpi.rank == mpi.root:
        log.info("\t n. config: " + str(p.nconf))
        log.info("\t n. spins: " + str(p.nsp))
        log.info("\t " + p.sep)
        log.info("\n")
    #
    # set up the spin Hamiltonian
    Hsp = spin_hamiltonian()
    Hsp.set_zfs_levels(gradHFI.struct_0, p.B0)
    # set up spin phonon interaction class
    sp_ph_inter = SpinPhononClass().generate_instance()
    sp_ph_inter.set_quantum_states(Hsp)
    #
    # extract phonon data
    #
    u, wu, nq, qpts, wq, mesh = extract_ph_data()
    if mpi.rank == 0:
        log.info("\n")
        log.info("\t " + p.sep)
        log.info("\t Q MESH INFORMATION")
        log.info("\n")
        log.info("\n")
        log.info("\t nq: " + str(nq))
        log.info("\t mesh: " + str(mesh))
        if nq > 10:
            for iq in range(10):
                log.info("\t wq[" + str(iq+1) + "]: " + str(wq[iq]))
            log.info("\t ...")
        else:
            for iq in range(nq):
                log.info("\t wq[" + str(iq+1) + "]: " + str(wq[iq]))
        log.info("\n")
        log.info("\t " + p.sep)
        log.info("\n")
    assert len(qpts) == nq
    assert len(u) == nq
    mpi.comm.Barrier()
    # if w_resolved define freq. grid
    if p.w_resolved:
        p.set_w_grid(wu)
    #
    # set q pts. grid
    if p.ph_resolved:
        p.set_wql_grid(wu, nq, nat)
    #
    # initialize ACF over (q,l) list
    acf = acf_sp_ph_inhom()
    acf.allocate_acf_arrays(nat)
    #
    # restart calculation if file found
    restart_file = p.write_dir + "/restart_calculation.yml"
    ic0, T2_calc_handler = acf.restart_calculation(nat, restart_file)
    # ---------------------------------------------------------
    #  
    #   START ITERATING OVER DIFFERENT CONFIG.
    #
    # ---------------------------------------------------------
    for ic in range(ic0, p.nconf):
        # set spin config.
        config = nuclear_spins_config(p.nsp, p.B0)
        config.set_nuclear_spins(nat, ic)
        # -------------------------------------
        #    first order HFI forces
        # -------------------------------------
        sp_ph_inter.set_Fax_hfi(gradHFI, Hsp, config)
        Fax = sp_ph_inter.Fhf_ax
        # eV / ang
        # -------------------------------------
        #    second order HFI forces
        # -------------------------------------
        if p.order_2_correct:
            sp_ph_inter.set_Faxby_hfi(grad2HFI, Hsp, struct_list_2nd, config)
            # eV / ang^2
            Faxby = sp_ph_inter.Fhf_axby
        else:
            Faxby = None
        # ---------------------------------------------------
        #
        #   COMPUTE ACF
        #
        # ---------------------------------------------------
        acf.compute_acf(wq, wu, u, qpts, nat, Fax, Faxby, Hsp)
        #
        # collect data from processes
        acf.collect_acf_from_processes(nat)
        # test acf -> check t=0 / w=0
        if log.level <= logging.INFO:
            acf.auto_correl_test()
        # update avg ACF
        acf.update_avg_acf()
        #
        # extract T2_inv
        T2_calc_handler.extract_physical_quantities(acf, ic, nat)
        if mpi.rank == mpi.root:
            log.info("\n\n")
            log.info("\t " + p.sep)
            log.warning("\t ic: " + str(ic+1) + " -> COMPLETED")
            log.info("\t " + p.sep)
            log.info("\n\n")
        # save temp. data
        if mpi.rank == mpi.root:
            acf.save_data(ic, T2_calc_handler)
        # re-set arrays
        acf.allocate_acf_arrays(nat)
        # wait
        mpi.comm.Barrier()
    #
    # complete calculation -> final average acf
    acf.compute_avg_acf()
    #
    # extract T2_inv
    T2_calc_handler.extract_avg_physical_quantities(acf, nat)
    if mpi.rank == mpi.root:
        log.info("\n\n")
        log.info("\t " + p.sep)
        log.warning("\t average calculation -> COMPLETED")
        log.info("\t " + p.sep)
        log.info("\n\n")
    # wait
    mpi.comm.Barrier()
    return T2_calc_handler