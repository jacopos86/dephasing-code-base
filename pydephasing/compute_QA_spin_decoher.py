import os
from pydephasing.atomic_list_struct import atoms
from pydephasing.set_param_object import p
from pydephasing.mpi import mpi
from pydephasing.log import log
from pydephasing.build_interact_grad import calc_interaction_grad
from pydephasing.build_unpert_struct import build_gs_spin_struct
from pydephasing.spin_hamiltonian import set_spin_hamiltonian
from pydephasing.nuclear_spin_config import nuclear_spins_config
#
def compute_dephas_QA(ZFS_CALC, HFI_CALC, config_index=0):
    #
    # main driver code for the calculation
    # of spin dephasing using a quantum algorithm
    # based on Suzuki-Trotter evolution
    #
    # first set atoms + index maps
    atoms.set_atoms_data()
    # check restart exists otherwise create one
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
    Hsp.qubitize_hamiltonian()