from pydephasing.set_structs import UnpertStruct
from pydephasing.set_structs import JDFTxStruct
from pydephasing.set_param_object import p

#
#   This function returns the
#   unperturbed crystal structure
#   to be used in gradients calculations
#

def build_gs_struct_base(gs_data_dir):
    # initialize unpert struct
    struct0 = UnpertStruct(gs_data_dir)
    # read POSCAR
    struct0.read_poscar()
    return struct0

def build_gs_spin_struct(gs_data_dir, HFI_CALC):
    struct0 = build_gs_struct_base(gs_data_dir)
    # read ZFS tensor
    struct0.read_zfs_tensor()
    # check here the transformation
    if HFI_CALC:
        struct0.set_hfi_Dbasis(p.fc_core)
    return struct0

def build_vasp_gs_elec_struct(gs_data_dir):
    struct0 = build_gs_struct_base(gs_data_dir)
    # extract energy eigenvalues
    struct0.extract_energy_eigv()
    struct0.extract_orbital_character()
    struct0.read_chem_pot_from_data()
    return struct0

def build_jdftx_gs_elec_struct(gs_data_dir):
    struct0 = JDFTxStruct(gs_data_dir)
    # set up electronic parameters
    struct0.set_elec_parameters()
    return struct0