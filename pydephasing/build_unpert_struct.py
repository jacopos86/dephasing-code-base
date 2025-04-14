from pydephasing.set_structs import UnpertStruct

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