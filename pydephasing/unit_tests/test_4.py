from pydephasing.wfc_overlap_interface import read_VASP_files, VASP_wfc_overlap_, read_file  
import os


def check_file_exists(path_to_calc, filename):
    file_path = os.path.join(path_to_calc, filename)
    assert os.path.isfile(file_path), f"{file_path} does not exists"

def load_perturbations_files(work_dir, header, disp_yaml):
    
    overlap_instance = VASP_wfc_overlap_(work_dir, header)
    result_displacement = overlap_instance.read_displacements(disp_yaml)
    N_mode, Rp_wswq_dirs, Rm_wswq_dirs = overlap_instance.read_all_wswq(result_displacement)
    
    return N_mode, Rp_wswq_dirs, Rm_wswq_dirs

def check_list_files(wswq_dirs, filename):
    for d in wswq_dirs:
        check_file_exists(d, filename) # Check if WSWQ file is inside d
        
def test_files():
    calc_dir = "/work/hdd/beee/szhan213/SiV-_el-ph_from_NERSC/VASP_soc_phononpy_accurate/"  # Path to all calculation workflow
    perturbed_structures_dirs = "1_Run_forces" # Sub dir with perturbed structures
    wswq_dirs = "3_Run_elph" # Sub dir where wavefunction overlap files are stored
    e_ph_dir = "4_Get_el-ph" # Sub dir where final e-ph calculation is computed
    header = "disp-"  # Header for
    disp_yaml = "4_Get_el-ph/phonopy_disp.yaml"

    N_mode, Rp_wswq_dirs, Rm_wswq_dirs = load_perturbations_files(os.path.join(calc_dir, perturbed_structures_dirs), header, os.path.join(calc_dir, disp_yaml))
    check_list_files(Rp_wswq_dirs, "POSCAR")
    check_list_files(Rm_wswq_dirs, "POSCAR")

    N_mode, Rp_wswq_dirs, Rm_wswq_dirs = load_perturbations_files(os.path.join(calc_dir, wswq_dirs), header, os.path.join(calc_dir, disp_yaml))
    check_list_files(Rp_wswq_dirs, "WSWQ")
    check_list_files(Rm_wswq_dirs, "WSWQ")
