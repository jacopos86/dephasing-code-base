from pathlib import Path
import yaml
import numpy as np
import warnings
from phonopy.interface.calculator import read_crystal_structure
import phonopy
from pydephasing.set_param_object import p
from pydephasing.utilities.log import log
from pydephasing.parallelization.mpi import mpi
from pydephasing.common.print_objects import print_2D_matrix
#
def extract_unit_cell():
    fil = Path(p.gs_data_dir+"/POSCAR")
    if fil.is_file():
        unitcell, _ = read_crystal_structure(fil, interface_mode="vasp")
        return unitcell

def extract_super_cell(pos_yaml_fil):
    supercell_matrix = None
    if Path(pos_yaml_fil).is_file():
        with open(pos_yaml_fil) as f:
            data = yaml.safe_load(f)
        if "supercell_matrix" in data.keys():
            supercell_matrix = data['supercell_matrix']
        else:
            log.error("Supercell matrix not found in " + pos_yaml_fil)
    else:
        log.error("file not found: " + pos_yaml_fil)
    if mpi.rank == mpi.root:
        log.info("\n")
        log.info("\t " + p.sep)
        log.info("\t SUPERCELL MATRIX:\n")
        print_2D_matrix(np.array(supercell_matrix))
        log.info("\t " + p.sep)
        log.info("\n")
    return supercell_matrix

# ---------------------------------------------------------
# 1. Initialize Phonopy object with FORCE_SETS
# ---------------------------------------------------------
def setup_phonopy_from_forcesets(YAML_POS_FIL, FORCE_SETS_FIL):
    """
    Load POSCAR + FORCE_SETS + phonopy.yaml (optional) and return
    a fully initialized Phonopy object with force constants.
    """
    # ---------------------------------------------------------
    # 1. Load unitcell
    # ---------------------------------------------------------
    unitcell = extract_unit_cell()

    # ---------------------------------------------------------
    # 2. Load supercell matrix
    # ---------------------------------------------------------
    supercell_matrix = extract_super_cell(YAML_POS_FIL)

    # ---------------------------------------------------------
    # 4. check FORCE_SETS exists
    # ---------------------------------------------------------
    forcesets_path = Path(FORCE_SETS_FIL)
    if not forcesets_path.is_file():
        log.error(f"FORCE_SETS file not found")

    # ---------------------------------------------------------
    # 3. Initialize Phonopy object
    # ---------------------------------------------------------
    phonon = phonopy.load(YAML_POS_FIL, force_sets_filename=FORCE_SETS_FIL)

    if mpi.rank == mpi.root:
        log.info("\t FORCE_SETS loaded")
        log.info(f"\t Number of displacements: {len(phonon.displacements)}")

    # ---------------------------------------------------------
    # 5. print force constants
    # ---------------------------------------------------------

    if mpi.rank == mpi.root:
        log.info(f"\t Force constants shape: {phonon.force_constants.shape}")
        log.info("\t Phonopy object is fully initialized.\n")

    # ---------------------------------------------------------
    # 6. Broadcast phonon object to all ranks
    #    (Phonopy object is Python serializable)
    # ---------------------------------------------------------
    phonon = mpi.comm.bcast(phonon, root=mpi.root)

    return phonon