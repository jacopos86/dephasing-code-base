from pathlib import Path
import yaml
import numpy as np
import warnings
from phonopy.interface.calculator import read_crystal_structure
from phonopy import Phonopy
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
    with open(pos_yaml_fil) as f:
        data = yaml.safe_load(f)
    supercell_matrix = data['supercell_matrix']
    if mpi.rank == mpi.root:
        log.info("\n")
        log.info("\t " + p.sep)
        log.info("\t SUPERCELL MATRIX:\n")
        print_2D_matrix(np.array(supercell_matrix))
        log.info("\t " + p.sep)
        log.info("\n")
    return supercell_matrix

#
def atomic_displacements_from_file(pos_yaml_fil):
    unitcell = extract_unit_cell()
    supercell_matrix = extract_super_cell(pos_yaml_fil)
    # generate phonon object
    phonon = Phonopy(unitcell, supercell_matrix)
    #generate_irred_to_full_displacements_map(pos_yaml_fil, phonon)
    '''
    # Step 5: Access the displacement dataset
    irred_displacements = data['displacements']
    print(f"Number of displaced supercells: {len(irred_displacements)}")
    print(irred_displacements[0])
    print(irred_displacements[1])
    print(irred_displacements[2])
    print(irred_displacements[3])
    print(irred_displacements[4])
    print(len(unitcell))
    symmetry = phonon.symmetry.dataset
    print(phonon._symprec)
    print(len(phonon.unitcell.scaled_positions))
    lattice = phonon.unitcell.cell
    print(lattice)
    exit()
    print(len(symmetry['rotations']))
    print(len(symmetry['translations']))
    print(symmetry['equivalent_atoms'])
    # irred. displ.
    irred_dofs = []
    for disp in irred_displacements:
        print(disp.keys())
        at = disp['atom']
        vec = disp['displacement']
        dir_idx = np.argmax(vec)
        irred_dofs.append((at, dir_idx))
    print(irred_dofs)
    set_map_operations()
    '''

'''
def get_atom_permutation(rot, trans, positions_frac, tol):
    nat = len(positions_frac)
    permut = np.empty(nat, dtype=int)
    for i in range(nat):
        Rp = np.dot(rot, positions_frac[i]) + trans
        Rp %= 1.0
        found_eqv_atom = False
        for j in range(nat):
            dd = Rp - positions_frac[j]
            dd -= np.round(dd)
            dist = np.linalg.norm(dd)
            if dist < tol:
                permut[i] = j
                found_eqv_atom = True
                break
        if not found_eqv_atom:
            log.warning(f"Could not find matching atom for atom {i} under symmetry operation")
    return permut

def generate_irred_to_full_displacements_map(pos_yaml_fil, phonon):
    with open(pos_yaml_fil) as f:
        data = yaml.safe_load(f)
    irred_displacements = data['displacements']
    print(f"Number of displaced supercells: {len(irred_displacements)}")
    # symmetry operations
    sym_ops = phonon.symmetry.dataset
    rotations = sym_ops['rotations']
    translations = sym_ops['translations']
    # fractional positions
    frac_pos = phonon.unitcell.scaled_positions
    print(f"Number of atoms: {len(frac_pos)}")
    for R, t in zip(rotations, translations):
        p = get_atom_permutation(R, t, frac_pos, phonon._symprec)
        print(p)

def set_map_operations():
    warnings.warn(
        "Symmetry._set_map_operations is deprecated."
        "This was replaced by "
        "_get_map_operations_from_permutations.",
        DeprecationWarning,
        stacklevel=2,
    )
    ops = _symmetry_operations
    pos = self._cell.scaled_positions
    lattice = self._cell.cell
    map_operations = np.zeros(len(pos), dtype="intc")

    for i, eq_atom in enumerate(self._map_atoms):
        for j, (r, t) in enumerate(zip(ops["rotations"], ops["translations"])):
            diff = np.dot(pos[i], r.T) + t - pos[eq_atom]
            diff -= np.rint(diff)
            dist = np.linalg.norm(np.dot(diff, lattice))
            if dist < self._symprec:
                map_operations[i] = j
                break
    self._map_operations = map_operations
'''