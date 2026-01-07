import numpy as np
import os
import re
import yaml
from pathlib import Path
import mendeleev
from pydephasing.utilities.input_parser import parser
from pydephasing.utilities.log import log
from pydephasing.set_param_object import p
from pydephasing.parallelization.mpi import mpi
from pydephasing.common.phys_constants import mp

# define atomic structure global list
class AtomicStructureClass:
    # initialization
    def __init__(self):
        self.nat = None
        # arrays
        self.atoms_dict = None
        self.index_to_ia_map = None
        self.index_to_idx_map = None
        self.atoms_mass = None
        # supercell
        self.supercell_size = None
        self.supercell_grid = None
        self.lattice = None
    # set index_to_ia_map
    def compute_index_to_ia_map(self):
        self.index_to_ia_map = np.zeros(3*self.nat, dtype=int)
        for jax in range(3*self.nat):
            self.index_to_ia_map[jax] = int(jax/3)
    # set index to idx map
    def compute_index_to_idx_map(self):
        self.index_to_idx_map = np.zeros(3*self.nat, dtype=int)
        for jax in range(3*self.nat):
            self.index_to_idx_map[jax] = jax%3
    # set atoms mass
    def set_atoms_mass(self):
        self.atoms_mass = np.zeros(self.nat)
        for ia in range(self.nat):
            m_ia = self.atoms_dict[ia][self.mass_key]
            self.atoms_mass[ia] = m_ia * mp
            # eV ps^2 / ang^2

class phonopyAtomicStructureClass(AtomicStructureClass):
    def __init__(self):
        super().__init__()
        # keys
        self.supercell_key = ''
        self.unitcell_key = ''
        self.atoms_pos_key = ''
        self.lattice_key = ''
        self.mass_key = ''
        self.coords_key = ''
    #  get atoms keys
    def get_atoms_keys(self):
        # open file
        with open(p.yaml_pos_file) as f:
            data = yaml.full_load(f)
            keys = list(data.keys())
            # dict keys
            for k in keys:
                if k == "supercell_matrix" or k == "Supercell_matrix" or k == "cell_matrix" or k == "Cell_matrix":
                    self.supercell_key = k
            if self.supercell_key == '':
                log.error("supercell matrix key not found")
            for k in keys:
                if k == "unit_cell" or k == "Unit_cell":
                    self.unitcell_key = k
            if self.unitcell_key == '':
                log.error("unit cell key not found")
            # atom position key
            keys = list(data[self.unitcell_key].keys())
            for k in keys:
                if k == "points" or k == "Points":
                    self.atoms_pos_key = k
            if self.atoms_pos_key == '':
                log.error("atoms position key not found")
            for k in keys:
                if k == "lattice" or k == "Lattice":
                    self.lattice_key = k
            if self.lattice_key == '':
                log.error("lattice key not found")
            # mass key
            keys = list(data[self.unitcell_key][self.atoms_pos_key][0].keys())
            for k in keys:
                if k == "mass" or k == "Mass":
                    self.mass_key = k
            if self.mass_key == '':
                log.error("mass key not found")
            # coordinates key
            for k in keys:
                if k == "coordinates" or k == "Coordinates":
                    self.coords_key = k
            if self.coords_key == '':
                log.error("coordinates key not found")
    # set number of atoms
    def set_number_of_atoms(self):
        with open(p.yaml_pos_file) as f:
            data = yaml.full_load(f)
            self.nat = len(list(data[self.unitcell_key][self.atoms_pos_key]))
            if mpi.rank == mpi.root:
                log.info("\t n. of atoms: " + str(self.nat))
    # set atoms data
    def set_atoms_data(self):
        self.get_atoms_keys()
        self.set_number_of_atoms()
        self.compute_index_to_ia_map()
        self.compute_index_to_idx_map()
        # extract atoms coordinate
        self.extract_atoms_coords()
        # set atoms mass
        self.set_atoms_mass()
        # set lattice and super cell
        self.set_supercell_grid()
    # set atoms dict
    def extract_atoms_coords(self):
        # read from yaml file
        # direct atoms positions
        with open(p.yaml_pos_file) as f:
            data = yaml.full_load(f)
            key = self.unitcell_key
            self.atoms_dict = list(data[key][self.atoms_pos_key])
        # check length
        assert len(self.atoms_dict) == self.nat
    # set supercell coordinates
    # with respect to unit cell lattice vectors
    def set_supercell_grid(self):
        with open(p.yaml_pos_file) as f:
            data = yaml.full_load(f)
            grid = list(data[self.supercell_key])
            n = grid[0][0] * grid[1][1] * grid[2][2]
            self.supercell_size = n
            # set grid
            self.supercell_grid = []
            for i1 in range(grid[0][0]):
                for i2 in range(grid[1][1]):
                    for i3 in range(grid[2][2]):
                        self.supercell_grid.append([i1, i2, i3])
            # lattice
            self.lattice = list(data[self.unitcell_key][self.lattice_key])
            if mpi.rank == mpi.root:
                log.info("\t n. supercell grid points: " + str(self.supercell_size))
                log.info("\t supercell grid: " + str(self.supercell_grid))

#
#   JDFTx atomic structure
# 

class JDFTxAtomicStructureClass(AtomicStructureClass):
    def __init__(self):
        super().__init__()
        # regex for "ion SYMBOL x y z [optional_index]" with flexible spacing
        self.pos_pattern = re.compile(
            r"^ion\s+(\w+)\s+([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)\s+"
            r"([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)\s+"
            r"([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)"
        )
        self.mass_key = "mass"
        self.DFT_OUTFILE = None
    # set atoms data
    def set_atoms_data(self, gs_data_dir):
        GSDATA_DIR = Path(gs_data_dir).resolve()
        self.DFT_OUTFILE = GSDATA_DIR / "totalE.out"
        self.set_number_of_atoms()
        self.compute_index_to_ia_map()
        self.compute_index_to_idx_map()
        # extract atoms coordinate
        self.extract_atoms_coords()
        # set atoms mass
        self.set_atoms_mass()
    # set number of atoms
    def set_number_of_atoms(self):
        self.nat = 0
        with open(self.DFT_OUTFILE, "r", encoding="utf-8", errors="replace") as f:
            data = f.readlines()
            reading_block = False
            for line in data:
                # start of the atomic block
                if line.strip().startswith("# Ionic positions"):
                    reading_block = True
                    continue
                # stop reading at the next empty line or comment after block
                if reading_block and (line.strip() == "" or line.strip().startswith("#")):
                    break
                if reading_block:
                    match = self.pos_pattern.search(line)
                    if match:
                        self.nat += 1
    # set atomic coordinates
    def extract_atoms_coords(self):
        keys = ["symbol", "coordinates", "mass"]
        # read from dft output
        self.atoms_dict = []
        with open(self.DFT_OUTFILE, "r", encoding="utf-8", errors="replace") as f:
            data = f.readlines()
            reading_block = False
            for line in data:
                # start of the atomic block
                if line.strip().startswith("# Ionic positions"):
                    reading_block = True
                    continue
                # stop reading at the next empty line or comment after block
                if reading_block and (line.strip() == "" or line.strip().startswith("#")):
                    break
                if reading_block:
                    match = self.pos_pattern.search(line)
                    if match:
                        species = match.group(1)
                        x = float(match.group(2))
                        y = float(match.group(3))
                        z = float(match.group(4))
                        R = np.array([x, y, z])
                        M = mendeleev.element(species).mass
                        self.atoms_dict.append(dict(zip(keys, [species, R, M])))
        assert(len(self.atoms_dict) == self.nat)
    def print_atoms_info(self):
        """
        Prints the atomic information including species and positions from the extracted atoms_dict.
        """
        log.info("\t Atomic Information:")
        if not self.atoms_dict:
            log.error("-> No atom information available...")
        else:
            for atom in self.atoms_dict:
                species = atom["symbol"]
                R = atom["coordinates"]
                mass = atom["mass"]
                log.info(f"\t Species: {species}, Position: ({R[0]:.6f}, {R[1]:.6f}, {R[2]:.6f}), Mass: {mass:.6f}")
            log.info("\t number of atoms: " + str(self.nat))

#  atoms proxy class

class atoms_proxy:
    def __init__(self):
        self._real_atoms = None
    
    def set_input_arguments(self):
        is_testing = os.getenv('PYDEPHASING_TESTING') == '1'
        if is_testing is True:
            args = parser.parse_args(args=[])
        else:
            args = parser.parse_args()
        return args

    def _init(self):
        # object initialization
        args = self.set_input_arguments()
        co = parser.parse_args().co[0]
        if co == 'elec-sys':
            ct2 = args.ct2
            if ct2 == "jdftx":
                self._real_atoms = JDFTxAtomicStructureClass()
            else:
                # default set phonopy
                self._real_atoms = phonopyAtomicStructureClass()
        else:
            self._real_atoms = phonopyAtomicStructureClass()

    def __getattr__(self, attr):
        if self._real_atoms is None:
            self._init()
        return getattr(self._real_atoms, attr)

atoms = atoms_proxy()