import numpy as np
import yaml
from pydephasing.log import log
from pydephasing.set_param_object import p
from pydephasing.mpi import mpi
from common.phys_constants import mp
# define atomic structure global list
class AtomicStructureClass:
    # initialization
    def __init__(self):
        self.nat = None
        # keys
        self.supercell_key = ''
        self.unitcell_key = ''
        self.atoms_pos_key = ''
        self.lattice_key = ''
        self.mass_key = ''
        self.coords_key = ''
        # arrays
        self.atoms_dict = None
        self.index_to_ia_map = None
        self.index_to_idx_map = None
        self.atoms_mass = None
        # supercell
        self.supercell_size = None
        self.supercell_grid = None
        self.lattice = None
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
#
atoms = AtomicStructureClass()