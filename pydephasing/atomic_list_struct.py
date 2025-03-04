import numpy as np
import yaml
import sys
from pydephasing.log import log
from pydephasing.set_param_object import p
# define atomic structure global list
class AtomicStructureClass:
    # initialization
    def __init__(self):
        self.atoms_dict = None
        self.index_to_ia_map = None
        self.index_to_idx_map = None
    # set atoms dict
    def extract_atoms_coords(self, nat):
        # read from yaml file
        # direct atoms positions
        with open(p.yaml_pos_file) as f:
            data = yaml.full_load(f)
            key = list(data.keys())[6]
            self.atoms_dict = list(data[key]['points'])
        # check length
        if len(self.atoms_dict) != nat:
            log.error("wrong number of atoms in yaml file...")
            sys.exit(1)
    # set supercell coordinates
    # with respect to unit cell lattice vectors
    def set_supercell_coords(self, nat):
        self.supercell_size = [1, 1, 1]
        # cell coordinate
        self.Rn = np.zeros((3,nat))
    # set index_to_ia_map
    def compute_index_to_ia_map(self, nat):
        self.index_to_ia_map = np.zeros(3*nat, dtype=int)
        for jax in range(3*nat):
            self.index_to_ia_map[jax] = int(jax/3) + 1
    # set index to idx map
    def compute_index_to_idx_map(self, nat):
        self.index_to_idx_map = np.zeros(3*nat, dtype=int)
        for jax in range(3*nat):
            self.index_to_idx_map[jax] = jax%3
#
atoms = AtomicStructureClass()