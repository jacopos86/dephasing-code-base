from pathlib import Path

#
#   spin qubit input class
#   -> used for both LR and RT base class
#

class InputSQ(BaseInput):
    """
    spin qubit input class for parsing core directories from YAML input.
    """
    def __init__(self, yaml_file: str = None):
        super().__init__()
        # general GS directory - no for grads
        self.gs_data_dir = None
        # unperturbed directory
        self.unpert_dirs = []
        # grad info file
        self.grad_info = ''
        # magnetic field
        self.B = None
        # ------------------
        # HFI parameters
        # ------------------
        self.nconf = None
        self.nsp = None
        self.fc_core = True
        self.rnd_orientation = False
        # read YAML if provided
        if yaml_file:
            self.read_yaml(yaml_file)
    # read yaml
    def read_yaml(self, yaml_file: str):
        """Read YAML file and extract work_dir, write_dir, and sep."""
        try:
            with open(yaml_file, 'r') as f:
                data = yaml.load(f, Loader=yaml.Loader)
        except FileNotFoundError:
            raise FileNotFoundError(f"Could not find YAML file: {yaml_file}")
        # unpert. dir
        if 'unpert_dir' in data:
            self.gs_data_dir = Path(self.work_dir) / data['unpert_dir']
        # grad info file
        if 'grad_info_file' in data:
            self.grad_info = Path(self.work_dir) / data['grad_info_file']
    # read HFI data
    def read_hfi_data(self, data):
        # read n. configurations
        if 'nconfig' in data:
            self.nconf = data['nconfig']
        # n. spins each config.
        if 'nspins' in data:
            self.nsp = data['nspins']
        # fermi contact term
        if 'core' in data:
            if data['core'] == False:
                self.fc_core = False
        # check random spin
        # orientation
        if 'random_orientation' in data:
            self.rnd_orientation = data['random_orientation']