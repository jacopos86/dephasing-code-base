from dataclasses import dataclass
from typing import Optional

class InputError(Exception):
    """Raised when input data is invalid."""
    pass

class BaseInput(ABC):
    """
    Base input class for parsing core directories from YAML input.
    Attributes:
        work_dir: working directory
        write_dir: output directory
        sep: separator string for logging
    """
    def __init__(self, yaml_file: str = None):
        # working directory
        self.work_dir = None
        # write directory
        self.write_dir = None
        # separator
        self.sep = ''
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
        # extract work_dir
        if 'working_dir' in data:
            self.work_dir = data['working_dir']
        # extract write_dir and create it if needed
        if 'output_dir' in data:
            self.write_dir = data['output_dir']
            self._ensure_write_dir()
    def _ensure_write_dir(self):
        """Create the write directory if it does not exist."""
        os.makedirs(self.write_dir, exist_ok=True)
    def _validate_directories(self):
        """Ensure that work_dir and write_dir are defined and create write_dir."""
        if not self.work_dir:
            raise InputError("Missing 'working_dir' in input YAML.")
        if not self.write_dir:
            raise InputError("Missing 'output_dir' in input YAML.")