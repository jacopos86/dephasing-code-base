from pathlib import Path
from utilities.log import log
from parallelization.mpi import mpi
from pydephasing.set_param_object import p

class VASP_pawpot:
    def __init__(self, file_name):
        self.__POTCAR = file_name
        fil = Path(self.__POTCAR)
        if not fil.exists():
            log.error("POTCAR file not found")
        self.element = None
        self.zval = None
        self.Z = None
    def extract_PAW_data(self):
        potstr = open(self.__POTCAR).read().split('End of Dataset')[0]
        assert len(potstr.strip()) != 0, "POTCAR string should not be empty!"
        non_radial_part, radial_part = potstr.split('PAW radial sets', 1)
        # read projector functions
        self.read_projectors(non_radial_part)
    def read_projectors(self, datastr):
        dump = datastr.split('Non local Part')
        head = dump[0].strip().split('\n')
        non_local_part = dump[1:]
        # element
        self.element = head[0].split()[1]
        self.zval = float(head[1])
        if mpi.rank == mpi.root:
            log.info("\t element: " + self.element)
            log.info("\t Zval: " + str(self.zval))
        # total charge nucleus
        iconfig = head.index('   Atomic configuration') + 1
        nentr = int(head[iconfig].split()[0])
        self.Z = sum([float(l.split()[-1])
                      for l in head[iconfig+2 : iconfig+2+nentr]])
        if mpi.rank == mpi.root:
            log.info("\t Z: " + str(self.Z))
            log.info("\t " + p.sep)