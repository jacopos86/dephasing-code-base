import numpy as np
from scipy.interpolate import interp1d
from parallelization.mpi import mpi
from utilities.log import log

class Wannier:
    def __init__(self, EIGENV_FILE, KPOINTS_FILE):
        self.EIG_FILE = EIGENV_FILE
        self.KPTS_FILE = KPOINTS_FILE
        self.nspin = None
        self.Kpts_input = None
        self.nKin = None
        self.Kpts = None
        self.nK = None
    def set_nspin(self):
        pass
    def read_k_pts(self, n_interp=10):
        self.Kpts_input = np.loadtxt(self.KPTS_FILE, skiprows=2, usecols=(1,2,3))
        self.nKin = self.Kpts_input.shape[0]
        # interpolate finer k path
        xIn = np.arange(self.nKin)
        x = (1./n_interp)*np.arange(1+n_interp*(self.nKin-1))
        self.Kpts = interp1d(xIn, self.Kpts_input, axis=0)(x)
        self.nK = self.Kpts.shape[0]
    def read_band_struct(self):
        Edft = np.fromfile(self.EIG_FILE).reshape(self.nsp,self.nKin,-1)
        print(Edft.shape)
    def set_wann_transf(self):
        # read k points
        if mpi.rank == mpi.root:
            log.info("\t EXTRACT K POINTS from -> " + self.KPTS_FILE)
        self.read_k_pts()
        # read DFT band structure
        if mpi.rank == mpi.root:
            log.info("\t EXTRACT BAND STRUCTURE from -> " + self.EIG_FILE)
        #self.read_band_struct()