import numpy as np
from scipy.interpolate import interp1d
from parallelization.mpi import mpi
from utilities.log import log
from pydephasing.set_param_object import p

class Wannier:
    def __init__(self, elec_struct, CELLMAP_FILE, WWEIGHTS_FILE, WMLH_FILE):
        self.Kpts = None
        self.nK = None
        self.nBands = None
        # elec. structure object
        self.elec_struct = elec_struct
        # cell map
        self.nCells = None
        self.cellMap = None
        self.wanWeights = None
        self.kfold = None
        self.kfoldProd = None
        self.kStride = None
        # wannier hamiltonian
        self.Hwan = None
        # files
        self.CELLMAP_FILE = CELLMAP_FILE
        self.WWEIGHTS_FILE = WWEIGHTS_FILE
        self.WMLH_FILE = WMLH_FILE
    def interp_k_pts(self, n_interp=10):
        # interpolate finer k path
        xIn = np.arange(self.elec_struct.nkpt)
        x = (1./n_interp)*np.arange(1+n_interp*(self.elec_struct.nkpt-1))
        self.Kpts = interp1d(xIn, self.elec_struct.Kpts, axis=0)(x)
        self.nK = self.Kpts.shape[0]
    def read_MLWF_cell_map(self):
        self.cellMap = np.loadtxt(self.CELLMAP_FILE)[:,0:3].astype(int)
        if mpi.rank == mpi.root:
            log.info("\t Wannier nCells: " + str(self.cellMap.shape[0]))
        self.nCells = self.cellMap.shape[0]
    def read_wan_weights(self):
        self.wanWeights = np.fromfile(self.WWEIGHTS_FILE)
        # set nBands
        self.set_nBands()
        self.wanWeights = self.wanWeights.reshape((self.nCells,self.nBands,self.nBands)).swapaxes(1,2)
    def set_nBands(self):
        self.nBands = int(np.sqrt(self.wanWeights.shape[0] / self.nCells))
        if mpi.rank == mpi.root:
            log.info("\t n. wannier bands: " + str(self.nBands))
            log.info("\t " + p.sep)
    def get_Kpts_folding(self):
        TOTAL_E_FILE = self.elec_struct.OUT_FILE
        for line in open(TOTAL_E_FILE, "r"):
            if line.startswith('kpoint-folding'):
                self.kfold = np.array([int(tok) for tok in line.split()[1:4]])
        self.kfoldProd = np.prod(self.kfold)
        self.kStride = np.array([self.kfold[1]*self.kfold[2], self.kfold[2], 1])
    def compute_Hwann(self):
        Hred = np.fromfile(self.WMLH_FILE)
        print(Hred.shape)
        print(type(Hred[0]))
    def plot_band_structure(self):
        # read MLW cell map
        self.read_MLWF_cell_map()
        # read weights
        self.read_wan_weights()
        # get K pts folding
        self.get_Kpts_folding()
        # Hwann
        self.compute_Hwann()
        # read k points
        if mpi.rank == mpi.root:
            log.info("\t INTERPOLATE K POINTS ")
        self.interp_k_pts()