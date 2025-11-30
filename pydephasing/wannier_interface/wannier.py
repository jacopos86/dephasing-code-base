import numpy as np
from scipy.interpolate import interp1d
from pydephasing.parallelization.mpi import mpi
from pydephasing.utilities.log import log
from pydephasing.set_param_object import p
from pydephasing.utilities.plot_functions import plot_elec_struct

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
        """
        Read the reduced Wannier Hamiltonian from file and reshape it properly.
        Handles spin-orbit / vector-spin and no-spin / z-spin cases.
        Ensures Hwan[ik] is Hermitian.
        """
        spintype = self.elec_struct.spintype
        if spintype in ('spin-orbit', 'vector-spin'):
            Hred = np.fromfile(self.WMLH_FILE, dtype=np.complex128)
            assert(Hred.shape == self.nBands**2 *self.kfoldProd)
        elif spintype == 'no-spin' or spintype == 'z-spin':
            '''spin index in nBands'''
            Hred = np.fromfile(self.WMLH_FILE)
            assert(Hred.shape == self.nBands**2 *self.kfoldProd)
        else:
            log.error('not recognized spintype: ' + spintype)
        Hred = Hred.reshape(self.kfoldProd,self.nBands,self.nBands).swapaxes(1, 2)
        iReduced = np.dot(np.mod(self.cellMap, self.kfold[None,:]), self.kStride)
        self.Hwan = self.wanWeights * Hred[iReduced]
    def check_Hw_hermitean(self):
        for i, R in enumerate(self.cellMap):
            # find the index of -R
            j = np.where((self.cellMap == -R).all(axis=1))[0]
            if len(j) == 1:
                herm = np.allclose(self.Hwan[i], self.Hwan[j[0]].conj().T, atol=1e-8)
                if herm is False:
                    log.warning(str(i) + " - " + str(j[0]) + " -> R=" + str(R) + "H(R) != H(-R)^dagg")
            else:
                log.warning(str(i) + " " + str(R) + "no -R found")
    def compute_band_struct(self):
        # Fourier transf. to k space
        Hk = np.tensordot(np.exp((2j*np.pi)*np.dot(self.Kpts, self.cellMap.T)), self.Hwan, axes=1)
        Ek, Vk = np.linalg.eigh(Hk)
        if mpi.rank == mpi.root:
            log.info("\t shape Ek: " + str(Ek.shape))
            log.info("\t Vk shape: " + str(Vk.shape))
            log.info("\n")
            #--- Save:
            out_file = p.write_dir + "/wannier.eigenvals"
            np.savetxt(out_file, Ek)
        mpi.comm.Barrier()
        return Ek
    def plot_band_structure(self):
        # read MLW cell map
        self.read_MLWF_cell_map()
        # read weights
        self.read_wan_weights()
        # get K pts folding
        self.get_Kpts_folding()
        # Hwann
        self.compute_Hwann()
        # check hermitean matrix
        self.check_Hw_hermitean()
        # read k points
        if mpi.rank == mpi.root:
            log.info("\t INTERPOLATE K POINTS ")
        self.interp_k_pts()
        Ew = self.compute_band_struct()
        if self.elec_struct.spintype in ('spin-orbit', 'vector-spin'):
            Eks = self.elec_struct.Eks[0]
        # call plot function
        if mpi.rank == mpi.root:
            mu = self.elec_struct.mu
            plot_elec_struct(Ew, Eks, mu)
        mpi.comm.Barrier()