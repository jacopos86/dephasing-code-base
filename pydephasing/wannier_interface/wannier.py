import numpy as np
from scipy.interpolate import interp1d
from pydephasing.parallelization.mpi import mpi
from pydephasing.utilities.log import log
from pydephasing.set_param_object import p
from pydephasing.utilities.plot_functions import plot_wan_struct
from pydephasing.common.phys_constants import hartree2ev

class Wannier:
    def __init__(self, gs_data_dir, elec_struct):
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
        self.wann_setup = False
        # files
        self.CELLMAP_FILE = gs_data_dir + '/wannier.mlwfCellMap'
        self.WWEIGHTS_FILE = gs_data_dir + '/wannier.mlwfCellWeights'
        self.WMLH_FILE = gs_data_dir + '/wannier.mlwfH'
    def setup_wann_calc(self):
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
        self.wann_setup = True
    def interp_k_pts(self, n_interp=10):
        # interpolate finer k path
        xIn = np.arange(self.elec_struct.nkpt)
        x = (1./n_interp)*np.arange(1+n_interp*(self.elec_struct.nkpt-1))
        Kpts = interp1d(xIn, self.elec_struct.Kpts, axis=0)(x)
        nK = Kpts.shape[0]
        return Kpts, nK
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
    def compute_band_struct(self, Kpts):
        # Fourier transf. to k space
        Hk = np.tensordot(np.exp((2j*np.pi)*np.dot(Kpts, self.cellMap.T)), self.Hwan, axes=1)
        Ek, Vk = np.linalg.eigh(Hk)
        if mpi.rank == mpi.root:
            log.info("\t shape Ek: " + str(Ek.shape))
            log.info("\t Vk shape: " + str(Vk.shape))
            log.info("\n")
            #--- Save:
            out_file = p.write_dir + "/wannier.eigenvals"
            np.savetxt(out_file, Ek)
        mpi.comm.Barrier()
        return Ek, Vk
    def plot_band_structure(self, n_interp=10, Ylim=[-0.3, 0.5]):
        if not self.wann_setup:
            self.setup_wann_calc()
        # interp. k points
        if mpi.rank == mpi.root:
            log.info("\t INTERPOLATE K POINTS ")
        Kpts, nK = self.interp_k_pts(n_interp=n_interp)
        # compute wann. bands
        Ew, Vw = self.compute_band_struct(Kpts=Kpts)
        if self.elec_struct.spintype in ('spin-orbit', 'vector-spin'):
            Eks = self.elec_struct.eigv[0]
        # call plot function
        if mpi.rank == mpi.root:
            mu = self.elec_struct.mu
            plot_wan_struct(Ew, Eks, mu, Ylim=Ylim)
        mpi.comm.Barrier()
    def get_band_structure(self, n_interp=1, units="eV"):
        """
        Compute Wannier-interpolated band energies and eigenvectors.
        Returns
        -------
        E  : array (nbnd, nkpt, nspin)
        V  : array (nbnd, nbnd, nkpt, nspin)
         (Eigenvectors: v[n, m, k, s] = component m of eigenvector n)
        """
        # Build k-dependent Hamiltonian matrix
        if not self.wann_setup:
            self.setup_wann_calc()
        nbnd = self.nBands
        Kpts, nK = self.interp_k_pts(n_interp=n_interp)
        nspin = self.elec_struct.nsp_index
        # Allocate arrays
        E  = np.zeros((nbnd, nK, nspin))
        V  = np.zeros((nbnd, nbnd, nK, nspin), dtype=np.complex128)
        # compute wann. bands
        Ew, Vw = self.compute_band_struct(Kpts=Kpts)
        if nspin == 1:
            E[:, :, 0] = Ew.T
            V[:, :, :, 0] = Vw.transpose(1, 2, 0)
        else:
            # duplicate spectrum / adjust if required
            E[:, :, 0] = Ew.T
            E[:, :, 1] = Ew.T
            V[:, :, :, 0] = Vw.transpose(1, 2, 0)
            V[:, :, :, 1] = Vw.transpose(1, 2, 0)
        # Optional conversion
        if units.lower() == "ev":
            E = E * hartree2ev
        elif units.lower() == "ha":
            pass  # already Ha
        else:
            log.error("units must be 'eV' or 'Ha'")
        return E, V