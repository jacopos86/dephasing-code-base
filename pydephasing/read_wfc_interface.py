from pathlib import Path
from abc import ABC, abstractmethod
import numpy as np
from jarvis.io.vasp.outputs import Wavecar
from utilities.log import log
from common.phys_constants import rytoev, AUTOA

#
#  this interface returns
#  wave function object used for further postprocessing

def read_wfc(gs_dir, struct_0):
    file_name = "WAVECAR"
    file_name = "{}".format(gs_dir + '/' + file_name)
    fil = Path(file_name)
    if fil.exists():
        wfc_obj = vasp_WFC_model(file_name, struct_0)
    else:
        log.error("ONLY WAVECAR INTERFACE AVAILABLE NOW")
    return wfc_obj

#
#  abstract wfc class
#

class WFC(ABC):
    def __init__(self):
        self._nbnd = None
        self._nkpt = None
        self._nspin = None
        self._encut = None
        self._cell = None
        self._rec_cell = None
        self._Vcell = None
        self._wfc = None
        self._wfPREC = None

    @abstractmethod
    def print_info(self):
        raise NotImplementedError("print_info must be implemented")

#
#  VASP WFC model
#

class vasp_WFC_model(WFC):
    def __init__(self, file_name, struct_0):
        super().__init__()
        self.__WAVECAR = file_name
        self.__struct0 = struct_0
        self.HSQDTM = rytoev * AUTOA * AUTOA
        if self.__struct0.has_soc:
            self._npol = 2
        else:
            self._npol = 1
    def print_info(self):
        log.info("\t n. k pts: " + str(self._nkpt))
        log.info("\t n. electronic bands: " + str(self._nbnd))
        log.info("\t n. spin polarization: " + str(self._nspin))
        log.info("\t cutoff energy: " + str(self._encut))
    def check_index(self, isp, ikpt, ibnd):
        assert(1 <= isp <= self._nspin)
        assert(1 <= ikpt <= self._nkpt)
        assert(1 <= ibnd <= self._nbnd)
    def RecPos(self, isp, ikpt, ibnd):
        rec = 2 + (isp - 1) * self._nkpt * (self._nbnd + 1) + \
        (ikpt - 1) * (self._nbnd + 1) + \
        ibnd
        return rec
    def set_wfPREC(self, prec_tag):
        if prec_tag == 45200:
            self._wfPREC = np.complex64
        elif prec_tag == 45210:
            self._wfPREC = np.complex128
        elif prec_tag == 53300 or prec_tag == 53310:
            log.error("WAVECAR format not implemented")
        else:
            log.error("FORMAT TAG NOT RECOGNIZED")
    def set_min_FFT_grid_size(self):
        norm = np.linalg.norm(self._cell, axis=1)
        CUTOFF = np.array(np.sqrt(self._encut / rytoev) / (2.*np.pi) * norm / AUTOA, dtype=int)
        self._ngrid_pts = 2 * CUTOFF + 1
    def extract_JARVIS_WFC(self):
        self._wfc = Wavecar(self.__WAVECAR)
        self._nbnd = self._wfc._nbands
        self._nkpt = self._wfc._nkpts
        self._nspin = self._wfc._nspin
    def extract_header(self):
        self._wfc = open(self.__WAVECAR, "rb")
        # rec 1
        self._wfc.seek(0)
        dump = np.array(np.fromfile(self._wfc, dtype=np.double, count=3), dtype=int)
        self._recl = dump[0]
        self._nspin = dump[1]
        prec_tag = dump[2]
        self.set_wfPREC(prec_tag)
        # rec 2
        self._wfc.seek(self._recl)
        dump = np.fromfile(self._wfc, dtype=np.double, count=12)
        self._nkpt = int(dump[0])
        self._nbnd = int(dump[1])
        self._encut= dump[2]
        self._cell = dump[3:].reshape((3,3))
        self._Vcell = np.linalg.det(self._cell)
        self._rec_cell = np.linalg.inv(self._cell).T
        self.set_min_FFT_grid_size()
        # read band info
        self.readBandData()
    def get_pw_number(self, ikpt):
        return self._nplws[ikpt-1]
    def readBandData(self):
        self._nplws = np.zeros(self._nkpt, dtype=int)
        self._kvecs = np.zeros((self._nkpt, 3))
        self._bands = np.zeros((self._nspin, self._nkpt, self._nbnd))
        self._occs = np.zeros((self._nspin, self._nkpt, self._nbnd))
        for isp in range(self._nspin):
            for ikpt in range(self._nkpt):
                rec = self.RecPos(isp+1, ikpt+1, 1) - 1
                self._wfc.seek(rec * self._recl)
                dump = np.fromfile(self._wfc, dtype=np.double, count=4+3*self._nbnd)
                if isp == 0:
                    self._nplws[ikpt] = int(dump[0])
                    self._kvecs[ikpt] = dump[1:4]
                dump = dump[4:].reshape((-1, 3))
                self._bands[isp, ikpt, :] = dump[:,0]
                self._occs[isp, ikpt, :] = dump[:,2]
    def set_gvectors(self, ikpt, gamma=False):
        """"
        generate G vectors from (G + k)^2 / 2 < ENCUT
        """
        assert(1 <= ikpt <= self._nkpt)
        kvec = self._kvecs[ikpt-1]
        # frequency list
        fx = [ix if ix < self._ngrid_pts[0] / 2 else ix - self._ngrid_pts[0] 
              for ix in range(self._ngrid_pts[0])]
        fy = [iy if iy < self._ngrid_pts[1] / 2 else iy - self._ngrid_pts[1]
              for iy in range(self._ngrid_pts[1])]
        fz = [iz if iz < self._ngrid_pts[2] / 2 else iz - self._ngrid_pts[2]
              for iz in range(self._ngrid_pts[2])]
        if gamma:
            grid = np.array([(fx[ix], fy[iy], fz[iz])
                         for iz in range(self._ngrid_pts[2])
                         for iy in range(self._ngrid_pts[1])
                         for ix in range(self._ngrid_pts[0])
                         if (
                             (fz[iz] > 0) or
                             (fz[iz] == 0 and fy[iy] > 0) or
                             (fz[iz] == 0 and fy[iy] == 0 and fx[ix] >= 0)
                         )], dtype=float
            )
        else:
            grid = np.array([(fx[ix], fy[iy], fz[iz])
                             for iz in range(self._ngrid_pts[2])
                             for iy in range(self._ngrid_pts[1])
                             for ix in range(self._ngrid_pts[0])], dtype=float)
        KENER = self.HSQDTM * np.linalg.norm(
            np.dot(grid + kvec[np.newaxis,:], 2.*np.pi*self._rec_cell), axis=1)**2
        gvec = grid[np.where(KENER < self._encut)[0]]
        if self.__struct0.has_soc:
            assert(self._nplws[ikpt-1] == gvec.shape[0]*2)
        else:
            assert(self._nplws[ikpt-1] == gvec.shape[0])
        return np.asarray(gvec, dtype=int)
    def read_cnk_ofG(self, isp, ikpt, ibnd, norm=False):
        self.check_index(isp, ikpt, ibnd)
        rec = self.RecPos(isp, ikpt, ibnd)
        self._wfc.seek(rec * self._recl)
        # extract coefficients
        npw = self._nplws[ikpt-1]
        dump = np.fromfile(self._wfc, dtype=self._wfPREC, count=npw)
        cnk_G= np.asarray(dump, dtype=np.complex128)
        if norm:
            cnk_G /= np.linalg.norm(cnk_G)
        return cnk_G