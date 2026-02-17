from abc import ABC
from pydephasing.parallelization.petsc import PETSc
import numpy as np




#
#   Abstract electron - light interaction class
#

class ElectronLightClass(ABC):
    def __init__(self, pe_k, ext_Apot):
        self.pe_k = pe_k
        self.ext_Apot = ext_Apot

#
# ============================================================
#   Electronic Model Electron-Light Coupling
# ============================================================
#

class ElectronLightCouplTwoBandsModel(ElectronLightClass):
    def __init__(self, pe_k, ext_Apot):
        super().__init__(pe_k, ext_Apot)

    def initialize_P_light(self, He):
        # Set up perterbation with same structure as He
        self.nbnd = He.nbnd
        self.nspin = He.nspin
        self.enk = He.enk
        self.P = He.H0.duplicate(PETSc.Mat.DuplicateOption.DO_NOT_COPY_VALUES)
        self.update_P_light(0)

    def update_P_light(self, t):
        self.P.zeroEntries()

        rstart, rend = self.P.getOwnershipRange()
        block_size = self.P.getBlockSize()
        
        # Iterate only over the systems assigned to this rank
        first = rstart // block_size
        last = rend // block_size
        A = self.ext_Apot.set_A(t)
        hw = self.ext_Apot.omega_d
        for i in range(first, last):
            ik = i // self.nspin
            isp = i % self.nspin
            P_np = np.zeros((self.nbnd,self.nbnd), dtype=np.complex128)
            P_np = np.dot(self.pe_k[...,ik], A) 
            self.P.setValuesBlocked([i], [i], P_np.flatten(), addv=PETSc.InsertMode.INSERT_VALUES)
        self.P.assemble()
