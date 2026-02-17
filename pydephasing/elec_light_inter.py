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
            E = self.enk[:,ik, isp]
            dE = np.abs(E[:,None] - E[None,:] + hw)
            #if ik == 40:
            #    print("*"*20 + f"AG ik 200 stop")
            #    print("*"*20 + f"AG dE stop")
            #    print("*"*20 + f"AG dE stop{hw}")
            #    print("*"*20 + f"AG E {dE.max()} {dE.min()}")
            #    exit()
            #if np.any( dE < 1.0E-6):
            #    print("*"*20 + f"AG dE stop")
            #    print("*"*20 + f"AG t {t}")
            #    print("*"*20 + f"AG A {A}")
            #    print("*"*20 + f"AG P {self.pe_k[:,:,ik]}")
            #    print("*"*20 + f"AG E {dE.max()} {dE.min()}")
            P_np = np.zeros((self.nbnd,self.nbnd), dtype=np.complex128)
            #if ik == 200:
            #    print("*"*20 + f"AG ik 200 stop")
            #    print("*"*20 + f"AG t {t}")
            #    print("*"*20 + f"AG A {A}")
            #    print("*"*20 + f"AG P {self.pe_k[:,:,ik]}")
            #    print("*"*20 + f"AG E {dE.max()} {dE.min()}")
            
            for ib in range(self.nbnd):
                for jb in range(self.nbnd):
                    #if t >= 1:
                    #    print("*"*20 + f"AG {np.dot(self.pe_k[ib,jb,ik], A)}")
                    #    #exit()
                    P_np[ib,jb] = np.dot(self.pe_k[ib,jb,ik], A)
            self.P.setValuesBlocked([i], [i], P_np.flatten(), addv=PETSc.InsertMode.INSERT_VALUES)

        self.P.assemble()
