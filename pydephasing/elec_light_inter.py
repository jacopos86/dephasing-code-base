from abc import ABC




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