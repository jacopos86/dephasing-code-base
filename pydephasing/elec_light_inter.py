from abc import ABC




#
#   Abstract electron - light interaction class
#

class ElectronLightClass(ABC):
    def __init__(self):
        self.D = None



#
# ============================================================
#   Electronic Model Electron-Light Coupling
# ============================================================
#

class ElectronLightCouplModel(ElectronLightClass):
    def __init__(self, dipole_strengths):
        super().__init__()
        # dipole coeff.
        self.dipole_coeff = dipole_strengths