from scipy.constants import physical_constants
from scipy.constants import electron_volt
import numpy as np
#
# physical constants module
#
mp = physical_constants["proton mass"][0]
mp = mp / electron_volt * 1.E30 * 1.E-20
# proton mass (eV fs^2/Ang^2)
mp = mp * 1.E-6                      # eV ps^2/Ang^2
#
THz_to_ev = physical_constants["Planck constant"][0]
# J sec
THz_to_ev = THz_to_ev / electron_volt * 1.E12
#
kb = physical_constants["Boltzmann constant"][0]
kb = kb / electron_volt              # eV/K
#
hbar = physical_constants["Planck constant over 2 pi"][0]
hbar = hbar / electron_volt * 1.E12  # eV ps
#
# tolerance parameter
#
eps = 1.E-7
#
# electron gyromagnetic ratio : gamma_e / 2pi
#
gamma_e = physical_constants["electron gyromag. ratio"][0]
gamma_e = gamma_e * 1.E-12 * 1.E-4 
gamma_e = gamma_e / (2.*np.pi)       # THz / G
#
# nuclear gyromagnetic ratio  : gamma_n / 2pi
#
gamma_n = physical_constants["nuclear magneton in MHz/T"][0]
gamma_n = gamma_n * 1.E-4            # MHz / G