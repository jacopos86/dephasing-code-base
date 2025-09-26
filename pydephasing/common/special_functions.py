import numpy as np
from math import exp

#
# special functions module
#

# 1) delta function
#    input : x, y
# 2) lorentzian
#    input : x, eta
# 3) gaussian
#    input : x, eta

#
#  function 1)
#
def delta(x, y):
	if x == y:
		return 1.
	else:
		return 0.
#
#  function 2)
#
def lorentzian(x, eta):
	ltz = eta/2. / (x ** 2 + (eta/2.) ** 2) * 1./np.pi
	return ltz

#
#   function 3)
#
def gaussian(x, eta):
	return exp(-x ** 2/(2.*eta ** 2)) / np.sqrt(2.*np.pi*eta ** 2)