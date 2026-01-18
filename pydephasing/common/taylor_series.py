import matplotlib.pyplot as plt
import math
import numpy as np
#
# Taylor series class
class TaylorSeries():
    def __init__(self, x, fx, dfx, order, center=0):
        self.center = center
        self.x = x
        self.fx = fx
        self.dfx = dfx
        self.order = order
        self.coeff = []
    def compute_taylor_exp_coeff(self):
        # compute taylor exp. coefficients
        for n in range(self.order+1):
            cn = self.dfx[n] / math.factorial(n)
            #print(cn, self.dfx[n], math.factorial(n))
            self.coeff.append(cn)
    def get_exp_coeff(self):
        return np.array(self.coeff)
    def set_taylor_exp(self):
        self.tex = np.zeros(len(self.x))
        # iterate over coeff.
        for n in range(self.order+1):
            self.tex[:] += self.coeff[n] * (self.x[:] - self.center) ** n
    def display_result(self):
        # Display the result
        plt.plot(self.x, self.fx, linewidth=1.5)
        plt.plot(self.x, self.tex, '--', linewidth=1.5)
        plt.xlim([0., 1.])
        plt.ylim([-1., 1.])
        plt.ylabel('$I_{x}(t)$')
        plt.grid()
        plt.show()