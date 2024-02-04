from tensorflow import keras
import matplotlib.pyplot as plt
from scipy.misc import derivative
import math
import numpy as np
#
# Taylor series class
class TaylorSeries():
    def __init__(self, x, fx, order, center=0):
        self.center = center
        self.x = x
        self.y = fx
        self.order = order
        self.d_pts = order * 2
        self.coeff = []
        self.model = None
        # number of points (order) for scipy.misc.derivative
        if self.d_pts % 2 == 0: # must be odd and greater than derivative order
            self.d_pts += 1
        # fit NN model
        self.train_model()
    def compute_taylor_exp(self):
        # fit function
        #y_predicted = self.fit_function_to_expand()
        # display fx / fit
        #self.display_result(y_predicted)
        self.find_coefficients()
    def train_model(self):
        # Create the model
        # to fit the function
        self.model = keras.Sequential()
        self.model.add(keras.layers.Dense(units = 1, activation = 'linear', input_shape=[1]))
        self.model.add(keras.layers.Dense(units = 64, activation = 'relu'))
        self.model.add(keras.layers.Dense(units = 64, activation = 'relu'))
        self.model.add(keras.layers.Dense(units = 64, activation = 'relu'))
        self.model.add(keras.layers.Dense(units = 1, activation = 'linear'))
        self.model.compile(loss='mse', optimizer="adam")
        # Display the model
        self.model.summary()
        self.model.fit(self.x, self.y, epochs=100, verbose=1)
    def f(self, xg):
        if not isinstance(xg, np.ndarray):
            xg = [xg]
        # Compute the output 
        y_predicted = self.model.predict(xg)
        return y_predicted
    def find_coefficients(self):
        print(self.f(0.), self.f(0.5), self.f(1))
        print(derivative(self.f, self.center, dx=1e-6))
        print(derivative(self.f, 0.5, dx=1e-6))
        print(derivative(self.f, self.center, n=3, dx=1e-4, order=5))
        import sys
        sys.exit()
        for i in range(self.order+1):
            self.coeff.append(round(derivative(self.f, self.center, n=i, order=self.d_pts)/math.factorial(i), 5))
    def display_result(self, xg):
        # Display the result
        y_predicted = self.f(xg)
        plt.plot(self.x, self.y, linewidth=1)
        plt.plot(xg, y_predicted, '--', 'r', linewidth=1)
        plt.grid()
        plt.show()