# Taylor series class
class TaylorSeries():
    def __init__(self, func, order, center=0):
        self.center = center
        self.f = func
        self.order = order
        self.d_pts = order * 2
        self.coeff = []