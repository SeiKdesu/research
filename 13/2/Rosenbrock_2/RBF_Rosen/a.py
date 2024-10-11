import numpy as np

def _h_gaussian( r):

    return np.exp(-(1.0/25.714285714285715*r)**2)

print(_h_gaussian(30))