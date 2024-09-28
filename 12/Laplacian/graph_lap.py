from LE import LE
import numpy as np
X = np.array([[2,3,4], [1,2,3]]) # nxd
le = LE(X, dim = 3, k = 3, graph = 'k-nearest', weights = 'heat kernel', 
        sigma = 5, laplacian = 'symmetrized')
Y = le.transform()