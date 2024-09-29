from LE import LE
import numpy as np
X = np.array([[0.2,0.3,-0.4,0.6,0.5], [-0.1,-0.2,0.3,0.3,0.2],[0.4,0.2,-0.2,0.3,-0.4],[0.1,-0.4,0.9,0.3,-0.7]]) # nxd
le = LE(X, dim = 4, k = 3, graph = 'k-nearest', weights = 'heat kernel', 
        sigma = 5, laplacian = 'symmetrized')
Y = le.transform()
print(Y)
# le.plot_embedding_2d(colors='red')
# le.plot_embedding_3d(colors='blue')