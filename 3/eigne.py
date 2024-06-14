import numpy as np
from numpy import linalg as LA

eienvalues,eigenvectors =LA.eig(np.array([[1,-1,0,0],[-1,3,-1,-1],[0,-1,2,-1],[0,-1,-1,2]]))

print('固有値',eienvalues)
print('固有ベクトル',eigenvectors)