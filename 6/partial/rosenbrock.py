import matplotlib.pyplot as plt
import numpy as np

from smt.problems import Rosenbrock

ndim = 10
problem = Rosenbrock(ndim=ndim)

num = 100
x = np.ones((num, ndim))
x[:, 0] = np.linspace(-2, 2.0, num)
x[:, 1] = 0.0
y = problem(x)

'''
print(x)
print(y)
yd = np.empty((num, ndim))
for i in range(ndim):
    yd[:, i] = problem(x, kx=i).flatten()

print(y.shape)
print(yd.shape)

plt.plot(x[:, 0], y[:, 0])
plt.xlabel("x")
plt.ylabel("y")
plt.savefig('a.png')
'''