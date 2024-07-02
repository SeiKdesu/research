import numpy as np
import matplotlib.pyplot as plt

def Schwefel(x, n):
    value = 0
    for i in range(n):
        value += x[i] * np.sin(np.sqrt(np.abs(x[i])))
    value = 418.9828873 * n - value
    return value

# Schwefel関数を目的関数として定義
def objective_function(x):
    n = len(x)
    return Schwefel(x, n)

# 初期化
N = 100  # 粒子数
max_iter = 1000
dim = 10  # 次元数
X = 1000 * np.random.rand(dim, N) - 500 * np.ones((dim, N))
V = 0.2 * np.random.rand(dim, N) - 0.1 * np.ones((dim, N))
Z_plain = np.zeros((1, N))
pbest = np.zeros((dim + 1, N))
pbest[0:dim, :] = X
Zinit = np.apply_along_axis(objective_function, 0, pbest[0:dim, :])
pbest[dim, :] = Zinit

# 初期化した粒子の内で最も良いものをgbestとして格納
ap = np.argmin(Zinit)
gbest = np.append(pbest[0:dim, ap], objective_function(pbest[0:dim, ap]))

w = 0.1
c1 = 0.01
c2 = 0.01
res = np.zeros((2, max_iter))

for i in range(max_iter):
    # N個分の粒子をfor文で計算
    for j in range(N):
        # 位置の更新
        X[:, j] = X[:, j] + V[:, j]
        # 速度の更新
        V[:, j] = w * V[:, j] + c1 * np.random.rand(dim) * (pbest[0:dim, j] - X[:, j]) + c2 * np.random.rand(dim) * (gbest[0:dim] - X[:, j])
        if objective_function(X[:, j]) < pbest[dim, j]:
            pbest[0:dim, j] = X[:, j]
            pbest[dim, j] = objective_function(X[:, j])
    # N個の粒子の内で最適なものを取得し，過去の最適値と比較する
    a = np.argmin(pbest[dim, :])
    if pbest[dim, a] < gbest[dim]:
        gbest[0:dim] = pbest[0:dim, a]
        gbest[dim] = objective_function(gbest[0:dim])

    # イタレーション回数iとその際の最適値を格納
    res[:, i] = np.array([i, gbest[dim]])
    if np.mod(i, 100) == 0:
        plt.plot(res[0, 0:i], res[1, 0:i])
        plt.savefig('2.pdf')

plt.plot(res[0, :], res[1, :])
plt.xlabel('Iteration')
plt.ylabel('Best Objective Function Value')
plt.title('PSO Optimization of 10-dimensional Schwefel Function')
plt.show()

print("Global Best Position:", gbest[0:dim])
print("Global Best Value:", gbest[dim])
