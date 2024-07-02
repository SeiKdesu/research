import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 目的関数
def objective_function(X,Y):
    t1 = 20
    t2 = -20 * np.exp(-0.2 * np.sqrt(1.0 / 2 * (X**2 + Y**2 )))
    t3 = np.e
    t4 = -np.exp(1.0 / 2 * (np.cos(2 * np.pi * X)+np.cos(2 * np.pi * Y)))
    return t1 + t2 + t3 + t4

# Figureと3DAxeS
fig = plt.figure(figsize = (8, 8))
ax = fig.add_subplot(111, projection="3d")

# 軸ラベルを設定
ax.set_xlabel("x", size = 16)
ax.set_ylabel("y", size = 16)
ax.set_zlabel("z", size = 16)

# 円周率の定義
pi = np.pi

# (x,y)データを作成
x = np.linspace(-2*pi, 2*pi, 256)
y = np.linspace(-2*pi, 2*pi, 256)

# 格子点を作成
X_p, Y_p = np.meshgrid(x, y)
Z = objective_function(X_p,Y_p)

# # 曲面を描画
ax.plot_surface(X_p, Y_p, Z, cmap = "summer")

# # 底面に等高線を描画
ax.contour(X_p, Y_p, Z, colors = "black", offset = -1)

plt.savefig('objective_function1.pdf')
plt.close()

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

#初期化
N = 100 #粒子数
max_iter = 1000
X = 6*np.random.rand(2, N) - 3*np.ones((2, N)) 
V = 0.2*np.random.rand(2, N) - 0.1*np.ones((2, N))
Z_plain = np.zeros((1,N))
pbest = np.zeros((3,N))
pbest[0:2,:] = X
Zinit = objective_function(pbest[0,:],pbest[1,:]) 
pbest[2,:] = Zinit
#初期化した粒子の内で最も良いものをgbestとして格納
ap = np.argmin(Zinit)
gbest = np.array([pbest[0,ap],pbest[1,ap],objective_function(pbest[0,ap],pbest[1,ap])])

w = 0.1
c1 = 0.01
c2 = 0.01
res = np.zeros((2,max_iter))
for i in range(max_iter):
  # N個分の粒子をfor文で計算
  for j in range(N):
    #位置の更新
    X[:,j] = X[:,j] + V[:,j]
    #速度の更新
    V[:,j] = w*V[:,j] + c1*np.random.rand(1)*(pbest[0:2,j]-X[:,j]) +c2*np.random.rand(1)*(gbest[0:2]-X[:,j])
    if  objective_function(X[0,j],X[1,j]) < pbest[2,j]:
       pbest[0:2,j] = X[:,j]
       pbest[2,j] = objective_function(X[0,j],X[1,j])
  # N個の粒子の内で最適なものを取得し，過去の最適値と比較する
    a = np.argmin(pbest[2,:])
  if pbest[2,a] < gbest[2]:
    gbest[0:2] = pbest[0:2,a]
    gbest[2] = objective_function(gbest[0],gbest[1])
  
  #イタレーション回数iとその際の最適値を格納
  res[:,i] = np.array([i,gbest[2]])
  if np.mod(i,100) == 0:
    plt.plot(res[0,0:i],res[1,0:i])
    plt.savefig('2.pdf')


plt.plot(res[0,:],res[1,:])
print(gbest)

