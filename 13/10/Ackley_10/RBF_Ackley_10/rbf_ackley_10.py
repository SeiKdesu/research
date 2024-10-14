import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
from torchinfo import summary


def Rosenbrock(x, n):
    value = 0
    for i in range(n - 1):
        value += 100 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2
    return value
def dixon_price(x):
    n = len(x)
    term1 = (x[0] - 1) ** 2
    term2 = sum([i * (2 * x[i]**2 - x[i-1])**2 for i in range(1, n)])
    return term1 + term2

def powell(x):
    n = len(x)
    if n % 4 != 0:
        raise ValueError("Input vector length must be a multiple of 4.")
    
    sum_term = 0
    for i in range(0, n, 4):
        term1 = (x[i] + 10 * x[i+1]) ** 2
        term2 = 5 * (x[i+2] - x[i+3]) ** 2
        term3 = (x[i+1] - 2 * x[i+2]) ** 4
        term4 = 10 * (x[i] - x[i+3]) ** 4
        sum_term += term1 + term2 + term3 + term4
    
    return sum_term
def ackley(x, dim):
    sum1=0.0
    sum2=0.0
    a=20
    b=0.2
    c=2*np.pi
    if len(x) != dim:
        raise ValueError(f"Input vector length must be {dim}.")
    for i in range(dim):
        sum1 += (x[i]**2)
        sum2 += np.cos(c * x[i])
    term1 = -a * np.exp(-b * np.sqrt(sum1 / dim))
    term2 = -np.exp(sum2 / dim)
    
    return term1 + term2 + a + np.exp(1)
def objective_function(x,dim):
    return ackley(x,dim)
    # n_rosenbrock = 3
    # n_dixon=3
    # n_powell=4
    # rosen_value = Rosenbrock(x[:n_rosenbrock], n_rosenbrock)
    # dixon_value = dixon_price(x[n_rosenbrock:n_rosenbrock+n_dixon])
    # powell_value= powell(x[n_rosenbrock+n_dixon:n_rosenbrock+n_dixon+n_powell])
    # return rosen_value + dixon_value+ powell_value

# パラメータの設定
dim = 10
max_gen = 100
pop_size = 1000
offspring_size = 300
bound = 5
from datetime import datetime

# 現在の時刻を取得
current_time = datetime.now()
name = f'{current_time}_{pop_size}'



# 初期集団の生成
def init_population(pop_size, dim, bound):
    return [np.random.uniform(-bound, bound, dim) for _ in range(pop_size)]

# 適合度の計算
def evaluate_population(population):
    return [objective_function(individual, dim) for individual in population]

def genetic_algorithm(dim, max_gen, pop_size, offspring_size, bound):
    population = init_population(pop_size, dim, bound)
    # for generation in range(max_gen):
    fitness = evaluate_population(population)
    
    return population, fitness

population, fitness = genetic_algorithm(dim, max_gen, pop_size, offspring_size, bound)



x_j = np.array(population, dtype=np.float32)    
y_j = np.array(fitness, dtype=np.float32)
print(x_j.shape)
np.savetxt(f"acc_loss/{name}_pop.txt", x_j, fmt='%.6f')  # フォーマットを指定

import numpy as np
from scipy.interpolate import Rbf
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


#c
# fig = plt.figure(figsize=(8, 5))
# ax = fig.add_subplot(111)
# ax.plot(x, y, color='magenta', lw=3, label=r'$g(x)$')
# ax.scatter(x_j, y_j, color='blue', s=100, zorder=2, label=r'$x_j$')
# ax.grid()
# ax.set_xlim(-100, 100)
# ax.set_ylim(0, 140)
# ax.legend(fontsize=16)
# plt.savefig('rbf_ex_2d.png')
# plt.close()

output=0
# function_list = ['multiquadric', 'inverse', 'gaussian', 'linear', 'cubic', 'quintic', 'thin_plate']
function_list = ['gaussian']
for i, function in enumerate(function_list):
    # 3D表示用にX, Y軸を拡張
    x_range = np.linspace(-5, 5, 100)
    y_range = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x_range, y_range)
    # Rosenbrock関数の計算 (n=2の場合)
    Z = np.zeros(X.shape)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = Rosenbrock([X[i, j], Y[i, j]], 2)
    # 3D表示
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 元の関数の表示
    ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.6)

    # 補完した結果を表示
    print('1inputのノード特徴量',x_j)
    print('2inputのノード特徴量',y_j)
    print(x_j[:,0].shape,x_j[:,1].shape,y_j.shape)
    interp_model = Rbf(x_j[:,0], x_j[:,1],y_j, function=function)

    print(X[0][32],Y[0][32],Z[0][32])
    Z_interp = interp_model(X,Y)
    print('outputのノードの特徴量：出力の値',Z_interp[int(x_j[0,0])][int(x_j[0,1])])


    # # 補間関数のプロット
    # ax.plot_wireframe(X, Y, Z_interp, color='green', label=f'{function} RBF', linewidth=0.5)

    # # 3Dの設定
    # ax.set_xlabel('X axis')
    # ax.set_ylabel('Y axis')
    # ax.set_zlabel('Z axis')
    # ax.set_title('3D Interpolation with RBF')
    # plt.legend(loc='upper right')

    # # 画像を保存
    # plt.savefig(f'rosenbrock/{name}_rbf_fig.png')  # 保存
    # plt.close()
error = Z - Z_interp
mse = np.mean(error**2)

# 結果を表示
print(f"Mean Squared Error (MSE) between Z and Z_interp: {mse}")



# print('ここまで')
# 重みと基底関数の中心点を出力
weight = interp_model.nodes



print("nodes:hidden to ouput wight ")
print(weight)

# # 元の関数 Z の等高線表示
# plt.figure(figsize=(10, 8))
# plt.contour(X, Y, Z, levels=30, cmap='viridis')
# plt.title("Contour Plot of Original Function Z")
# plt.colorbar()
# plt.savefig(f'rosenbrock/{name}_rbf_original_contour.png')  # 保存
# plt.close()
# # 補間された関数 Z_interp の等高線表示
# plt.figure(figsize=(10, 8))
# plt.contour(X, Y, Z_interp, levels=30, cmap='viridis')
# plt.title("Contour Plot of Interpolated Function Z_interp")
# plt.colorbar()
# plt.savefig(f'rosenbrock/{name}_rbf_predict_contour.png')  # 保存