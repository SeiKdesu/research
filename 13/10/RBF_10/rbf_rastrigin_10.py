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
def objective_function(x,dim):
    return Rosenbrock(x,dim)
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
pop_size = 100
offspring_size = 200
bound = 100
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


# function_list = ['multiquadric', 'inverse', 'gaussian', 'linear', 'cubic', 'quintic', 'thin_plate']
function_list = ['gaussian']
for i, function in enumerate(function_list):
    # -100から100までの範囲をpop_size個に分割して10次元ベクトルを生成
    x_population = np.linspace(-100, 100, pop_size * dim).reshape(pop_size, dim)
    X=x_population
    # 各個体に対してRosenbrock関数を計算
    Z = np.array([Rosenbrock(x, dim) for x in x_population])
    # 3D表示
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 元の関数の表示

    # 補完した結果を表示
    print('1inputのノード特徴量',x_j)
    print('2inputのノード特徴量',y_j)
    print(x_j[:,0].shape,x_j[:,1].shape,y_j.shape)
    """変更点"""
    interp_model = Rbf(x_j[:,0], x_j[:,1],x_j[:,2],x_j[:,3],x_j[:,4],x_j[:,5],x_j[:,6],x_j[:,7],x_j[:,8],x_j[:,9],y_j, function=function)
    # print(x_population[0][32],x_populatiZ[0][32])
    Z_interp = interp_model(x_population[:,0],x_population[:,1],x_population[:,2],x_population[:,3],x_population[:,4],x_population[:,5],x_population[:,6],x_population[:,7],x_population[:,8],x_population[:,9])
    """ここまで変更点"""

error = Z - Z_interp
mse = np.mean(error**2)

# 結果を表示
print(f"Mean Squared Error (MSE) between Z and Z_interp: {mse}")


print('di:データ値をもつ１次元配列',interp_model.di)
# print('smooth:近時の滑らかさ',interp_model.smooth)
# print('mode:1次元かN次元か',interp_model.mode)
# print('距離関数',interp_model.norm)
# print('ここから')
print('episilon',interp_model.epsilon)
# print('ここまで')
# 重みと基底関数の中心点を出力
centers = interp_model.nodes
weights = interp_model.A

# print("nodes:weight for hidden to output")
# print(centers)
# print("A: weight for input to hidden")
# print(weights)

# 元の関数 Z の等高線表示
# plt.figure(figsize=(10, 8))
# plt.contour(X,  Z, levels=30, cmap='viridis')
# plt.title("Contour Plot of Original Function Z")
# plt.colorbar()
# plt.savefig(f'rosenbrock/{name}_rbf_original_contour.pdf')  # 保存
# plt.close()
# # 補間された関数 Z_interp の等高線表示
# plt.figure(figsize=(10, 8))
# plt.contour(X,  Z_interp, levels=30, cmap='viridis')
# plt.title("Contour Plot of Interpolated Function Z_interp")
# plt.colorbar()
# plt.savefig(f'rosenbrock/{name}_rbf_predict_contour.pdf')  # 保存