import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
from torchinfo import summary
import math
def rastrigin(*X, **kwargs):
    A = kwargs.get('A', 10)
    return A + sum([(x**2 - A * np.cos(2 * math.pi * x)) for x in X])

def objective_function(individual,dim):
    # メッシュグリッドの作成
    x=individual[0]
    y=individual[1]
    X, Y = np.meshgrid(x, y)

    # Z値の計算 (ここではシンプルな関数を使用)
    X, Y = np.meshgrid(X, Y)

    Z = rastrigin(X, Y, A=10)

    return Z

# パラメータの設定
dim = 2
max_gen = 100


pop_size = 5
offspring_size = 200
bound = 5.12

# 初期集団の生成
def init_population(pop_size, dim, bound):
    return [np.random.uniform(-bound, bound, dim) for _ in range(pop_size)]

# 適合度の計算
def evaluate_population(population):
    return [objective_function(individual, dim) for individual in population]

def genetic_algorithm(dim, max_gen, pop_size, offspring_size, bound):
    population = init_population(pop_size, dim, bound)
    for generation in range(max_gen):
        fitness = evaluate_population(population)
    
    return population, fitness

population, fitness = genetic_algorithm(dim, max_gen, pop_size, offspring_size, bound)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

np_array = np.array(population, dtype=np.float32)
x_data = np_array.squeeze()
x_data = x_data.squeeze()
# x_data = torch.from_numpy(np_array).to(device)
print(x_data)
np_array1 = np.array(fitness, dtype=np.float32)
y_data = np_array1.squeeze()
# y_data = torch.from_numpy(np_array1).unsqueeze(1).to(device)
print(y_data)
# test_population, test_fitness = genetic_algorithm(dim, max_gen, pop_size, offspring_size, bound)

# test_np_array = np.array(test_population, dtype=np.float32)
# test_x_data = torch.from_numpy(test_np_array).to(device)

# test_np_array1 = np.array(test_fitness, dtype=np.float32)
# test_y_data = torch.from_numpy(test_np_array1).unsqueeze(1).to(device)

from smt.surrogate_models import RBF

sm = RBF(d0=5)
sm.set_training_values(np.double(x_data), np.double(y_data))
sm.train()

# xとyの範囲を設定
# 1変数の範囲を定義
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)

# メッシュグリッドの作成
X_teacher, Y_teacher = np.meshgrid(x, y)

# Z値の計算 (ここではシンプルな関数を使用)
X_teacher, Y_teacher = np.meshgrid(X_teacher, Y_teacher)

Z_teacher = rastrigin(X_teacher, Y_teacher, A=10)
print(Z_teacher)

# 2変数用のグリッドを作成
X1, X2 = np.meshgrid(x, x)

# 2変数の形状に合わせて (10000, 2) の形に変換
X = np.vstack([X1.ravel(), X2.ravel()]).T


print(Z_teacher.shape)
Z = sm.predict_values(X)
print(Z)
print(Z.shape)
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
Z=Z.reshape(100,100)
# 等高線を描画
plt.contour(x, y, Z, levels=10, cmap='viridis')

# カラーバーの追加
plt.colorbar()

# グラフの表示
plt.savefig(' s_rastrigin_contour.png')
plt.close()

# 塗りつぶされた等高線を描画
plt.contourf(x, y, Z, levels=10, cmap='viridis')

# カラーバーの追加
plt.colorbar()

# グラフの表示
plt.savefig('s_rastrigin_contour_color.png')
plt.close()


# 塗りつぶされた等高線を描画
plt.contourf(x, y, Z, levels=10, cmap='viridis')

# カラーバーの追加
plt.colorbar()

# グラフの表示
plt.savefig('s_rastrigin_contour_num.png')
plt.close()

