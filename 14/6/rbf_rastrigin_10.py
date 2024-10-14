

import matplotlib.pyplot as plt
import numpy as np
import numpy as np
import random

from smt.problems import Rosenbrock


def Rosenbrock(x, n):
    value = 0
    for i in range(n-1):
        value += 100 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2
    return value
def dixon_price(x,n):
    n = len(x)
    term1 = (x[0] - 1) ** 2
    term2 = sum([i * (2 * x[i]**2 - x[i-1])**2 for i in range(51, 100)])
    return term1 + term2
def Rosenbrock1(x, n):
    value = 0
    for i in range(101,150):
        value += 100 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2
    return value
# def booth(xy):
#     """
#     Booth関数を計算します。

#     引数:
#     xy : array-like
#         入力ベクトル [x, y]
    
#     戻り値:
#     float
#         Booth関数の値
#     """
#     x, y = xy[0], xy[1]
    
#     term1 = x + 2*y - 7
#     term2 = 2*x + y - 5
    
#     return term1**2 + term2**2

# def matyas_function(x):
#     return 0.26 * (x[0]**2 + x[1]**2) - 0.48 * x[0] * x[1]


def trid(x,n):
    """
    指定された次元数Dに対してTrid関数を計算します。

    引数:
    n : int
        次元数（変数の数）
    
    戻り値:
    float
        Trid関数の値
    """
    # 入力ベクトル x を生成（例として 1 から n までの整数）
   # x = np.arange(1, n + 1)
    
    # 最初の和の計算
    sum1 = 0
    for i in range(n):
        sum1 += (x[i] - 1)**2
    
    # 2つ目の和の計算（隣接する要素の積）
    sum2 = 0
    for i in range(1, n):
        sum2 += x[i] * x[i-1]
    
    return sum1 - sum2
def objective_function(x,dim):
    n_rosenbrock = 2
    n_dixon=2
    # n_powell=100
    rosen_value = Rosenbrock(x, n_rosenbrock)
    rosen_value2 = Rosenbrock(x,n_dixon)
    # rosen_value2 = Rosenbrock1(x, n_rosenbrock)
    # powell_value= trid(x,n_powell)
    # return rosen_value + dixon_value+ powell_value
    return rosen_value+rosen_value2

# パラメータの設定
dim = 4

max_gen = 3
pop_size = 10
offspring_size = 200
bound_rastrigin = 5.12
bound = 30  # Typical bound for Rosenbrock function

# 初期集団の生成
def init_population(pop_size, dim, bound):
    return [np.random.uniform(-bound, bound, dim) for _ in range(pop_size)]

# 適合度の計算
def evaluate_population(population):
    return [objective_function(individual,dim) for individual in population]

# ルーレット選択
def roulette_wheel_selection(population, fitness):
    # current_best_fitness_index = np.argmin(fitness)
    # return population[current_best_fitness_index]
    max_val = sum(fitness)
    pick = random.uniform(0, max_val)
    current = 0
    for i, f in enumerate(fitness):
        current += f
        if current > pick:
            return population[i]
    return population[-1]

# UNDX交叉操作
def undx_crossover(parent1, parent2, parent3, dim):
    alpha = 0.5 #親の情報をどれだけ持ってくるか
    beta = 0.35 #乱数をどれだけ受け入れるか
    g = 0.5 * (parent1 + parent2)
    d = parent2 - parent1
    norm_d = np.linalg.norm(d)
    if norm_d == 0:#parent2とparent1が等しいとき
        return parent1, parent2
    d = d / norm_d#どれだけ解に近いか。
    
    rand = np.random.normal(0, 1, dim)
    

    child1 = g + alpha * (parent3 - g) + beta * np.dot(rand, d) * d #乱数
    child2 = g + alpha * (g - parent3) + beta * np.dot(rand, d) * d
    for i in range(dim):
        child1[i] = rand[i]
        child2[i] = rand[i]
    
    return child1, child2

# 変異操作
def mutate(individual, bound, mutation_rate=0.01):
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            individual[i] = random.uniform(-bound, bound)
    return individual

# メインの遺伝的アルゴリズム
def genetic_algorithm(dim, max_gen, pop_size, offspring_size, bound):
    population = init_population(pop_size, dim, bound)
    print(population)
    #population=population[0]
    best_individual = None
    best_fitness = float('inf')
    fitness_history = []
    best_fitness_history = []
    avg_fitness_history = []    

            
    for generation in range(max_gen):
        
                
        fitness = evaluate_population(population)
        current_best_fitness = min(fitness)
        avg_fitness = np.mean(fitness)
        best_fitness_history.append(current_best_fitness)
        avg_fitness_history.append(avg_fitness)
        if generation % 1 == 0:
            avg_fitness = np.mean(fitness)
            print(f"Generation {generation}: Best Fitness = {best_fitness}, Average Fitness = {avg_fitness}")

        fitness_history.append(np.mean(fitness))

        new_population = []
        while len(new_population) < offspring_size:
            parent1 = roulette_wheel_selection(population, fitness)
            parent2 = roulette_wheel_selection(population, fitness)
            parent3 = roulette_wheel_selection(population, fitness)
            child1, child2 = undx_crossover(parent1, parent2, parent3, dim)
            new_population.append(mutate(child1, bound))
            if len(new_population) < offspring_size:
                new_population.append(mutate(child2, bound))

        population = population + new_population
        population = sorted(population, key=lambda x: objective_function(x,dim))[:pop_size]

        current_best_fitness = min(fitness)
        if current_best_fitness < best_fitness:
            best_fitness = current_best_fitness
            best_individual = population[fitness.index(current_best_fitness)]
            print('chage')
        # if abs(np.mean(fitness) - best_fitness) < 1e-6 and generation > 1000:
        #     break

    return best_individual, best_fitness, best_fitness_history, avg_fitness_history,population,fitness

# 実行
best_individual, best_fitness, best_fitness_history, avg_fitness_history,pop,fit = genetic_algorithm(dim, max_gen, pop_size, offspring_size, bound)

print(f"最良個体の適合度：{best_fitness}")
print(f"最良個体のパラメータ：{best_individual}")
print(f"最終世代の個体:{pop}")
print(f"最終世代の適応度:{fit}")
x_j = np.array(pop, dtype=np.float32)    
y_j = np.array(fit, dtype=np.float32)
from datetime import datetime

# 現在の時刻を取得
current_time = datetime.now()
name = f'{current_time}_{pop_size}'
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