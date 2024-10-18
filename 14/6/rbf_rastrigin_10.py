import numpy as np


import matplotlib.pyplot as plt



import numpy as np



def Rosenbrock(x, n):
    value = 0
    for i in range(n-1):
        value += 100 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2
    return value
# def dixon_price(x,n):
#     n = len(x)
#     term1 = (x[0] - 1) ** 2
#     term2 = sum([i * (2 * x[i]**2 - x[i-1])**2 for i in range(11, 20)])
#     return term1 + term2
def Rosenbrock1(x, n):
    value = 0
    for i in range(3,n-1):
        value += 100 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2
    return value

def objective_function(p):
    x=p
    n_rosenbrock = 3
    # n_dixon=10
    # n_powell=32
    rosen_value = Rosenbrock(x, n_rosenbrock)
    # dixon_value = dixon_price(x,n_dixon)
    rosen_value2 = Rosenbrock1(x, n_rosenbrock)
    
    # powell_value= powell(x[n_rosenbrock+n_dixon:n_rosenbrock+n_dixon+n_powell],n_powell)
    return rosen_value + rosen_value2

# %%
from sko.GA import GA
dim=6
ga = GA(func=objective_function, n_dim=dim, size_pop=10, max_iter=3, prob_mut=0.001, lb=[-1]*dim, ub=[1]*dim, precision=1e-7)
best_x, best_y = ga.run()



print('best_x:', best_x, '\n', 'best_y:', best_y)

# %% Plot the result
import pandas as pd
import matplotlib.pyplot as plt

Y_history = pd.DataFrame(ga.all_history_Y)
fig, ax = plt.subplots(2, 1)
ax[0].plot(Y_history.index, Y_history.values, '.', color='red')
Y_history.min(axis=1).cummin().plot(kind='line')
plt.savefig('a.png')


def pop ():
    population = ga.X
    population= np.squeeze(population)
    return population
def fitness():
    fitness = ga.Y
    return fitness

population = pop()
fitness= fitness()

from datetime import datetime

# 現在の時刻を取得
current_time = datetime.now()
name = f'{current_time}'


from scikit import *

x_j = np.array(population, dtype=np.float32)    
y_j = np.array(fitness, dtype=np.float32)

np.savetxt(f"acc_loss/{name}_pop.txt", x_j, fmt='%.6f')  # フォーマットを指定

import numpy as np
from scipy.interpolate import Rbf
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import itertools
output=0
# function_list = ['multiquadric', 'inverse', 'gaussian', 'linear', 'cubic', 'quintic', 'thin_plate']
function_list = ['gaussian']
Z = []
Z_interp=[]
num=0
for i, function in enumerate(function_list):
    range_per_dim = np.linspace(-5, 5, 10)  # 各次元の範囲

    # itertools.productを使ってn次元の全組み合わせを逐次的に生成
    for point in itertools.product(range_per_dim, repeat=dim):
        
        Z.append(objective_function(np.array(point)))
        num = num +1
    # 3D表示
    # fig = plt.figure(figsize=(10, 8))
    # ax = fig.add_subplot(111, projection='3d')

    # # 元の関数の表示
    # ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.6)

    # 補完した結果を表示
    # print('1inputのノード特徴量',x_j)
    # print('2inputのノード特徴量',y_j)
    # print(x_j[:,0].shape,x_j[:,1].shape,y_j.shape)
    # RBF補間モデルの作成（n次元対応）
    interp_model = Rbf(*[x_j[:, i] for i in range(x_j.shape[1])], function=function)

    #interp_model = Rbf(x_j[:,0], x_j[:,1],x_j[:,2],x_j[:,3],x_j[:,4],x_j[:,5],x_j[:,6],x_j[:,7],x_j[:,8],x_j[:,9], function=function)

    range_per_dim = np.linspace(-5, 5, 10)  # 各次元の範囲



    # itertools.productを使ってn次元の全組み合わせを逐次的に生成
    for point in itertools.product(range_per_dim, repeat=dim):
        Z_interp.append(interp_model(np.array(point)))

    # 結果を確認
    print(Z_interp)
error = Z - Z_interp
mse = np.mean(error**2)

# 結果を表示
print(f"Mean Squared Error (MSE) between Z and Z_interp: {mse}")



# print('ここまで')
# 重みと基底関数の中心点を出力
weight = interp_model.nodes



print("nodes:hidden to ouput wight ")
print(weight)

