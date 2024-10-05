import numpy as np
from scipy.interpolate import Rbf
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# def sample_func_g(x):
#     return x**2
def sample_func_g(x):
    value  = 0
    value +=  x**2 - 10 * np.cos(2 * np.pi * x)
    value += 10  * 1
    return value

x_j = np.arange(-95, 95, 10)
y_j = sample_func_g(x_j) 

x = np.arange(-101, 101, 100)
y = sample_func_g(x)

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


function_list = ['multiquadric', 'inverse', 'gaussian', 'linear', 'cubic', 'quintic', 'thin_plate']
for i, function in enumerate(function_list):
    # 3D表示用にX, Y軸を拡張
    x_range = np.linspace(-100, 100, 100)
    y_range = np.linspace(-100, 100, 100)
    X, Y = np.meshgrid(x_range, y_range)
    Z = sample_func_g(X) + sample_func_g(Y)

    # 3D表示
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 元の関数の表示
    ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.6)

    # 補完した結果を表示

    interp_model = Rbf(x_j, y_j, function=function)
    Z_interp = interp_model(X)

    # 補間関数のプロット
    ax.plot_wireframe(X, Y, Z_interp, color='green', label=f'{function} RBF', linewidth=0.5)

    # 3Dの設定
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.set_title('3D Interpolation with RBF')
    plt.legend(loc='upper right')

    # 画像を保存
    plt.savefig(function, dpi=300)
    plt.close()

print('N入力のデータポイント数',interp_model.N)
print('di:データ値をもつ１次元配列',interp_model.di)
print('smooth:近時の滑らかさ',interp_model.smooth)
print('mode:1次元かN次元か',interp_model.mode)
print('距離関数',interp_model.norm)
# 重みと基底関数の中心点を出力
centers = interp_model.nodes
# weights = interp_model.A

print("nodes:補完に使用されるノードの一次元配列")
print(centers)
print("Weights (重み):")
# print(weights)
