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

x_j = np.arange(-95, 95, 100)
y_j = sample_func_g(x_j) 

x = np.arange(-101, 101, 100)
y = sample_func_g(x)

# 2D表示（元のコードのプロット）
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

# 重みと基底関数の中心点を出力
centers = interp_model.nodes
weights = interp_model.A

print("Centers (基底関数の中心点):")
print(centers)
print("Weights (重み):")
print(weights)
