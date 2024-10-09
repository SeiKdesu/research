import numpy as np
from scipy.interpolate import Rbf
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# サンプル関数
def sample_func_g(x):
    value  = 0
    value +=  x**2 - 10 * np.cos(2 * np.pi * x)
    value += 10 * 1
    return value

# x_jを10次元に変更
x_j = np.random.uniform(-95, 95, (10,))  # 10次元
y_j = sample_func_g(x_j)

# 新しい入力範囲を10次元に変更
x = np.random.uniform(-101, 101, (10,))
y = sample_func_g(x)

# RBF補間を行う
function_list = ['gaussian']
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

    # 補間モデルの作成
    print('前にxの入力+x:10次元のノード特徴量', x_j)
    print('x:2inputのノード特徴量+yの出力', y_j)
    interp_model = Rbf(x_j, y_j, function=function)
    print('ノードの特徴量上とうまく組み合わせ', x_j.shape, Y[0][32], Z[0][32])
    Z_interp = interp_model(X)

    # # 補間結果の表示
    # ax.plot_wireframe(X, Y, Z_interp, color='green', label=f'{function} RBF', linewidth=0.5)

    # 3Dの設定
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.set_title('3D Interpolation with RBF')
    # plt.legend(loc='upper right')

    # # 画像を保存
    # plt.savefig(function, dpi=300)
    # plt.close()

# RBFモデルの重みと基底関数の中心点を出力
