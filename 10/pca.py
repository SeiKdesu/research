import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

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
def dixon_price(x,n):
    n = len(x)
    term1 = (x[0] - 1) ** 2
    term2 = sum([i * (2 * x[i]**2 - x[i-1])**2 for i in range(1, n)])
    return term1 + term2
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


def powell(x,n):
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
    n_rosenbrock = 50
    n_dixon=50
    n_powell=32
    rosen_value = Rosenbrock(x[:n_rosenbrock], n_rosenbrock)
    dixon_value = dixon_price(x[n_rosenbrock:n_rosenbrock+n_dixon],n_dixon)
    rosen_value2 = Rosenbrock(x[n_rosenbrock+n_dixon:n_rosenbrock+n_dixon+n_rosenbrock], n_rosenbrock)
    
    # powell_value= powell(x[n_rosenbrock+n_dixon:n_rosenbrock+n_dixon+n_powell],n_powell)
    return rosen_value + dixon_value+rosen_value2

# パラメータの設定
dim = 150
max_gen = 100
pop_size = 300
offspring_size = 200
bound = 5.12

# 初期集団の生成
def init_population(pop_size, dim, bound):
    return [np.random.uniform(-bound, bound, dim) for _ in range(pop_size)]

def draw_vector(pt_from, pt_to, ax=None):
    """ベクトル描画関数

    Args:
        pt_from: 矢印の始点座標
        pt_to: 矢印の終点座標
        ax: 描画対象axis (デフォルト:None)
    """

    if ax is None:
        ax = plt.gca()
    arrowprops = dict(arrowstyle="->", linewidth=2, shrinkA=0, shrinkB=0)
    # ax.annotate("", pt_to, pt_from, arrowprops=arrowprops)


def main():
    """メイン関数"""

    np.random.seed(123)

    # ===== データの作成と描画
    # n_point = 200
    # data = np.dot(np.random.rand(50, 50), np.random.randn(50, n_point)).T
    # plt.scatter(data[:, 0], data[:, 1])
    # plt.axis("equal")
    data = init_population(pop_size,dim,bound)
    print(data.shape)
    # ===== 主成分分析の実行
    pca = PCA()
    pca.fit(data)

    # 結果の表示
    print(f"主成分: \n{pca.components_}")
    print(f"分散: \n{pca.explained_variance_}")
    print(f"寄与率: \n{pca.explained_variance_ratio_}")

    # ===== 主成分ベクトルを表示してみる
    for var, vector in zip(pca.explained_variance_, pca.components_):
        # 標準偏差分の長さのベクトルを作成
        vec = vector * np.sqrt(var)
        draw_vector(pca.mean_, pca.mean_ + vec)
    plt.show()

    # =====
    data_pca = pca.transform(data)
    plt.scatter(data_pca[:, 0], data_pca[:, 1])
    plt.axis("equal")
    plt.xlabel("component1")
    plt.ylabel("component2")
    plt.show()


if __name__ == "__main__":
    main()