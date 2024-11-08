# coding:utf-8
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import qmc
from sklearn.metrics import mean_squared_error

def Rosenbrock(x):
    value = 0
    for i in range(0,2):
        value += 100 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2
    return value
def Rosenbrock1(x):
    value = 0
    for i in range(3,4):
        value += 100 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2
    
    return value

def objective_function(x):
    values=[]
    for tmp in x:
        tmp1 = Rosenbrock(tmp)
        tmp2 = Rosenbrock1(tmp)
        values.append(tmp1+tmp2)
    return np.array(values).reshape(1,-1)
# カーネル関数の定義
def k(x, mu, sigma):
    return np.array([np.exp(-np.sum((x - m)**2, axis=1) / (2 * sigma**2)) for m in mu])

# パラメータ設定

n_dim = 6       # 次元数
sigma = np.sqrt(1 / 200)
loss_history=[]
pops=range(10,100)
for i in pops:

    n_samples = 10  # サンプル数
    # LHSによるサンプリング
    sampler = qmc.LatinHypercube(d=n_dim)
    x = sampler.random(n_samples) * 1.2 - 0.6  # サンプリング範囲を[-0.6, +0.6]にスケーリング
    y = objective_function(x)  # 目的関数の実行結果
    # y = np.sum(x, axis=1) + np.random.normal(0, 0.05, n_samples)  # ノイズを加えた関数の出力としてyを生成

    # カーネル行列の計算
    Phi = k(x, x, sigma)

    # パラメータthetaの計算
    theta = np.dot(np.dot(np.linalg.inv(np.dot(Phi, Phi.T)), Phi), y.T)

    # 予測用のグリッド作成
    xx = sampler.random(1201) * 1.2 - 0.5  # [-0.6, +0.6]の範囲で再サンプリング
    P = k(xx, x, sigma)
    yy= objective_function(xx)
    y_pred =np.dot(P.T,theta)
    print(y_pred.shape)
    print(yy.shape)
    yy= np.reshape(yy, (1201,))
    error = mean_squared_error(np.dot( P.T, theta ),yy)
    # 結果確認
    print(error)
    loss_history.append(error)



# プロット
plt.figure(figsize=(8, 6))
plt.plot(pops, loss_history, marker='o', linestyle='-', color='b', label='pop Loss')
plt.xlabel('pop')
plt.ylabel('Loss')
plt.title('Loss Function Over POP')
plt.legend()
plt.grid(True)
plt.savefig(f"results/pop_and_loss_zinon.pdf")