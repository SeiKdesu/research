import numpy as np
import matplotlib.pyplot as plt
from nngp import NNGP

L = 20
n_training = 11
n_test = 30
sigma_w_2 = 2
sigma_b_2 = 1
sigma_eps = 0.2

# 修正された signal 関数
def signal(x):
    t1 = 20
    t2 = - 20 * np.exp(- 0.2 * np.sqrt(1.0 / len(x) * np.sum(x ** 2)))
    t3 = np.e
    t4 = - np.exp(1.0 / len(x) * np.sum(np.cos(2 * np.pi * x)))
    return t1 + t2 + t3 + t4

# トレーニングデータとテストデータ
training_data = np.linspace(-10, 10, n_training).reshape((n_training, 1))
training_output = signal(training_data)
training_targets = training_output + sigma_eps * np.random.randn(n_training, 1)

test_data = np.linspace(-10, 10, n_test).reshape(n_test, 1)

# NNGP モデルの作成とトレーニング
regression = NNGP(
    training_data,
    training_targets,
    test_data, L,
    sigma_b_2=sigma_b_2,
    sigma_w_2=sigma_w_2,
    sigma_eps_2=sigma_eps**2)

regression.train()

# 予測と共分散行列
predictions, covariance = regression.predict()
variances = np.diag(covariance)

# プロット
plt.scatter(training_data, training_targets, label='train')

# signal 関数の出力とプロット用データを1次元に変換
plt.plot(test_data.flatten(), signal(test_data).flatten(), label='signal', ls='--', color='black', alpha=.7, lw=1)

# predictions と test_data を 1次元に変換してから errorbar でプロット
plt.errorbar(test_data.flatten(), predictions.flatten(), yerr=variances, color='red', label='prediction')

# グラフの設定と保存
plt.legend()
plt.savefig('1D_regression')
plt.savefig('1D_reg.png')
