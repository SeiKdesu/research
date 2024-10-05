
from smt.problems.problem import Problem

import numpy as np
from smt.problems.problem import Problem

class Rastrigin(Problem):
    def _initialize(self):
        self.options.declare("name", "Rastrigin", types=str)
        self.options.declare("A", 10.0, types=float)

    def _setup(self):
        self.xlimits[:, 0] = -5.12
        self.xlimits[:, 1] = 5.12

    def _evaluate(self, x, kx):
        """
        Arguments
        ---------
        x : ndarray[ne, nx]
            Evaluation points.
        kx : int or None
            Index of derivative (0-based) to return values with respect to.
            None means return function value rather than derivative.

        Returns
        -------
        ndarray[ne, 1]
            Functions values if kx=None or derivative values if kx is an int.
        """
        A = self.options["A"]
        ne, nx = x.shape

        y = np.zeros((ne, 1))
        if kx is None:
            sum_terms = np.sum(x**2 - A * np.cos(2 * np.pi * x), axis=1)
            y[:, 0] = A * nx + sum_terms
        else:
            if kx < nx:
                # Derivative with respect to x[kx]
                term = 2 * x[:, kx] + 2 * A * np.pi * np.sin(2 * np.pi * x[:, kx])
                y[:, 0] = term
            else:
                raise ValueError("kx is out of range")

        return y

import numpy as np
from smt.utils.misc import compute_rms_error

# from smt.problems import Rastrigin
from smt.sampling_methods import LHS
from smt.surrogate_models import LS, QP, KPLS, KRG, KPLSK, GEKPLS, MGP


from smt.surrogate_models import IDW, RBF, RMTC, RMTB
import matplotlib.pyplot as plt
# to ignore warning messages
import warnings
from matplotlib import cm
from datetime import datetime
import os
warnings.filterwarnings("ignore")

def get_time_based_directory(result_dir='result'):
    # 実行時間を使って新しいディレクトリを生成
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")  # 年月日_時分秒形式
    time_dir = os.path.join(result_dir, current_time)
    
    # ディレクトリが存在しなければ作成
    if not os.path.exists(time_dir):
        os.makedirs(time_dir)
    
    return time_dir

# ディレクトリは1度だけ生成して保持
time_based_dir = get_time_based_directory()

########### Initialization of the problem, construction of the training and validation points

ndim = 2
ndoe = 10*ndim # int(10*ndim)
# Define the function
fun = Rastrigin(ndim=ndim)

# Construction of the DOE
# in order to have the always same LHS points, random_state=1
sampling = LHS(xlimits=fun.xlimits, criterion="ese", random_state=1)
xt = sampling(ndoe)
# Compute the outputs
yt = fun(xt)

# Construction of the validation points
ntest = 200  # 500
sampling = LHS(xlimits=fun.xlimits, criterion="ese", random_state=1)
xtest = sampling(ntest)
ytest = fun(xtest)


# To visualize the DOE points
fig = plt.figure(figsize=(10, 10))
plt.scatter(xt[:, 0], xt[:, 1], marker="x", c="b", s=200, label="Training points")
plt.scatter(
    xtest[:, 0], xtest[:, 1], marker=".", c="k", s=200, label="Validation points"
)
plt.title("DOE")
title="DOE"
plt.xlabel("x1")
plt.ylabel("x2")
plt.legend()
save_path = os.path.join(time_based_dir, f"{title}.png")
plt.savefig(save_path)
plt.close()


# To plot the Rastrigin function
x = np.linspace(-2, 2, 50)
res = []
for x0 in x:
    for x1 in x:
        res.append(fun(np.array([[x0, x1]])))
res = np.array(res)
res = res.reshape((50, 50)).T
X, Y = np.meshgrid(x, x)
fig = plt.figure(figsize=(15, 10))
ax = fig.add_subplot(projection="3d")
surf = ax.plot_surface(
    X, Y, res, cmap=cm.viridis, linewidth=0, antialiased=False, alpha=0.5
)


plt.title("Rastrigin function")
title="Rastrigin function"
plt.xlabel("x1")
plt.ylabel("x2")
plt.legend()
save_path = os.path.join(time_based_dir, f"{title}.png")
plt.savefig(save_path)
plt.close()


# 等高線を描画
plt.contour(  X, Y, res, levels=30, cmap='viridis')

# カラーバーの追加
plt.colorbar()
title='Rastrigin_contour.png'
save_path = os.path.join(time_based_dir, f"{title}.png")
plt.savefig(save_path)
# グラフの表示

plt.close()



"""# 6. RBF Model
Here we implement the RBF model.
"""


########### The RBF model

t = RBF(print_prediction=True, poly_degree=0)

t.set_training_values(xt, yt[:, 0])

t.train()

# Prediction of the validation points
y = t.predict_values(xtest)
print("RBF,  err: " + str(compute_rms_error(t, xtest, ytest)))
# Plot prediction/true values
print('input',xt.shape)
print('ouput',yt[:,0].shape)
print('nt訓練の変数の数？',t.nt)
print('num',t.num)
print('nx:入力の次元数',t.nx)
print('ny：出力の次元数',t.ny)
print('',t.options)
print('',t.mtx.shape)
fig = plt.figure()
plt.plot(ytest, ytest, "-", label="$y_{true}$")
plt.plot(ytest, y, "r.", label=r"$\hat{y}$")

plt.xlabel("$y_{true}$")
plt.ylabel(r"$\hat{y}$")

plt.legend(loc="upper left")
plt.title("RBF model: validation of the prediction model")
title="RBF model: validation of the prediction model"
save_path = os.path.join(time_based_dir, f"{title}.png")
plt.savefig(save_path)

plt.close()
