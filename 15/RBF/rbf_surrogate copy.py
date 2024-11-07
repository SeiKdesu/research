import numpy as np
from smt.surrogate_models import RBF
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from smt.sampling_methods import LHS
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
    return np.array(values).reshape(-1,1)



random_state = 25 # fixes the seed for consistent results

# Bounds
xlower = -5.0
xupper = 5.0
import warnings

warnings.filterwarnings("ignore")

from datetime import datetime
import os 
# 現在の時刻を取得
current_time = datetime.now()
name = f'{current_time}'
# 保存するディレクトリを作成
if not os.path.exists(name):
    os.makedirs(name)
def dirs():
    return name
def make_file_path():
    return file_path
# ファイルのパスを指定
file_path = os.path.join(name, "population.txt")
file_path2 = os.path.join(name,"weight.txt")
file_path3 = os.path.join(name,"matrix.txt")


# Number of training and testing points
num_training_pts = 5
num_testing_pts = 100
ndim=6

ndoe_test=200

loss_history=[]

ndoe=20

# Generating training data
sampling = LHS(xlimits=np.array([[-5.0,5.0]]*ndim), criterion="ese", random_state=1)
x_train = sampling(ndoe)
y_train = objective_function(x_train)
with open(file_path, "a") as file:
    file.write(f"{x_train}\n")

# Generating testing data
sampling = LHS(xlimits=np.array([[-5.0,5.0]]*ndim), criterion="ese", random_state=1)
x_test = sampling(ndoe_test)
y_test = objective_function(x_test)
# Fitting the RBF
sm = RBF(d0=6, poly_degree=-1, print_global=False)
sm.set_training_values(x_train, y_train)
sm.train()
with open(file_path2, "a") as file:
    file.write(f"{sm.mtx}\n")
with open(file_path3, "a") as file:
    file.write(f"{sm.sol[:,0]}\n")
print("Psi matrix: \n{}".format(sm.mtx))
print("\nWeights: {}".format(sm.sol[:,0]))

def get_xt():
    return x_train

def xt_all():
    return x_train
def get_yt():
    return y_train

def matrix():
    return sm.mtx.tolist()
def weight():

    return sm.sol



# Predict at test values
y_test_pred = sm.predict_values(x_test)
print("\nMSE: {}".format(mean_squared_error(y_test, y_test_pred)))
loss_history.append(mean_squared_error(y_test,y_test_pred))
# Plotting
# plt.plot(x_train, y_train, 'ro', label="Training data")
# plt.plot(x_test, y_test, 'b--', label="True function")
# plt.plot(x_test, y_test_pred, 'k', label="Prediction")
# plt.xlim((xlower, xupper))
# plt.xlabel("x", fontsize=14)
# plt.ylabel("y", fontsize=14)
# plt.legend()
# plt.grid()
# plt.show()
# pops=np.arange(20,100)
# # プロット
# plt.figure(figsize=(8, 6))
# plt.plot(pops, loss_history, marker='o', linestyle='-', color='b', label='pop Loss')
# plt.xlabel('pop')
# plt.ylabel('Loss')
# plt.title('Loss Function Over POP')
# plt.legend()
# plt.grid(True)
# plt.savefig(f"results/pop_and_loss_ga_20,100.pdf")