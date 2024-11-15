import numpy as np
from smt.surrogate_models import RBF
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from smt.sampling_methods import LHS
def Rosenbrock(x,n):
    value = 0
    for i in range(n):
        value += 100 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2
    return value
def Rosenbrock1(x):
    value = 0
    for i in range(3,4):
        value += 100 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2
    
    return value
def dixon_price(x,n):
    # n = len(x)
    term1 = (x[0] - 1) ** 2
    term2 = sum([i * (2 * x[i]**2 - x[i-1])**2 for i in n])
    return term1 + term2

def objective_function(x,dim):
    values=[]
    dim1 = [1,2]
    dim2 = [4,5]
    for tmp in x:
        tmp1 = dixon_price(tmp,dim1)
        tmp2 = dixon_price(tmp,dim2)
        values.append(tmp1+tmp2)
    return np.array(values).reshape(-1,1)

# def objective_function(x,dim):
#     dim1=[1,2]
#     dim2=[4,5]
#     tmp1 = dixon_price(x,dim1)
#     tmp2 = dixon_price(x,dim2)
#     return tmp1+tmp2

def objective_function1(x,dim1,dim2):
    values=[]

    for tmp in x:
        tmp1 = dixon_price(tmp,dim1)
        tmp2 = dixon_price(tmp,dim2)
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
dim=ndim
ndoe_test=200

loss_history=[]

#ここがポイント
ndoe=101

# Generating training data
sampling = LHS(xlimits=np.array([[-5.0,5.0]]*ndim), criterion="ese", random_state=25)
x_train = sampling(ndoe)
y_train = objective_function(x_train,ndim)
# 最終次元にtrain_yを追加
train_data = np.concatenate((x_train,y_train), axis=1)
print(train_data.shape)
# Generating testing data
sampling = LHS(xlimits=np.array([[-5.0,5.0]]*ndim), criterion="ese", random_state=25)
x_test = sampling(ndoe_test)
y_test = objective_function(x_test,ndim)

def data_loader():
    # 1から10までのランダムな整数を10個生成
    random_integers = np.random.randint(0, 9, size=len(train_data))
    return train_data,random_integers