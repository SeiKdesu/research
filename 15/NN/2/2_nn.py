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
def dixon_price(x):
    n = len(x)
    term1 = (x[0] - 1) ** 2
    term2 = sum([i * (2 * x[i]**2 - x[i-1])**2 for i in range(1, n)])
    return term1 + term2

def powell(x):
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
def rastrigin(x, n):
    A = 10
    return A * n + sum([(x[i]**2 - A * np.cos(2 * np.pi * x[i])) for i in range(n)])
def objective_function(x,dim):
    return rastrigin(x,dim)
    # n_rosenbrock = 3
    # n_dixon=3
    # n_powell=4
    # rosen_value = Rosenbrock(x[:n_rosenbrock], n_rosenbrock)
    # dixon_value = dixon_price(x[n_rosenbrock:n_rosenbrock+n_dixon])
    # powell_value= powell(x[n_rosenbrock+n_dixon:n_rosenbrock+n_dixon+n_powell])
    # return rosen_value + dixon_value+ powell_value

# パラメータの設定
dim = 2
max_gen = 100
pop_size = 100
offspring_size = 200
bound = 5
from datetime import datetime

# 現在の時刻を取得
current_time = datetime.now()
name = f'{current_time}_{pop_size}'



# 初期集団の生成
def init_population(pop_size, dim, bound):
    return [np.random.uniform(-bound, bound, dim) for _ in range(pop_size)]

# 適合度の計算
def evaluate_population(population):
    return [objective_function(individual, dim) for individual in population]

def genetic_algorithm(dim, max_gen, pop_size, offspring_size, bound):
    population = init_population(pop_size, dim, bound)
    # for generation in range(max_gen):
    fitness = evaluate_population(population)
    
    return population, fitness

population, fitness = genetic_algorithm(dim, max_gen, pop_size, offspring_size, bound)


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

np_array = np.array(population, dtype=np.float32)
x_data = torch.from_numpy(np_array).to(device)

np_array1 = np.array(fitness, dtype=np.float32)
y_data = torch.from_numpy(np_array1).unsqueeze(1).to(device)

test_population, test_fitness = genetic_algorithm(dim, max_gen, pop_size, offspring_size, bound)

test_np_array = np.array(test_population, dtype=np.float32)
test_x_data = torch.from_numpy(test_np_array).to(device)

test_np_array1 = np.array(test_fitness, dtype=np.float32)
test_y_data = torch.from_numpy(test_np_array1).unsqueeze(1).to(device)
print(x_data.shape)
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten(start_dim=1)
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(2, 3),
            nn.Tanh(),
            nn.Linear(3, 5),
            nn.Tanh(),
            nn.Linear(5,3),
            nn.Sigmoid(),
            nn.Tanh(3,1),
          
        )


    def forward(self, x):
       
        x = self.flatten(x)
        
        logits = self.linear_relu_stack(x)
        return logits
   

model = NeuralNetwork().to(device)

all_loss = []
# 学習
learning_rate = 0.01
batch_size = 60
epochs = 80
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

data_x_fatures=[]
def train_loop(x_data, y_data, model, loss_fn, optimizer):
    
    size = len(x_data)
    
    model.train()
    epoch_loss = 0
    for batch in range(0, size, batch_size):
        X = x_data[batch:batch+batch_size]
        y = y_data[batch:batch+batch_size]
        
        pred = model(X)
        loss = loss_fn(pred, y)
        epoch_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return epoch_loss / size

def test_loop(test_x_data, test_y_data, model, loss_fn):
    model.eval()
    with torch.no_grad():
        pred = model(test_x_data)
        test_loss = loss_fn(pred, test_y_data).item()
    print(f"Test Error: \n Avg loss: {test_loss:>8f} \n")
    return test_loss

train_losses = []
test_losses = []
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loss = train_loop(x_data, y_data, model, loss_fn, optimizer)
    test_loss = test_loop(test_x_data, test_y_data, model, loss_fn)
    train_losses.append(train_loss)
    test_losses.append(test_loss)
import numpy as np
from scipy.interpolate import Rbf
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


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

output=0
# function_list = ['multiquadric', 'inverse', 'gaussian', 'linear', 'cubic', 'quintic', 'thin_plate']
function_list = ['gaussian']
for i, function in enumerate(function_list):
    # 3D表示用にX, Y軸を拡張
    x_range = np.linspace(-5, 5, 100)
    y_range = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x_range, y_range)
    # Rosenbrock関数の計算 (n=2の場合)
    Z = np.zeros(X.shape)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = objective_function([X[i, j], Y[i, j]], dim)
    # 3D表示
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 元の関数の表示
    ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.6)


    x = np.vstack([X.ravel(), Y.ravel()]).T
    x =torch.tensor(x,dtype=torch.float32).to(device)
    print(x.shape)
    Z_interp = model(x)

    Z_interp = Z_interp.view(100, 100).cpu().detach().numpy()


    # 補間関数のプロット
    ax.plot_wireframe(X, Y, Z_interp, color='green', label=f'{function} RBF', linewidth=0.5)

    # 3Dの設定
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.set_title('3D Interpolation with RBF')
    plt.legend(loc='upper right')

    # 画像を保存
    plt.savefig(f'acc_loss/{name}_rbf_fig.png')  # 保存
    plt.close()
error = Z - Z_interp
mse = np.mean(error**2)

# 結果を表示
print(f"Mean Squared Error (MSE) between Z and Z_interp: {mse}")



# print('ここまで')
# 重みと基底関数の中心点を出力






# 元の関数 Z の等高線表示
plt.figure(figsize=(10, 8))
plt.contour(X, Y, Z, levels=30, cmap='viridis')
plt.title("Contour Plot of Original Function Z")
plt.colorbar()
plt.savefig(f'acc_loss/{name}_rbf_original_contour.png')  # 保存
plt.close()
# 補間された関数 Z_interp の等高線表示
plt.figure(figsize=(10, 8))
plt.contour(X, Y, Z_interp, levels=30, cmap='viridis')
plt.title("Contour Plot of Interpolated Function Z_interp")
plt.colorbar()
plt.savefig(f'acc_loss/{name}_rbf_predict_contour.png')  # 保存