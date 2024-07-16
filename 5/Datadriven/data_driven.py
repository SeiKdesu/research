import numpy as np
import torch
from torch import nn

# Rastrigin関数の定義
def OriginalRastrigin(x, n):
    value = 0
    for i in range(n):
        value += x[i]**2 - 10 * np.cos(2 * np.pi * x[i])
    value += 10 * n
    return value

# パラメータの設定
dim = 10
max_gen = 100
pop_size = 300
offspring_size = 200
bound = 5.12

# 初期集団の生成
def init_population(pop_size, dim, bound):
    return [np.random.uniform(-bound, bound, dim) for _ in range(pop_size)]

# 適合度の計算
def evaluate_population(population):
    return [OriginalRastrigin(individual, dim) for individual in population]

def genetic_algorithm(dim, max_gen, pop_size, offspring_size, bound):
    population = init_population(pop_size, dim, bound)
    for generation in range(max_gen):
        fitness = evaluate_population(population)
    return population, fitness

population, fitness = genetic_algorithm(dim, max_gen, pop_size, offspring_size, bound)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

np_array = np.array(population, dtype=np.float32)
#print(np_array.shape)
x_data = torch.from_numpy(np_array).to(device)
#print(x_data)

np_array1 = np.array(fitness, dtype=np.float32)
#print(np_array1.shape)
y_data = torch.from_numpy(np_array1).unsqueeze(1).to(device)
#print(y_data)


test_population, test_fitness = genetic_algorithm(dim, max_gen, pop_size, offspring_size, bound)


test_np_array = np.array(test_population, dtype=np.float32)
#print(test_np_array.shape)
test_x_data = torch.from_numpy(test_np_array).to(device)
##print(test_x_data)

test_np_array1 = np.array(test_fitness, dtype=np.float32)
#print(test_np_array1.shape)
test_y_data = torch.from_numpy(test_np_array1).unsqueeze(1).to(device)
#print(test_y_data)


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten(start_dim=1)
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(10, 8),
            nn.Linear(8, 6),
            nn.Linear(6,4),
            nn.Linear(4,2),
            nn.Linear(2, 1),
      
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)

all_loss=[]
# 学習
learning_rate = 0.00001
batch_size = 60
epochs = 100
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

def train_loop(x_data, y_data, model, loss_fn, optimizer):
    size = len(x_data)
    model.train()
    for batch in range(0, size, batch_size):
        X = x_data[batch:batch+batch_size]
        y = y_data[batch:batch+batch_size]
        pred = model(X)
        loss = loss_fn(pred, y)
        all_loss.append(loss)
        #print('y',y)
        #print('pred',pred)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss_val = loss.item()
            print(f"loss: {loss_val:>7f}  [{batch:>5d}/{size:>5d}]")
            #print('y',y)
            #print('pred',pred)

def test_loop(test_x_data, test_y_data, model, loss_fn):
    size = len(test_x_data)
    model.eval()
    test_loss = 0

    with torch.no_grad():
        pred = model(test_x_data)
        test_loss = loss_fn(pred, test_y_data).item()

    print(f"Test Error: \n Avg loss: {test_loss:>8f} \n")

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(x_data, y_data, model, loss_fn, optimizer)
    test_loop(test_x_data, test_y_data, model, loss_fn)
print("Done!")

import matplotlib.pyplot as plt
# lossをプロットするためにnumpy配列に変換
#loss_graph = all_loss.cpu().detach().numpy()
loss_graph = all_loss
plt.plot(loss_graph)
plt.savefig('loss.png')
