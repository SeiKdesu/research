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

all_fitness = []
all_genom = []

class Individual:
    def __init__(self, genom):
        self.genom = genom
        all_genom.append(genom)
        self.fitness = 0
        self.set_fitness()
    def set_fitness(self):
        self.fitness = OriginalRastrigin()
        all_fitness.append(self.fitness)

    def get_fitness(self):
        return self.fitness
    
def create_generation(POPURATIONS, GENOMS):
    generation = []
    for i in range(POPURATIONS):
        individual = Individual(np.random.rand(0, 2, GENOMS))
        generation.append(individual)
    return generation

np.random.seed(seed=65)

POPURATIONS  = 100
GENOMS = 50

generations = create_generation(POPURATIONS, GENOMS)
print(len(generations))

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

np_array = np.array(all_fitness, dtype=np.float32)
print(np_array.shape)
y_data = torch.from_numpy(np_array).to(device)
print(y_data)
np_array1 = np.array(all_genom, dtype=np.float32)
print(np_array1.shape)
x_data = torch.from_numpy(np_array1).to(device)
print(x_data)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(50, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)

# 学習
learning_rate = 1e-3
batch_size = 64
epochs = 50
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

def train_loop(x_data, y_data, model, loss_fn, optimizer):
    size = len(x_data)
    model.train()
    for batch in range(0, size, batch_size):
        X = x_data[batch:batch+batch_size]
        y = y_data[batch:batch+batch_size].unsqueeze(1)

        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss_val = loss.item()
            print(f"loss: {loss_val:>7f}  [{batch:>5d}/{size:>5d}]")

def test_loop(x_data, y_data, model, loss_fn):
    size = len(x_data)
    model.eval()
    test_loss = 0

    with torch.no_grad():
        pred = model(x_data)
        test_loss = loss_fn(pred, y_data.unsqueeze(1)).item()

    print(f"Test Error: \n Avg loss: {test_loss:>8f} \n")

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(x_data, y_data, model, loss_fn, optimizer)
    test_loop(x_data, y_data, model, loss_fn)
print("Done!")