import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
from torchinfo import summary

# Rastrigin関数の定義
def Rastrigin(x, n):
    value = 0
    for i in range(n):
        value += x[i]**2 - 10 * np.cos(2 * np.pi * x[i])
    value += 10 * n
    return value

# Schwefel関数の定義
def Schwefel(x, n):
    value = 0
    for i in range(n):
        value += x[i] * np.sin(np.sqrt(np.abs(x[i])))
    value = 418.9828873 * n - value
    return value

# オブジェクト関数の定義
def objective_function(x,dim):
    n_rastrigin = dim//2
    n_schwefel = dim//2
    rastrigin_value = Rastrigin(x[:n_rastrigin], n_rastrigin)
    schwefel_value = Schwefel(x[n_rastrigin:], n_schwefel)
    return rastrigin_value + schwefel_value

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
    return [objective_function(individual, dim) for individual in population]

def genetic_algorithm(dim, max_gen, pop_size, offspring_size, bound):
    population = init_population(pop_size, dim, bound)
    for generation in range(max_gen):
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

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
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
#summary(model,input_size=10,verbose=1)

all_loss = []
# 学習
learning_rate = 0.01
batch_size = 60
epochs = 100
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
def convert_bool_to_int(bool_list):
    train=[]
    for x in bool_list:
        if x:
            train.append(1)
        else: 
            train.append(0)
    return train

def param_weights(t):
    param_weight = []
    key_list=[]
    param_list=[]
    
    for key,param in model.state_dict().items():
        key_list.append(key)
        param_list.append(model.state_dict()[key].cpu().numpy())
    j=0
    print(key_list)
    """
    for i in range(len(key_list)):
        for row in param_list[i]:
            j += 1
            for element in row:
                param_weight.append(element)
    param_weight = torch.tensor(param_weight)
    print('更新前',param_weight)
    
    """
    param_weight=model.state_dict()['linear_relu_stack.1.weight']
    train_mask=[]
    train_mask=param_weight < torch.quantile(param_weight,0.4+t*0.001)
    train_mask = train_mask.to(torch.int64)
    param_weight = train_mask * param_weight
    
    print('更新後',param_weight)

    return param_weight

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loss = train_loop(x_data, y_data, model, loss_fn, optimizer)
    test_loss = test_loop(test_x_data, test_y_data, model, loss_fn)
    train_losses.append(train_loss)
    test_losses.append(test_loss)
    param_weight = param_weights(t)
    #param_weight = nn.Parameter(param_weight)
    model.state_dict()['linear_relu_stack.1.weight'][0:8] = param_weight
    print('はな')
    print(model.state_dict()['linear_relu_stack.0.weight'])

    model.linear_relu_stack.weight = param_weight
    print('param_weightです',param_weight)
  

    print("Done!")

# lossをプロットするためにnumpy配列に変換
plt.plot(train_losses, label='Training Loss')
plt.plot(test_losses, label='Testing Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('data_driven_rastrigin+schwful.png')




def weight():
    key_list=[]
    param_list=[]
    param_bias=[]
    param_weight=[]
    for key,param in model.state_dict().items():
        key_list.append(key)
        param_list.append(model.state_dict()[key].cpu().numpy())
    j=0
  
    for i in range(0,len(key_list)-5,2):
       
        for row in param_list[i]:
            j += 1
            for element in row:
                if element < 0.0:
                    element = 0.0
                param_weight.append(element)
            
    print('jjjjjjjjj',j)
    for i in range(0,len(key_list)-3,2):
        print('----------------------------')
        print(param_list[i].shape)
       
        print(param_list[i])
    
    #for i in range(1,len(key_list)-3,2):
     #   print('----------------------------')
      ## print(param_list[i])
        #for element in param_list[i]:
     #
      #      param_weight.append(element)
            
    param_weight=torch.tensor(param_weight)
    param_weight=param_weight.view(152,1)
    param_bias_torch=torch.tensor(param_bias)
    #param_bias_reshape=param_bias_torch.view(18,1)


    #input=torch.empty(1,10)
    #temp_x_data = torch.ones_like(input)


    #param_bias = torch.cat((input_data,param_bias_reshape),0)
    print('weight',param_weight)
    print(param_weight.shape)
 
    #param_weight=param_weight.tolist()
    #param_bias = param_bias.tolist()
    
    #param_weight.append(param_bias)
  
    
    #params = torch.tensor(param_weight)
    return param_weight
#weight()

def correct_data():
    return x_data.T

