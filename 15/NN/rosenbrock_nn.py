import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
from torchinfo import summary


def Rosenbrock(x, n):
    value = 0
    for i in range(10):
        value += 100 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2
    return value
def dixon_price(x,n):
    n = len(x)
    term1 = (x[0] - 1) ** 2
    term2 = sum([i * (2 * x[i]**2 - x[i-1])**2 for i in range(11, 20)])
    return term1 + term2
def Rosenbrock1(x, n):
    value = 0
    for i in range(21,30):
        value += 100 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2
    return value
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
    n_rosenbrock = 10
    n_dixon=10
    n_powell=32
    rosen_value = Rosenbrock(x, n_rosenbrock)
    dixon_value = dixon_price(x,n_dixon)
    rosen_value2 = Rosenbrock1(x, n_rosenbrock)
    
    # powell_value= powell(x[n_rosenbrock+n_dixon:n_rosenbrock+n_dixon+n_powell],n_powell)
    return rosen_value + dixon_value+rosen_value2

# パラメータの設定
dim = 31
max_gen = 100
pop_size = 300
offspring_size = 2
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
            nn.Linear(31, 10),
            nn.Linear(10, 5),
            nn.Linear(5,3),
            nn.Linear(3,1),
          
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
def convert_bool_to_int(bool_list):
    train=[]
    for x in bool_list:
        if x:
            train.append(1)
        else: 
            train.append(0)
    return train


# #枝刈りを行うコード
# def param_weights(t):
#     param_weight = []
#     key_list=[]
#     param_list=[]
    
#     for key,param in model.state_dict().items():
#         key_list.append(key)
#         param_list.append(model.state_dict()[key].cpu().numpy())
#     j=0
#     print(key_list)
#     """
#     for i in range(len(key_list)):
#         for row in param_list[i]:
#             j += 1
#             for element in row:
#                 param_weight.append(element)
#     param_weight = torch.tensor(param_weight)
#     print('更新前',param_weight)
    
#     """
#     param_weight1=model.state_dict()['linear_relu_stack.0.weight']
#     train_mask=[]
#     train_mask=param_weight1 < torch.quantile(param_weight1,0.9-t*0.004)
#     train_mask = train_mask.to(torch.int64)
#     print(train_mask)
#     param_weight1 = train_mask * param_weight1
    
#     param_weight2=model.state_dict()['linear_relu_stack.1.weight']
#     train_mask=[]
#     train_mask=param_weight2 < torch.quantile(param_weight2,0.9-t*0.004)
#     print(0.9-t*0.01)
#     train_mask = train_mask.to(torch.int64)
#     param_weight2 = train_mask * param_weight2


#     param_weight3=model.state_dict()['linear_relu_stack.2.weight']
#     train_mask=[]
#     train_mask=param_weight3 < torch.quantile(param_weight3,0.9-t*0.004)
#     train_mask = train_mask.to(torch.int64)
#     param_weight3 = train_mask * param_weight3

#     param_weight4=model.state_dict()['linear_relu_stack.3.weight']
#     train_mask=[]
#     train_mask=param_weight4 < torch.quantile(param_weight4,0.9-t*0.004)
#     train_mask = train_mask.to(torch.int64)
#     param_weight4 = train_mask * param_weight4
#     return param_weight1,param_weight2,param_weight3,param_weight4

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loss = train_loop(x_data, y_data, model, loss_fn, optimizer)
    test_loss = test_loop(test_x_data, test_y_data, model, loss_fn)
    train_losses.append(train_loss)
    test_losses.append(test_loss)
    # param_weight1,param_weight2,param_weight3,param_weight4 = param_weights(t)
    # #param_weight = nn.Parameter(param_weight)
    # model.state_dict()['linear_relu_stack.0.weight'][0:10] = param_weight1
    # model.state_dict()['linear_relu_stack.1.weight'][0:5] = param_weight2
    # model.state_dict()['linear_relu_stack.2.weight'][0:3] = param_weight3
    # model.state_dict()['linear_relu_stack.3.weight'][0:1] = param_weight4


    print("Done!")

# lossをプロットするためにnumpy配列に変換
plt.plot(train_losses, label='Training Loss')
plt.plot(test_losses, label='Testing Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('neural_network_loss.png')




def weight():
    key_list=[]
    param_list=[]
    param_bias=[]
    param_weight=[]
    for key,param in model.state_dict().items():
        key_list.append(key)
        param_list.append(model.state_dict()[key].cpu().numpy())
    j=0
    z=0
    temp_down=0
    temp_up=0
    row_indices=[]
    col_indices=[]
    for i in range(0,len(key_list)-5,2):
        if i == 2:
            temp_down = j +10
        temp_down=j+temp_down
        j=0
        temp_up=z+temp_up
        
        for row in param_list[i]:
            j += 1
         
            z=0
            for element in row:
                z+=1
                if element != 0.0:
                    
                    if i == 0:
                        row_indices.append(z)
                        col_indices.append(j+10)
                    else:
                        print('---------')
                        row_indices.append(z+temp_up)
                        col_indices.append(j+temp_down)
                    param_weight.append(element)
            

            

    
    for i in range(1,len(key_list)-3,2):
       
       print('----------------------------')
       print(param_list[i])
       for element in param_list[i]:
           param_weight.append(element)
            
    param_weight=torch.tensor(param_weight)
    param_weight=param_weight.view(375,1)
    
    # param_weight=param_weight.squeeze()

    # input=torch.empty(18,1)
    # temp_x_data = torch.ones_like(input)


    # param_bias = torch.cat((temp_x_data,param_bias_reshape),0)
    print(param_weight)
   
    num=0
    j=0
    #今回のデータセットにのみバイアスのノードを書き加える。
    for i in range(50):
        num+=1
        row_indices.append(260)
  
    for i in range(30):
        num+=1
        row_indices.append(261)
    print(num)
    for i in range(10):
        num+=1
        row_indices.append(262)

   
    for i in range(150,200):
        j+=1
        col_indices.append(i)
    
    for i in range(201,231):
        j+=1
        col_indices.append(i)
    print(j)
    for i in range(232,242):
        j+=1
        col_indices.append(i)
    # print(num,j)
    row_indices = torch.tensor(row_indices)
    col_indices = torch.tensor(col_indices)
    # print(row_indices)#エッジの上の部分の情報
    # print(col_indices)#エッジの下の部分の情報
    return param_weight,row_indices,col_indices
# weight()

def correct_data():
    return x_data.T

param_weight = []
key_list=[]
param_list=[]
for key,param in model.state_dict().items():
    key_list.append(key)
    param_list.append(model.state_dict()[key].cpu().numpy())
j=0
print(key_list)
print(param_list)
print(param_list[0])
for i in range(0,7,2):
  
    argmax_index = np.argmax(param_list[i],axis=0)
    print(argmax_index)
column_means = np.mean(param_list[0], axis=1)

print("各列の平均:", column_means)