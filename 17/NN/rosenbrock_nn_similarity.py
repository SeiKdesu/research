import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
from torchinfo import summary


def Rosenbrock(x, n):
    value = 0
    for i in range(n-1):
        value += 100 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2
    return value
def dixon_price(x,n):
    n = len(x)
    term1 = (x[0] - 1) ** 2
    term2 = sum([i * (2 * x[i]**2 - x[i-1])**2 for i in range(11, 20)])
    return term1 + term2
def Rosenbrock1(x, n):
    value = 0
    for i in range(3,5):
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
    n_rosenbrock = 3
    # n_dixon=10
    # n_powell=32
    rosen_value = Rosenbrock(x, n_rosenbrock)
    # dixon_value = dixon_price(x,n_dixon)
    rosen_value2 = Rosenbrock1(x, n_rosenbrock)
    
    # powell_value= powell(x[n_rosenbrock+n_dixon:n_rosenbrock+n_dixon+n_powell],n_powell)
    return rosen_value +rosen_value2

# パラメータの設定
dim = 6
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
            nn.Linear(6, 12),
            nn.Linear(12,10),
            nn.Linear(10,5),
            nn.Linear(5, 2),
            nn.Linear(2,1),
        
          
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
            

            

    
    for i in range(1,len(key_list),2):
       
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
from icecream import ic
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
result = []
















from torch_geometric.data import Data
for dim_num in range(dim):
    num_nodes =0
    # 各ノードに対して接続先を追加
    node = dim_num
        # srcとdstを格納するリスト
    src = []
    dst = []
    ex_dst = []
    edge_attr=[]
    tmp =0 
    for i in range(0,8,2):
        # 結果を保存するリスト
        largest_indices = []      # 最大値のインデックス
        second_largest_indices = []  # 2番目に大きい値のインデックス
        float_largest = []
        float_seconde_large = []
        # 各列に対して計算
        for col in range(param_list[i].shape[1]):
            # 各列の値を取得
            column = param_list[i][:, col]
            sorted_indices = np.argsort(column)[::-1]  # 降順ソートのインデックス
            largest_indices.append(sorted_indices[0])  # 最大値のインデックス
            second_largest_indices.append(sorted_indices[1])  # 2番目に大きい値のインデックス

            float_sorted = np.sort(column)[::-1]
            float_largest.append(float_sorted[0])
            float_seconde_large.append(float_sorted[1])

        # 結果を表示
        ic(largest_indices)
        ic(second_largest_indices)

        
        print("各列で最大値のインデックス:", largest_indices)
        print("各列で2番目に大きい値のインデックス:", second_largest_indices)


        if tmp == 0:
            num_nodes += len(largest_indices)
            # 最大値のインデックスと2番目に大きい値のインデックスを使って接続を作成
            src.append(node)
            dst.append(largest_indices[node] + num_nodes )  # ノード番号を5足して接続
            ex_dst.append(largest_indices[node] )
            src.append(node)
            dst.append(second_largest_indices[node]+num_nodes)  # 同様に接続
            ex_dst.append(second_largest_indices[node] )
            edge_attr.append(float_largest[node])
            edge_attr.append(float_seconde_large[node])
            tmp=1
        else:
            ic(tmp)
            ic(ex_dst)
            for d in ex_dst:
                src.append(d+num_nodes)
                src.append(d+num_nodes)
            ex_dst1=[]
            num_nodes += len(largest_indices)
            for d in ex_dst:


                    # 最初の接続: 最大値のインデックス
                # src.append(node + 5)  # ノード番号を5足して接続

                print(d)
                dst.append(largest_indices[d] + num_nodes)  # 最大値に5を足して接続
                edge_attr.append(float_largest[largest_indices[d]])
                edge_attr.append(float_seconde_large[second_largest_indices[d]])
                # 2番目の接続: 2番目に大きい値のインデックス
                # src.append(node + 5)  # ノード番号を5足して接続
                dst.append(second_largest_indices[d] + num_nodes)  # 2番目に大きい値に5を足して接続
                ex_dst1.append(largest_indices[d] )
                ex_dst1.append(second_largest_indices[d] )
            tmp+=1
            ex_dst = ex_dst1
        # 結果を表示
        print("src:", src)
        print("dst:", dst)
        print("------------------------------------------------")
    column_means = np.mean(param_list[0], axis=0)

    print("各列の平均:", column_means)
    # 結果を表示

    ic(result)
    src = np.array(src)
    ic(src.shape)
    edge_attr = np.array(edge_attr)
    ic(edge_attr)


    unit_vectors = [[1 if i == j else 0 for i in range(num_nodes+2)] for j in range(num_nodes+2)]


    x = torch.tensor(unit_vectors,dtype=torch.float)

    y_tmp=[]
    for i in range(3):
        y_tmp.append(0)
    for i in range(3,6):
        y_tmp.append(1)
    # for i in range(3):
    #     y_tmp.append(1)
    for i in range(num_nodes - dim +2):
        y_tmp.append(2)

    y = torch.tensor(y_tmp)
    edge_index=torch.tensor([src,dst],dtype=torch.long)
    dataset=Data(x=x,edge_index=edge_index,edge_attr=edge_attr,y=y,num_classes=4) 
    # data0=dataset
    # data1 = dataset
    # data2 = dataset
    # data3 = dataset
    # data4  = dataset
    # data5 = dataset
    ic(dataset)


    def visual_graph(dim_num ):
        import torch
        import networkx as nx
        import matplotlib.pyplot as plt

        # テンソルを定義
        edges_tensor = edge_index
        # srcとdstを取り出し
        src = edges_tensor[0].tolist()
        dst = edges_tensor[1].tolist()

        # NetworkXグラフの作成
        G = nx.Graph()

        # エッジ情報の追加
        edges = list(zip(src, dst))
        G.add_edges_from(edges)

        # ノードの配置 (手動で縦に並べる設定)
        pos = {}

        # ノードを区切って縦に配置
        # ノード0~6
        for i in range(6):
            pos[i] = (0, i)  # 横位置0、縦位置i

        # ノード6~13
        for i in range(6, 18):
            pos[i] = (1, i - 6)  # 横位置1、縦位置i-7

        # ノード14~24
        for i in range(18, 28):
            pos[i] = (2, i - 18)  # 横位置2、縦位置i-14

        # ノード24~29
        for i in range(28, 33):
            pos[i] = (3, i - 28)  # 横位置3、縦位置i-24
        for i in range(33,35):
            pos[i] = (4, i -33)

        # グラフ描画
        plt.figure(figsize=(8, 6))  # 描画サイズ
        nx.draw(G, pos, with_labels=True, node_size=700, node_color='skyblue', font_size=12, font_weight='bold', edge_color='gray')

        # 描画
        plt.title("Graph Visualization (Nodes arranged vertically)")
        plt.savefig(f'graph_{dim_num}.png')
        if dim_num == 0:
            data0 = dataset
        elif dim_num == 1:
            print(dataset)
            data1 == dataset
        elif dim_num == 2:
            data2 = dataset
        elif dim_num == 3:
            data3 = dataset
        elif dim_num == 4:
            data4 = dataset
        elif dim_num == 5:
            data5 = dataset
    
    visual_graph(dim_num=dim_num)

def weight_return():
    return result
from torch_geometric.nn import GCNConv
# GCNモデルを定義

from torch_geometric.nn import GCNConv, global_mean_pool

import torch
from torch_geometric.nn import GCNConv, global_mean_pool

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
    
    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return global_mean_pool(x, batch)
class SiameseGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.gnn = GCN(in_channels, hidden_channels, out_channels)
        self.fc = torch.nn.Linear(out_channels * 2, 1)  # 埋め込み結合用

    def forward(self, data1, data2):
        # グラフ1とグラフ2の埋め込み
        embed1 = self.gnn(data1.x, data1.edge_index, data1.batch)
        embed2 = self.gnn(data2.x, data2.edge_index, data2.batch)
        
        # 埋め込みを結合して類似スコアを出力
        combined = torch.cat([embed1, embed2], dim=1)
        return torch.sigmoid(self.fc(combined))
    
# 類似スコアを計算
model = SiameseGNN(in_channels=35, hidden_channels=32, out_channels=16)
# データ (例: data1, data2) はPyTorch Geometric形式
output = model(data0, data1)
ic(data0,data1,data2,data3,data4,data5)
ic(model(data0,data1))
ic(model(data0,data2))
ic(model(data0,data3))
ic(model(data0,data4))
ic(model(data0,data5))
ic(model(data1,data2))
ic(model(data1,data3))
ic(model(data1,data4))
ic(model(data1,data5))
ic(model(data2,data3))
ic(model(data2,data4))
ic(model(data2,data5))
ic(model(data3,data4))
ic(model(data3,data5))
ic(model(data4,data5))
print(output)
print(edge_index.max(), num_nodes)

# x1 = model(data0.x,data0.edge_index)
# x2 = model(data1.x,data1.edge_index)
# similartiy = torch.cosine_similarity(x1,x2)
# ic(similartiy)