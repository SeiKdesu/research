import os 
import numpy as np
import torch 
from torch_geometric.data import Data

import torch.nn.functional as F
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import to_networkx
# from rosenbrock import weight
from torch import optim
# def visualize_graph(G, color):
#     plt.figure(figsize=(3, 3))
#     plt.xticks([])
#     plt.yticks([])

#     # ノードの範囲ごとに縦1列に配置するための位置を設定
#     pos = {}

#     # 各範囲ごとにノードを縦1列に並べる
#     ranges = [
#         list(range(1, 31)) + [54],
#         list(range(32, 42)) + [55],
#         list(range(43, 48)) + [56],
#         list(range(48, 51)) + [57],
#         list(range(52,53))
#     ]
#     x_offset = 0  # X軸のオフセット

#     # すべてのノードの位置を設定
#     for r in ranges:
#         for i, node in enumerate(r):
#             pos[node] = (x_offset, -i)
#         x_offset += 1  # 次の列に移動

#     # 位置が指定されていないノードにも位置を設定
#     for node in G.nodes():
#         if node not in pos:
#             pos[node] = (x_offset, 0)  # 既定の位置を設定

#     # エッジの重みに基づいて太さを決定
#     weights = nx.get_edge_attributes(G, 'weight')
#     default_width = 1.0
#     edge_widths = [weights[edge] if edge in weights else default_width for edge in G.edges()]

#     # グラフを描画
#     plt.figure(figsize=(8, 8))
#     nx.draw_networkx_nodes(G, pos, node_size=700, node_color=color, cmap=plt.cm.rainbow)
#     nx.draw_networkx_labels(G, pos)
#     nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color='gray', arrows=True)

#     plt.savefig('predict_graph_30dim.pdf')
from rbf_surrogate_100_train import *
src=[]
dst=[]
for j in range(6):
    for i in range(101):
        src.append(j)

for i in range(6,107):
    src.append(i)

for j in range(6):
    for i in range(6,107):
        dst.append(i)
for i in range(101):
    dst.append(108)
edge_index=torch.tensor([src,dst],dtype=torch.long)


###############################################################################
tmp =[]
length_matrix = matrix()
one = length_matrix[0]


for i in range(dim):
    tmp.append(one)
    
weight1 = weight()
weight1=weight1.squeeze(1)
weight1 = weight1.tolist()

tmp.append(weight1)


tmp  = [item for sublist in tmp for item in sublist]
params=[]
for item in tmp:
    params.append(item)
params = torch.tensor(params)
edge_attr=params

#########################################################################

# ファイル名を指定
file_name = make_file_path()

# ファイルを読み込み、行ごとにデータを処理
with open(file_name, 'r') as file:
    lines = file.readlines()

# 読み込んだデータをリスト形式に変換
num = get_xt()
num=num[0]
num.reshape(6)
formatted_weight_data = []
formatted_weight_data = xt_all()

# for i in range(dim-1,-1,-1):
#     new_array = np.array([num[i],1,1,1,1,1])
#     formatted_weight_data = np.vstack([new_array,formatted_weight_data])
new_array = np.array([num[0],0,0,0,0,0])
formatted_weight_data = np.vstack([new_array,formatted_weight_data])
new_array = np.array([0,num[1],0,0,0,0])
formatted_weight_data = np.vstack([new_array,formatted_weight_data])
new_array = np.array([0,0,num[2],0,0,0])
formatted_weight_data = np.vstack([new_array,formatted_weight_data])
new_array = np.array([0,0,0,num[3],0,0])
formatted_weight_data = np.vstack([new_array,formatted_weight_data])
new_array = np.array([0,0,0,0,num[4],0])
formatted_weight_data = np.vstack([new_array,formatted_weight_data])
new_array = np.array([0,0,0,0,0,num[5]])
formatted_weight_data = np.vstack([new_array,formatted_weight_data])
    # formatted_weight_data.insert(i, [num[i] , 1,1,1,1,1])
# formatted_weight_data.insert(1, [-2.20  , 1,1,1,1,1])
# formatted_weight_data.insert(2, [0.21596364  , 1,1,1,1,1])
# formatted_weight_data.insert(3, [ 0.70831308    , 1,1,1,1,1])
# formatted_weight_data.insert(4, [ -3.63318154  , 1,1,1,1,1])
# formatted_weight_data.insert(5, [-0.62985134, 1,1,1,1,1])
to_match = np.array([1,1,1,1,1,1])
formatted_weight_data = np.vstack([formatted_weight_data,to_match])


num1 = get_yt()
new_temp= np.array([num1[0][0],1,1,1,1,1])
formatted_weight_data = np.vstack([formatted_weight_data,new_temp])
# formatted_weight_data.append([num1[0], 1,1,1,1,1])#yの値
x=formatted_weight_data

#x=np.zeros_like(a)
x=torch.tensor(x,dtype=torch.float)
#x=torch.tensor([[0],[0],[0],[0],[0],[1],[1],[1],[1],[1],[2],[2],[2],[2],[2],[2],[2],[2],[2],[2],[2],[2],[2],[2],[2],[2],[2],[2],[3],[3],[3]],dtype=torch.float)
y_tmp=[]
for i in range(3):
    y_tmp.append(0)
for i in range(3):
    y_tmp.append(1)
for i in range(102):
    y_tmp.append(2)

y = torch.tensor(y_tmp)



dataset=Data(x=x,edge_index=edge_index,edge_attr=edge_attr,y=y,num_classes=3)
ic(dataset)
# dataset=dataset[0]
G=to_networkx(dataset, to_undirected=False)
# visualize_graph(G,color=dataset.y)

print(dataset)

print('==============================================================')


from torch_geometric.nn.conv.gcn_conv import GCNConv

class Net(torch.nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        hidden_size = 20
       
        self.conv1=GCNConv(dataset.num_node_features,hidden_size)
        self.conv2=GCNConv(hidden_size,25)
       
        self.conv3=GCNConv(25,20)
        self.conv4=GCNConv(20,10)

        self.linear=torch.nn.Linear(10,dataset.num_classes)

    def forward(self,data):
        x = data.x
       
        edge_index = data.edge_index

        x = self.conv1(x,edge_index)
        x=F.relu(x)

        x=self.conv2(x,edge_index)
        x=F.relu(x)
        # x=self.conv7(x,edge_index)
        # x=F.relu(x)
        x=self.conv3(x,edge_index)
        x=F.relu(x)
        x=self.conv4(x,edge_index)
        x=F.relu(x)
       
        
   
        x=self.linear(x)
        return x

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model =Net()
# model.to(device)
# dataset.to(device)
# model.train()

model =Net()
model.to('cpu')
dataset.to('cpu')
model.train()
optimizer=torch.optim.Adam(model.parameters(),lr=0.01)
# scheduler = optim.lr_scheduler.CosineAnnealingLR(
#     optimizer,
#     T_max = 4000
# )
# loss_func = torch.nn.CrossEntropyLoss()
loss_func  = torch.nn.KLDivLoss()
losses=[]
acces=[]
from icecream import ic
from nasi_train import teacher_vector
teacher_probs = teacher_vector().cuda()

for epoch in range(1500):
    optimizer.zero_grad()
    dataset.to('cpu')
    out = model(dataset)
    # out = torch.tensor(out,requires_grad=True)
    # probs = F.log_softmax(out,dim=-1)
    # probs = torch.tensor(probs,dtype=torch.long)
    # loss_per_sample = -torch.sum(teacher_probs * probs,dim=1)
    # loss =torch.mean(loss_per_sample)
    # ic(loss.requires_grad)
    # ic(loss.grad_fn)




    loss = loss_func(out,teacher_probs)
    ic(teacher_vector.cpu())
    print(loss)
    losses.append(loss)
    loss.backward()

    optimizer.step()

    

    model.eval()

    _,pred = model(dataset).max(dim=1)
    # scheduler.step()
 
    predict=pred.cpu()
    ic(predict)
    data_y=dataset.y.cpu()
    count=0
    for i in range(dim): 
        if predict[i]==data_y[i]:
            count += 1
    acces.append(count/len(data_y))
    print('Epoch %d | Loss: %.4f | ACC: %.4f' % (epoch,loss.item(),count/len(data_y)))
print("結果：",predict)
print("真値：",data_y)
# visualize_graph(G,color=predict)
# リスト内の各テンソルをdetachしてnumpy配列に変換
losses_np = [loss.detach().numpy() for loss in losses]
# acces_np=   [acc.detach().numpy() for acc in acces]
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 5))
plt.plot(losses_np, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.title('Loss vs. Epoch')
plt.legend()
plt.grid(True)
plt.savefig('Graph Neural Network')
plt.close()
plt.figure(figsize=(10, 5))
plt.plot(acces, label='Accracy')
plt.xlabel('Epoch')
plt.ylabel('Accracy')
# plt.ylim(0,0.8)
plt.title('Accracy GNN')
plt.legend()
plt.grid(True)
plt.savefig('GNN_ACC')
plt.close()