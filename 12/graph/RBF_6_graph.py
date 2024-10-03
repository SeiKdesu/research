import os 
import numpy as np
import torch 
from torch_geometric.data import Data

import torch.nn.functional as F
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import to_networkx
import matplotlib.pyplot as plt
import networkx as nx

def visualize_graph(G, color):
    plt.figure(figsize=(3, 3))
    plt.xticks([])
    plt.yticks([])

    # ノードの範囲ごとに縦1列に配置するための位置を設定
    pos = {}

    # 各範囲ごとにノードを縦1列に並べる
    ranges = [[0, 1,2, 3, 4, 5],[ 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66], [67]]
    x_offset = 0  # X軸のオフセット

    # ノードを正しく配置するためにループを修正
    for r in ranges:
        for i, node in enumerate(r):
            pos[node] = (x_offset, -i)  # Y座標は負の値に設定
        x_offset += 1  # 次の列に移動

    # エッジの重みに基づいて太さを決定
    weights = nx.get_edge_attributes(G, 'weight')
    default_width = 1.0
    edge_widths = [weights[edge] if edge in weights else default_width for edge in G.edges()]

    # グラフを描画
    plt.figure(figsize=(8, 8))
    nx.draw_networkx_nodes(G, pos, node_size=700, node_color=color, cmap=plt.cm.rainbow)
    nx.draw_networkx_labels(G, pos)
    nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color='gray', arrows=True)

    # 画像を保存
    plt.savefig('rastrigin_RBF_teacher.png')



# edge_index=torch.tensor([
#     [0,2],[0,3],[0,4],[0,5],[0,6],[0,7],[0,8],[0,9],    [0,10],[0,11],[0,12],[0,13],[0,14],[0,15],[0,16],[0,17],    [0,18],[0,19],[0,20],[0,21],[0,22],
#         [1,2],[1,3],[1,4],[1,5],[1,6],[1,7],[1,8],[1,9],    [1,10],[1,11],[1,12],[1,13],[1,14],[1,15],[1,16],[1,17],    [1,18],[1,19],[1,20],[1,21],[1,22],
#     [2,23],[3,23],[4,23],[5,23],[6,23],[7,23],[8,23],[9,23],    [10,23],[11,23],[12,23],[13,23],[14,23],[15,23],[16,23],[17,23],    [18,23],[19,23],[20,23],[21,23],[22,23]
    
# ],dtype=torch.long)
src =[]
dst=[]
for j in range(6):
    for i in range(61):
        src.append(j)
for i in range(6):
    for i in range(6,67):
        dst.append(i)
for i in range(6,67):
    src.append(i)
for i in range(6,67):
    dst.append(67)
edge_index=torch.tensor([src,dst],dtype=torch.long)
edge_index=edge_index.squeeze(1)
print(edge_index.shape)
tmp=[]
for i in range(366):
    tmp.append(1)
tmp = torch.tensor(tmp)
params=[486.80922995 ,-1852.68957654  ,  26.42773745  ,2565.72875544,
   -91.25139632 ,  117.08585445  , 437.57281284  , 416.93700268,
  -502.98358182, -1031.37170104 , -563.7559948   , 595.24634664,
  1140.38984581 ,-1553.23786605  , 390.62142444  ,-245.25717727,
   321.50326016 , -658.64590383 , -826.4091008 ,    16.74340164,
  -109.14604615 ,-1044.7661057,   -135.60374978 , 1608.29276161,
    19.48196195 ,   52.4210256 , -1492.57338516 , -383.26557379,
  2288.17658221 , -556.49193532  , 847.76618402 , -620.53858155,
  1135.66912791 ,-1048.48832343 , -865.66292076 ,  -82.53190988,
 -2159.52163985  ,1312.20463682  ,-846.61852455  ,-447.36244816,
   116.37481124  , 195.31740237  ,1657.99670289 ,  -31.68716604,
  -242.80708267 , -647.86086355, -1089.83769434 ,-1204.90159241,
   567.99514157  , 500.26994407 ,  618.83182767  , 424.8319369,
  2091.30396422  , 202.53166032  ,-935.29968311 , 1534.02675445,
 -1069.84108288  ,-717.83892037 , -848.13029293 , 2217.81972353,
  2309.64631648]
params=torch.tensor(params)
params=torch.cat((tmp,params))
edge_attr=params
np.random.seed(1234)
x=np.random.rand(68,2)
#x=np.zeros_like(a)
x=torch.tensor(x,dtype=torch.float)
#x=torch.tensor([[0],[0],[0],[0],[0],[1],[1],[1],[1],[1],[2],[2],[2],[2],[2],[2],[2],[2],[2],[2],[2],[2],[2],[2],[2],[2],[2],[2],[3],[3],[3]],dtype=torch.float)
y_tmp=[]
for i in range(62):
    y_tmp.append(1)
y_tmp = torch.tensor(y_tmp)
y=torch.tensor([0,1,2,3,4,5])
y= torch.cat((y,y_tmp))
print(edge_index)
print(y.shape)
dataset=Data(x=x,edge_index=edge_index,edge_attr=edge_attr,y=y,num_classes=7)
Data.train_mask=np.array([1 for i in range(len(y))])

G=to_networkx(dataset, to_undirected=False)
visualize_graph(G,color=dataset.y)

print(dataset)
print('==============================================================')


from torch_geometric.nn.conv.gcn_conv import GCNConv

class Net(torch.nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        hidden_size = 5
        print(dataset.num_classes)
        self.conv1=GCNConv(dataset.num_node_features,hidden_size)
        self.conv2=GCNConv(hidden_size,hidden_size)

        self.linear=torch.nn.Linear(hidden_size,dataset.num_classes)

    def forward(self, data):
        if not hasattr(data, 'edge_index'):
            raise ValueError("edge_index is missing from the dataset.")
    
        x = data.x
        edge_index = data.edge_index  # edge_index をデータセットから取得する
        assert edge_index.dtype == torch.long  # データ型の確認

        # GCN 層の計算
        x = self.conv1(x, edge_index)
        x = F.relu(x)

        x = self.conv2(x, edge_index)
        x = F.relu(x)
        
        # 出力層の計算
        x = self.linear(x)
        
        return x


# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device =torch.device('cpu')
model =Net()
model.to(device)
model.train()

optimizer=torch.optim.Adam(model.parameters(),lr=0.01)

loss_func = torch.nn.CrossEntropyLoss()

for epoch in range(100):
    optimizer.zero_grad()
    dataset.to(device)
    out = model(dataset)
    loss = loss_func(out,dataset.y)

    loss.backward()

    optimizer.step()

    print('Epoch %d | Loss: %.4f' % (epoch,loss.item()))
    
    model.eval()

    _,pred = model(dataset).max(dim=1)
    predict=pred.cpu()
    print("結果：",predict)
predict=pred.cpu()
data_y=dataset.y.cpu()
count=0
for i in range(len(predict)): 
    if predict[i]==data_y[i]:
        count += 1
# visualize_graph(G,color=predict)
print(count/len(data_y))
print("結果：",predict)
print("真値：",data_y)
