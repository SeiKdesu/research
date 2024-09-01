import os 
import numpy as np
import torch 
from torch_geometric.data import Data
from rastrigin_and_schwefl_bias_node import weight
import torch.nn.functional as F
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import to_networkx
def visualize_graph(G, color):
    plt.figure(figsize=(3,3))
    plt.xticks([])
    plt.yticks([])
    
    # ノードの範囲ごとに縦1列に配置するための位置を設定
    pos = {}

    # 各範囲ごとにノードを縦1列に並べる
    ranges = [[0,1,2,3,4,5,6,7,8,9,28], [10,11,12,13,14,15,16,17,29], [18,19,20,21,22,23,30], [24,25,26,27]]
    x_offset = 0  # X軸のオフセット

    for r in ranges:
        for i, node in enumerate(r):
            pos[node] = (x_offset, -i)
        x_offset += 1  # 次の列に移動

    # エッジの重みに基づいて太さを決定
      # エッジの重みに基づいて太さを決定
    weights = nx.get_edge_attributes(G, 'weight')
    default_width = 1.0
    edge_widths = [weights[edge] if edge in weights else default_width for edge in G.edges()]

    # グラフを描画
    plt.figure(figsize=(8, 8))
    nx.draw_networkx_nodes(G, pos, node_size=700, node_color=color, cmap=plt.cm.rainbow)
    nx.draw_networkx_labels(G, pos)
    nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color='gray', arrows=True)

    # ノードの順序を指定
    
    #nx.draw_networkx(G, pos=nx.spring_layout(G, k=1.5, seed=13648), with_labels=True,node_color=color, cmap="Set2")
   
    plt.savefig('predict_delete_edge_graph.png')

edge_index=torch.tensor([
    [0,13],[0,14],[0,15],[0,16],[0,17],
    [1,10],[1,11],[1,12],[1,13],[1,14],[1,15],[1,17],
    [2,15],
    [3,16],
    [4,10],[4,11],[4,13],[4,15],
    [5,13],[5,17],
    [6,16],[6,17],
    [7,11],[7,14],
    [8,10],[8,11],[8,13],[8,14],[8,15],[8,16],[8,17],
    [9,10],[9,11],[9,12],[9,14],[9,15],[9,16],[9,17],
    [10,19],[10,20],[10,21],
    [11,19],[11,20],[11,21],
    [12,22],[12,23],
    [13,18],[13,22],[13,23],
    [14,18],[14,22],[14,23],
    [15,19],[15,20],[15,21],
    [16,18],[16,19],[16,20],[16,21],[16,22],[16,23],
    [17,19],[17,20],[17,21],
    [18,24],[18,25],[18,26],[18,27],
    [19,24],[19,25],[19,26],[19,27],
    [20,24],[20,25],[20,26],[20,27],
    [21,24],[21,25],[21,26],[21,27],
    [22,24],[22,25],[22,26],[22,27],
    [23,24],[23,25],[23,26],[23,27],
    [28,10],[28,11],[28,12],[28,13],[28,14],[28,15],[28,16],[28,17],
    [29,18],[29,19],[29,20],[29,21],[29,22],[29,23],
    [30,24],[30,25],[30,26],[30,27]
],dtype=torch.long)

params=weight()
print('これがぱらむ',params.shape)
edge_attr=params
np.random.seed(1234)
x=np.random.rand(31,1)
#x=np.zeros_like(a)
x=torch.tensor(x,dtype=torch.float)
#x=torch.tensor([[0],[0],[0],[0],[0],[1],[1],[1],[1],[1],[2],[2],[2],[2],[2],[2],[2],[2],[2],[2],[2],[2],[2],[2],[2],[2],[2],[2],[3],[3],[3]],dtype=torch.float)

y=torch.tensor([2,2,2,2,2,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
#y=torch.tensor([0,0,0,0,0,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,3,3,3])
print('重み',edge_attr)
dataset=Data(x=x,edge_index=edge_index.t(),edge_attr=edge_attr,y=y,num_classes=4)
Data.train_mask=np.array([2 for i in range(len(y))])

G=to_networkx(dataset, to_undirected=False)
#visualize_graph(G,color=dataset.y)

print(dataset)

print('==============================================================')


from torch_geometric.nn.conv.gcn_conv import GCNConv

class Net(torch.nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        hidden_size = 5

        self.conv1=GCNConv(dataset.num_node_features,hidden_size)
        self.conv2=GCNConv(hidden_size,hidden_size)

        self.linear=torch.nn.Linear(hidden_size,dataset.num_classes)

    def forward(self,data):
        x = data.x
       
        edge_index = data.edge_index

        x = self.conv1(x,edge_index)
      
        x=F.relu(x)

        x=self.conv2(x,edge_index)
        x=F.relu(x)
        x=self.linear(x)
        return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model =Net()
model.to(device)
model.train()

optimizer=torch.optim.Adam(model.parameters(),lr=0.01)

loss_func = torch.nn.CrossEntropyLoss()

for epoch in range(200):
    optimizer.zero_grad()
    dataset.to(device)
    out = model(dataset)

    loss = loss_func(out,dataset.y)
    #loss = loss_func(out[dataset.y != 2],dataset.y[dataset.y != 2])
 
    loss.backward()

    optimizer.step()

    print('Epoch %d | Loss: %.4f' % (epoch,loss.item()))

    model.eval()

    _,pred = model(dataset).max(dim=1)
    print(pred.cpu())
predict=pred.cpu()
data_y=dataset.y.cpu()
count=0
for i in range(len(predict)): 
    if predict[i]==data_y[i]:
        count += 1
visualize_graph(G,color=predict)
print(count/len(data_y))
print("結果：",predict)
print("真値：",data_y)
