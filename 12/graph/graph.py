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
        for i, node in enumerate(r[0], r[1] + 1):
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
   
    plt.savefig('rastrigin_schwful_10.png')

edge_index=torch.tensor([
    [0,2],[0,3],[0,4],[0,5],[0,6],[0,7],[0,8],[0,9],    [0,10],[0,11],[0,12],[0,13],[0,14],[0,15],[0,16],[0,17],    [0,18],[0,19],[0,20],[0,21],[0,22]
    [2,23],[3,23],[4,23],[5,23],[6,23],[7,23],[8,23],[9,23],    [10,23],[11,23],[12,23],[13,23],[14,23],[15,23],[16,23],[17,23],    [18,23],[19,23],[20,23],[21,23],[22,23]
    
],dtype=torch.long)

params=torch.tensor([  310.58538755,   -43.00581245 , -606.15205714,  -352.73591899,
   -80.42808353 , -558.22728729 , -110.32097476 , -503.16140508,
  -503.85149654  ,  61.96488896  , 484.05027297 ,  -63.29922403,
  -153.53891832 , -459.2063449 ,   505.73459888,  209.66500692,
   906.46603592 ,-1208.64864154,  1609.57523274,   554.53474061,
   598.495502  ])
print(params)
edge_attr=params
np.random.seed(1234)
x=np.random.rand(2,1)
#x=np.zeros_like(a)
x=torch.tensor(x,dtype=torch.float)
#x=torch.tensor([[0],[0],[0],[0],[0],[1],[1],[1],[1],[1],[2],[2],[2],[2],[2],[2],[2],[2],[2],[2],[2],[2],[2],[2],[2],[2],[2],[2],[3],[3],[3]],dtype=torch.float)
print(x)
y=torch.tensor([0,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2])
print(edge_index.shape)

dataset=Data(x=x,edge_index=edge_index.t(),edge_attr=edge_attr,y=y,num_classes=3)
Data.train_mask=np.array([1 for i in range(len(y))])

G=to_networkx(dataset, to_undirected=False)
#visualize_graph(G,color=dataset.y)

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

    def forward(self,data):
        x = data.x
       
        edge_index = data.edge_index
        print(edge_index.shape)
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

    loss.backward()

    optimizer.step()

    print('Epoch %d | Loss: %.4f' % (epoch,loss.item()))

    model.eval()

    _,pred = model(dataset).max(dim=1)
 
predict=pred.cpu()
data_y=dataset.y.cpu()
count=0
for i in range(len(predict)): 
    if predict[i]==data_y[i]:
        count += 1

print(count/len(data_y))
print("結果：",predict)
print("真値：",data_y)
