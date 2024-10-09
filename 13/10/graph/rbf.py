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
    ranges = [[0, 1], [2, 3, 4, 5, 6, 7, 8], [9]]
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
    plt.savefig('rosenbrock_RBF.png')



# edge_index=torch.tensor([
#     [0,2],[0,3],[0,4],[0,5],[0,6],[0,7],[0,8],[0,9],    [0,10],[0,11],[0,12],[0,13],[0,14],[0,15],[0,16],[0,17],    [0,18],[0,19],[0,20],[0,21],[0,22],
#         [1,2],[1,3],[1,4],[1,5],[1,6],[1,7],[1,8],[1,9],    [1,10],[1,11],[1,12],[1,13],[1,14],[1,15],[1,16],[1,17],    [1,18],[1,19],[1,20],[1,21],[1,22],
#     [2,23],[3,23],[4,23],[5,23],[6,23],[7,23],[8,23],[9,23],    [10,23],[11,23],[12,23],[13,23],[14,23],[15,23],[16,23],[17,23],    [18,23],[19,23],[20,23],[21,23],[22,23]
    
# ],dtype=torch.long)


# src =[]
# dst=[]
# for j in range(6):
#     for i in range(61):
#         src.append(j)
# for i in range(6):
#     for i in range(6,67):
#         dst.append(i)
# for i in range(6,67):
#     src.append(i)
# for i in range(6,67):
#     dst.append(67)
src=[0,0,0,0,0,0,0,1,1,1,1,1,1,1,2,3,4,5,6,7,8]
dst=[2,3,4,5,6,7,8,2,3,4,5,6,7,8,9,9,9,9,9,9,9]
edge_index=torch.tensor([src,dst],dtype=torch.long)

print(edge_index.shape)
tmp=[ 124.29292929 , 94.29292929 , 64.29292929 , 34.29292929   ,4.29292929,
  25.70707071 , 55.70707071,126.31313131 , 96.31313131 , 66.31313131 , 36.31313131  , 6.31313131,
  23.68686869 , 53.68686869,8552.13418799 ,1831.1166481  , 790.06251868 ,-284.97518196,  368.38050553,
 1155.84860551 ,6927.07831051]
params=[]
params=torch.tensor(tmp)

edge_attr=params
np.random.seed(1234)
x=[[-35.0,0],[-100.0,0],[-95.,0],[ -65.,0],[ -35.,0]  ,[-5. ,0],[ 25.,0],[  55. ,0],[ 85.,0],[1265.0,0]]
#x=np.zeros_like(a)
x=torch.tensor(x,dtype=torch.float)
#x=torch.tensor([[0],[0],[0],[0],[0],[1],[1],[1],[1],[1],[2],[2],[2],[2],[2],[2],[2],[2],[2],[2],[2],[2],[2],[2],[2],[2],[2],[2],[3],[3],[3]],dtype=torch.float)
y_tmp=[0,1,1,1,2,0,2,2,2,2]

y = torch.tensor(y_tmp)


dataset=Data(x=x,edge_index=edge_index,edge_attr=edge_attr,y=y,num_classes=7)
Data.train_mask=np.array([1 for i in range(len(y))])

G=to_networkx(dataset, to_undirected=False)
# visualize_graph(G,color=dataset.y)

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

for epoch in range(300):
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
