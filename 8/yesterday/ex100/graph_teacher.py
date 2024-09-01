import os 
import numpy as np
import torch 
from torch_geometric.data import Data

import torch.nn.functional as F
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import to_networkx
from rosenbrock import weight
from torch import optim
def visualize_graph(G, color):
    plt.figure(figsize=(3, 3))
    plt.xticks([])
    plt.yticks([])

    # ノードの範囲ごとに縦1列に配置するための位置を設定
    pos = {}

    # 各範囲ごとにノードを縦1列に並べる
    ranges = [
        list(range(1, 95)) + [194],
        list(range(95, 146)) + [195],
        list(range(146, 176)) + [196],
        list(range(176, 186)) + [197]
    ]
    x_offset = 0  # X軸のオフセット

    # すべてのノードの位置を設定
    for r in ranges:
        for i, node in enumerate(r):
            pos[node] = (x_offset, -i)
        x_offset += 1  # 次の列に移動

    # 位置が指定されていないノードにも位置を設定
    for node in G.nodes():
        if node not in pos:
            pos[node] = (x_offset, 0)  # 既定の位置を設定

    # エッジの重みに基づいて太さを決定
    weights = nx.get_edge_attributes(G, 'weight')
    default_width = 1.0
    edge_widths = [weights[edge] if edge in weights else default_width for edge in G.edges()]

    # グラフを描画
    plt.figure(figsize=(8, 8))
    nx.draw_networkx_nodes(G, pos, node_size=700, node_color=color, cmap=plt.cm.rainbow)
    nx.draw_networkx_labels(G, pos)
    nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color='gray', arrows=True)

    plt.savefig('rastrigin_schwful_10.png')


weight_edge,row_indices,col_indices=weight()
# エッジのインデックス（ノード間の接続）を定義
#row_indices = torch.tensor([0, 0, 0, 0,0,1,1,1,1,1,1,1,2,3,4,4,4,4,5,5,6,6,7,7,8,8,8,8,8,8,8,9,9,9,9,9,9,9,10,10,10,11,11,11,12,12,13,13,13,14,14,14,15,15,15,16,16,16,16,16,16,17,17,17,18,18,18,18,19,19,19,19,20,20,20,20,21,21,21,21,22,22,22,22,23,23,23,23,28,28,28,28,28,28,28,28,29,29,29,29,29,29,30,30,30,30])
#col_indices = torch.tensor([13,14,15,16,17,10,11,12,13,14,15,17,15,16,10,11,13,15,13,17,16,17,11,14,10,11,13,14,15,16,17,10,11,12,14,15,16,17,19,20,21,19,20,21,22,23,18,22,23,18,22,23,19,20,21,18,19,20,21,22,23,19,20,21,24,25,26,27,24,25,26,27,24,25,26,27,24,25,26,27,24,25,26,27,24,25,26,27,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27])

# 各エッジの重みを定義
values = weight_edge
#values =x
# 行と列のインデックスを組み合わせて、疎行列のインデックスを作成
print(row_indices.shape,col_indices.shape)
tensor_data= torch.stack([row_indices, col_indices],dim=0)

result = [[tensor_data[0, i].item(), tensor_data[1, i].item()] for i in range(tensor_data.shape[1])]
edge_index=result
print(result)
edge_index=torch.tensor(edge_index)

edge_attr=weight_edge

np.random.seed(0000)
x=np.random.rand(197,1)
#x=np.zeros_like(a)
x=torch.tensor(x,dtype=torch.float)
#x=torch.tensor([[0],[0],[0],[0],[0],[1],[1],[1],[1],[1],[2],[2],[2],[2],[2],[2],[2],[2],[2],[2],[2],[2],[2],[2],[2],[2],[2],[2],[3],[3],[3]],dtype=torch.float)
print(x)
#y=torch.tensor([0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])
# 配列のサイズを指定
size = 197

# テンソルをゼロで初期化
y = torch.zeros(size, dtype=torch.long)

# 条件に基づいてテンソルの値を設定
y[31:60] = 1       # 71~72のインデックスは1
y[60:92] = 2       # 63~74のインデックスは2
y[93:197] = 3      # 75~118のインデックスは3

print(y)


dataset=Data(x=x,edge_index=edge_index.t(),edge_attr=edge_attr,y=y,num_classes=4)

# dataset=dataset[0]
G=to_networkx(dataset, to_undirected=False)
# visualize_graph(G,color=dataset.y)

print(dataset)

print('==============================================================')


from torch_geometric.nn.conv.gcn_conv import GCNConv

class Net(torch.nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        hidden_size = 85
       
        self.conv1=GCNConv(dataset.num_node_features,hidden_size)
        self.conv2=GCNConv(hidden_size,80)
       
        self.conv6=GCNConv(80,50)
        self.conv3=GCNConv(50,30)
        self.conv4=GCNConv(30,20)
        #self.conv5=GCNConv(20,10)

        self.linear=torch.nn.Linear(20,dataset.num_classes)

    def forward(self,data):
        x = data.x
       
        edge_index = data.edge_index

        x = self.conv1(x,edge_index)
        x=F.relu(x)

        x=self.conv2(x,edge_index)
        x=F.relu(x)
        # x=self.conv7(x,edge_index)
        # x=F.relu(x)
        x=self.conv6(x,edge_index)
        x=F.relu(x)
        x=self.conv3(x,edge_index)
        x=F.relu(x)
        
        x=self.conv4(x,edge_index)
        x=F.relu(x)
        # x=self.conv5(x,edge_index)
        # x=F.relu(x)
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
optimizer=torch.optim.Adam(model.parameters(),lr=0.03)
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max = 4000
)
loss_func = torch.nn.CrossEntropyLoss()
losses=[]
for epoch in range(4000):
    optimizer.zero_grad()
    dataset.to('cpu')
    out = model(dataset)
    loss = loss_func(out,dataset.y)
    losses.append(loss)
    loss.backward()

    optimizer.step()

    

    model.eval()

    _,pred = model(dataset).max(dim=1)
    scheduler.step()
 
    predict=pred.cpu()
    data_y=dataset.y.cpu()
    count=0
    for i in range(len(predict)): 
        if predict[i]==data_y[i]:
            count += 1

    print('Epoch %d | Loss: %.4f | ACC: %.4f' % (epoch,loss.item(),count/len(data_y)))
print("結果：",predict)
print("真値：",data_y)

# リスト内の各テンソルをdetachしてnumpy配列に変換
losses_np = [loss.detach().numpy() for loss in losses]

import matplotlib.pyplot as plt
plt.figure(figsize=(10, 5))
plt.plot(losses_np, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
# plt.ylim(0,0.2e-6)
plt.title('Loss vs. Epoch')
plt.legend()
plt.grid(True)
plt.savefig('loss')
plt.close()