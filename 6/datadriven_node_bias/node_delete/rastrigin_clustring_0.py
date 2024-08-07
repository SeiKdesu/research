#ファイルの参照
import os.path as osp
#pytrochのインストール
import torch
#線形変換
from torch.nn import Linear
#グラフデータの変換
import torch_geometric.transforms as T
#データセット
from torch_geometric.datasets import Planetoid
#GCNの畳み込み
from torch_geometric.nn import GCNConv
#ユーティリティ関数
from torch_geometric import utils
#シーケンシャルモデル
from torch_geometric.nn import Sequential
#クラスタリングの評価指標である正規化相互情報量
from sklearn.metrics import normalized_mutual_info_score as NMI
#自作ライブラリjust_balance_pygからjust_balance_pool関数
from just_balance_pyg import just_balance_pool
#、ランダムシード
torch.manual_seed(1) # for (inconsistent) reproducibility
torch.cuda.manual_seed(1)
import os 
import numpy as np
import torch 
from torch_geometric.data import Data
from rastrigin_and_schwefl_bias_node import param_weights,weight
import torch.nn.functional as F
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import to_networkx



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
print(params)
edge_attr=params
np.random.seed(1111)
x=np.random.rand(31,1)
x=np.ones_like(x)
x=torch.tensor(x,dtype=torch.float)
#x=torch.tensor([[0],[0],[0],[0],[0],[1],[1],[1],[1],[1],[2],[2],[2],[2],[2],[2],[2],[2],[2],[2],[2],[2],[2],[2],[2],[2],[2],[2],[3],[3],[3]],dtype=torch.float)
print('こここここここおこここおれががががｇ',x)
y=torch.tensor([0,0,0,0,0,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,3,3,3])
#print(edge_index.shape)

dataset=Data(x=x,edge_index=edge_index.t(),edge_attr=edge_attr,y=y,num_classes=4)
Data.train_mask=np.array([1 for i in range(len(y))])

G=to_networkx(dataset, to_undirected=False)
 
print(G)
data = dataset

# Compute connectivity matrix
#グラフの接続行列を計算します。まず、正規化ラプラシアン行列
#接続行列を計算し、それをスパース形式
delta = 0.85
edge_index, edge_weight = utils.get_laplacian(data.edge_index, data.edge_weight, normalization='sym')
L = utils.to_dense_adj(edge_index, edge_attr=edge_weight)
A = torch.eye(data.num_nodes) - delta*L
data.edge_index, data.edge_weight = utils.dense_to_sparse(A)

#__init__:メッセージパッシング層とMLP層
#forward: 入力データをモデルに通し、クラスタリング結果とバランスプール損失を返します。
class Net(torch.nn.Module):
    def __init__(self, 
                 mp_units,
                 mp_act,
                 in_channels, 
                 n_clusters, 
                 mlp_units=[],
                 mlp_act="Identity"):
        super().__init__()
        
        mp_act = getattr(torch.nn, mp_act)(inplace=True)
        mlp_act = getattr(torch.nn, mlp_act)(inplace=True)
        
        # Message passing layers
        mp = [
            (GCNConv(in_channels, mp_units[0], normalize=False, cached=False), 'x, edge_index, edge_weight -> x'),
            mp_act
        ]
        for i in range(len(mp_units)-1):
            mp.append((GCNConv(mp_units[i], mp_units[i+1], normalize=False, cached=False), 'x, edge_index, edge_weight -> x'))
            mp.append(mp_act)
        self.mp = Sequential('x, edge_index, edge_weight', mp)
        out_chan = mp_units[-1]
        
        # MLP layers
        self.mlp = torch.nn.Sequential()
        for units in mlp_units:
            self.mlp.append(Linear(out_chan, units))
            out_chan = units
            self.mlp.append(mlp_act)
        self.mlp.append(Linear(out_chan, n_clusters))
        

    def forward(self, x, edge_index, edge_weight):
        
        # Propagate node feats
        x = self.mp(x, edge_index, edge_weight)
        
        # Cluster assignments (logits)
        s = self.mlp(x)
        
        # Compute loss
        adj = utils.to_dense_adj(edge_index, edge_attr=edge_weight)
        _, _, b_loss = just_balance_pool(x, adj, s)
        
        return torch.softmax(s, dim=-1), b_loss


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = data.to(device)
print(dataset.num_classes)
model = Net([64]*10, "ReLU", dataset.num_features, 4, [16], "ReLU").to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)


def train():
    model.train()
    optimizer.zero_grad()
    _, loss = model(data.x, data.edge_index, data.edge_weight)
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def test():
    model.eval()
    clust, _ = model(data.x, data.edge_index, data.edge_weight)
    print(clust.max(1)[1])
    return NMI(clust.max(1)[1].cpu(), data.y.cpu())
    

for epoch in range(1, 500):
    train_loss = train()
    nmi = test()
    print(f'Epoch: {epoch:03d}, Loss: {train_loss:.4f}, ' f'NMI: {nmi:.3f}')
