import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import MessagePassing
from torchinfo import summary


# データセットのロード
dataset = Planetoid(root='./data', name='cora')

# GCN (Graph Convolutional Network) モデルの定義
class GCN(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNGhostConv(num_features, hidden_channels)
        self.conv2 = GCNGhostConv(hidden_channels, num_classes)
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# GCNConvに対応する形に調整したGhost Module風のカスタムモジュール
class GCNGhostConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(GCNGhostConv, self).__init__(aggr='add')  # 'add' aggregationを使用
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.weight = nn.Parameter(torch.Tensor(in_channels, out_channels))
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)
        
    def forward(self, x, edge_index):
        # x: ノードの特徴量行列 shape = (N, in_channels)
        # edge_index: エッジのインデックス行列 shape = (2, E)
        
        # メッセージ関数の定義
        def message_func(x_j):
            return x_j
        
        # 更新関数の定義
        def update_func(aggr_out):
            return aggr_out
        
        # メッセージの伝播
        return self.propagate(edge_index, x=x, weight=self.weight, bias=self.bias,
                              message_func=message_func, update_func=update_func)
    
    def message(self, x_j, weight):
        # x_j: 隣接するノードの特徴量行列 shape = (E, out_channels)
        return torch.matmul(x_j, weight)
    
    def update(self, aggr_out):
        # aggr_out: 集約されたメッセージ行列 shape = (N, out_channels)
        return aggr_out + self.bias

# モデルのインスタンスを作成
model = GCN(num_features=dataset.num_features, hidden_channels=16, num_classes=dataset.num_classes)

# オプティマイザの設定
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

# 学習と評価のループ
model.train()
for epoch in range(200):
    optimizer.zero_grad()
    data = dataset[0]  # データセットからデータを取得
    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

model.eval()
data = dataset[0]  # データセットからデータを取得
_, pred = model(data).max(dim=1)
correct = int(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
acc = correct / int(data.test_mask.sum())
print('Accuracy: {:.4f}'.format(acc))
