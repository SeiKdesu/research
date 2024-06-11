import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv

import torch_geometric.transforms as T

import matplotlib.pyplot as plt
import os
from torch_geometric.datasets import CitationFull

# データセットのロード
dataset = Planetoid(root='./data', name='cora')

# GCN (Graph Convolutional Network) モデルの定義
class GCN(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, num_classes)
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# モデルのインスタンスを作成
model = GCN(num_features=dataset.num_features, hidden_channels=16, num_classes=dataset.num_classes).to(device)
data = dataset[0].to(device)
print('data',data)
# オプティマイザの設定
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

model.train()
for epoch in range(200):
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data.train_mask],
    data.y[data.train_mask])
    loss.backward()
    optimizer.step()
model.eval()
_,pred = model(data).max(dim=1)
correct= int(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
acc = correct/int (data.test_mask.sum())
print('Accuracy:{:.4f}'.format(acc))
