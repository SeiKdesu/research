import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import torch_geometric.transforms as T
from torch_geometric.utils import to_networkx
import networkx as nx
import matplotlib.pyplot as plt
import os
from torch_geometric.datasets import CitationFull

# データセットのロード
dataset = CitationFull(root='./data', name='cora')
data = dataset[0]
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
# モデルのインスタンスを作成
model = GCN(num_features=dataset.num_features, hidden_channels=16, num_classes=dataset.num_classes)
# オプティマイザの設定
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
# トレーニングループ
def train():
    model.train()
    optimizer.zero_grad()
    output = model(data)
    loss = F.nll_loss(output[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()
# テストループ
def test():
    model.eval()
    logits = model(data)
    pred = logits.argmax(dim=1)
    correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
    acc = correct.float() / data.test_mask.sum()
    return acc
# トレーニング
num_epochs = 200
for epoch in range(num_epochs):
    loss = train()
    if epoch % 10 == 0:
        acc = test()
        print(f'Epoch: {epoch}, Loss: {loss:.4f}, Test Accuracy: {acc:.4f}')
# 最終結果の評価
final_acc = test()
print(f'Final Test Accuracy: {final_acc:.4f}')
# グラフを描画して、どのようにノードが配置されているか確認
G = to_networkx(data)
nx.draw(G, with_labels=True)
plt.show()
