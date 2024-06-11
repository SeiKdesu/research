import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
import os

# HTTPおよびHTTPSプロキシの設定
os.environ['HTTP_PROXY'] = 'http://133.78.130.254:8081'
os.environ['HTTPS_PROXY'] = 'http://133.78.130.254:8081'
# Coraデータセットのロード
dataset = Planetoid(root='data/Planetoid', name='Cora', transform=NormalizeFeatures())

# GNNモデルの定義
class GCN(torch.nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(dataset.num_node_features, 16)
        self.conv2 = GCNConv(16, dataset.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# デバイスの設定（GPUが利用可能な場合はGPUを使用）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN().to(device)
data = dataset[0].to(device)

# 最適化手法と損失関数の定義
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

# 訓練関数
def train():
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

# テスト関数
def test():
    model.eval()
    _, pred = model(data).max(dim=1)
    correct = int(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
    acc = correct / int(data.test_mask.sum())
    return acc

# 訓練ループ
for epoch in range(200):
    loss = train()
    if epoch % 10 == 0:
        acc = test()
        print(f'Epoch: {epoch}, Loss: {loss:.4f}, Test Accuracy: {acc:.4f}')

# 最終的なテスト精度
acc = test()
print(f'Final Test Accuracy: {acc:.4f}')
