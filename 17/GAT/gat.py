import torch
import torch_geometric
import numpy as np
import matplotlib.pyplot as plt
from dataset_my import *


dataset = load_my_dataset()

print("number of graphs:\t\t",len(dataset))
print("number of classes:\t\t",dataset.num_classes)
print("number of node features:\t",dataset.num_node_features)
print("number of edge features:\t",dataset.num_edge_features)
print("edge_index:\t\t",dataset.edge_index.shape)
print(dataset.edge_index)
print("\n")
print("train_mask:\t\t",dataset.train_mask.shape)
print(dataset.train_mask)
print("\n")
print("x:\t\t",dataset.x.shape)
print(dataset.x)
print("\n")
print("y:\t\t",dataset.y.shape)
print(dataset.y)


def train_val_test_split(data, val_ratio: float = 0.15,
                             test_ratio: float = 0.15):
    rnd = torch.rand(len(data.x))
    train_mask = [False if (x > val_ratio + test_ratio) else True for x in rnd]
    val_mask = [False if (val_ratio + test_ratio >= x) and (x > test_ratio) else True for x in rnd]
    test_mask = [False if (test_ratio >= x) else True for x in rnd]
    return torch.tensor(train_mask), torch.tensor(val_mask), torch.tensor(test_mask)
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class GAT(torch.nn.Module):
    def __init__(self):
        super(GAT, self).__init__()
        self.hid = 8
        self.in_head = 8
        self.out_head = 1
        
        
        self.conv1 = GATConv(dataset.num_features, self.hid, heads=self.in_head, dropout=0.6)
        self.conv2 = GATConv(self.hid*self.in_head, dataset.num_classes, concat=False,
                             heads=self.out_head, dropout=0.6)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
                
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        
        return F.log_softmax(x, dim=1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = "cuda"

data = dataset.to(device)
model = GAT().to(device)

train_mask, val_mask, test_mask = train_val_test_split(data)

data.train_mask = train_mask
data.val_mask = val_mask
data.test_mask = test_mask


optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

losses = []
for epoch in range(2000):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    model.eval()
    _, pred = model(data).max(dim=1)
    ic(pred)

    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss = loss.cpu()
    losses.append(loss.detach().numpy())
    loss = loss.to(device)
    if epoch % 200 == 0:
        print(loss)
    
    loss.backward()
    optimizer.step()

import matplotlib.pyplot
from icecream import ic
ic(losses)


import matplotlib.pyplot as plt

# 損失のプロット
plt.figure(figsize=(10, 6))
plt.plot(range(len(losses)), losses, label="Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Over Epochs")
plt.legend()
plt.grid(True)
plt.savefig('loss_GAT.png')

model.eval()
_, pred = model(data).max(dim=1)
ic(pred)
correct = float(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
acc = correct / data.test_mask.sum().item()
print('Accuracy: {:.4f}'.format(acc))

