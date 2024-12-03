#!/usr/bin/env python
# coding: utf-8

# In[5]:


import os
import torch

torch.manual_seed(53)
os.environ['TORCH'] = torch.__version__
print(torch.__version__)

# get_ipython().system('pip install -q torch-scatter -f https://data.pyg.org/whl/torch-${TORCH}.html')
# get_ipython().system('pip install -q torch-sparse -f https://data.pyg.org/whl/torch-${TORCH}.html')
# get_ipython().system('pip install -q git+https://github.com/pyg-team/pytorch_geometric.git')

import torch_geometric


# In[6]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = "cpu"


# In[ ]:


import numpy as np
from scipy.spatial import distance
from torch_geometric.data import Data, InMemoryDataset
from rosenbrock_nn_similarity import dim_num_dataset
class GridDataset(InMemoryDataset):
    def __init__(self, transform = None):
        # super().__init__('.', transform)

        # f = lambda x: np.linalg.norm(x) - np.arctan2(x[0], x[1])
        # embeddings = []
        # ys = []
        # for x in range(-10, 11, 2):
        #     for y in range(-10, 11, 2):
        #         embeddings.append([x, y])
        #         ys.append(f([x, y]))
        # embeddings = torch.tensor(embeddings, dtype=torch.float)
        # y2 = []
        # for y in ys:
        #     if y > np.array(ys).mean():
        #         y2.append(1)
        #     else:
        #         y2.append(0)
        # ys = torch.tensor(y2, dtype=torch.float)

        # dist_matrix = distance.cdist(embeddings, embeddings, metric='euclidean')
        # edges = []
        # edge_attr = []
        # for i in range(len(dist_matrix)):
        #     for j in range(len(dist_matrix)):
        #         if i < j:
        #             if dist_matrix[i][j] == 2:
        #                 edges.append([i, j])
        #                 edge_attr.append(abs(f(embeddings[i]) - f(embeddings[j])))
        #             elif dist_matrix[i][j] < 3 and (
        #                 embeddings[i][0] == embeddings[j][1] or
        #                 embeddings[i][1] == embeddings[j][0]
        #             ):
        #                 edges.append([i, j])
        #                 edge_attr.append(abs(f(embeddings[i]) - f(embeddings[j])))

        # edges = torch.tensor(edges, dtype=torch.long).T
        # edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        # data = Data(x=embeddings, edge_index=edges, y=ys, edge_attr=edge_attr)
        data = dim_num_dataset(dim_valuem=1)
        self.data.num_features = 35
        self.data, self.slices = self.collate([data])
        # self.data.num_nodes = len(embeddings)

    def layout(self):
        return {i:x.detach().numpy() for i, x in enumerate(self.data.x)}

    def node_color(self):
        c = {0:"red", 1:"blue"}
        return [c[int(x.detach().numpy())] for (i, x) in enumerate(self.data.y)]


# In[ ]:





# In[ ]:


# import networkx as nx
# import matplotlib.pyplot as plt

# dataset = ()
# G = torch_geometric.utils.convert.to_networkx(dataset.data)
# plt.figure(figsize=(12,12))
# nx.draw_networkx(G, pos=dataset.layout(), with_labels=False, alpha=0.5, node_color=dataset.node_color())


# In[ ]:


# import numpy as np
# from scipy.spatial import distance
# from torch_geometric.data import Data, InMemoryDataset

# class ColonyDataset(InMemoryDataset):
#     def __init__(self, transform = None):
#         super().__init__('.', transform)

#         f = lambda x: np.linalg.norm(x) - np.arctan2(x[0], x[1])
#         embeddings = []
#         ys = []

#         for x in range(-10, 11, 5):
#             for y in range(-10, 11, 5):
#                 embeddings.append([x, y])
#                 ys.append(f([x, y]))
#                 for theta in range(max(0, 15 - abs(x) - abs(y))):
#                     x2 = x + np.sin(theta + np.random.rand()) * abs(17 - theta) * 0.1
#                     y2 = y + np.cos(theta + np.random.rand()) * abs(17 - theta) * 0.1
#                     embeddings.append([x2, y2])
#                     ys.append(f([x2, y2]))
                
#         embeddings = torch.tensor(embeddings, dtype=torch.float)
#         y2 = []
#         for y in ys:
#             if y > np.array(ys).mean():
#                 y2.append(1)
#             else:
#                 y2.append(0)
#         ys = torch.tensor(y2, dtype=torch.float)

#         dist_matrix = distance.cdist(embeddings, embeddings, metric='euclidean')
#         edges = []
#         edge_attr = []
#         for i in range(len(dist_matrix)):
#             for j in range(len(dist_matrix)):
#                 if i < j:
#                     if dist_matrix[i][j] == 5 or dist_matrix[i][j] < 2:
#                         edges.append([i, j])
#                         edge_attr.append(abs(f(embeddings[i]) - f(embeddings[j])))

#         edges = torch.tensor(edges).T
#         edge_attr = torch.tensor(edge_attr)
#         data = Data(x=embeddings, edge_index=edges, y=ys, edge_attr=edge_attr)
#         self.data, self.slices = self.collate([data])
#         self.data.num_nodes = len(embeddings)

#     def layout(self):
#         return {i:x.detach().numpy() for i, x in enumerate(self.data.x)}
    
#     def node_color(self):
#         c = {0:"red", 1:"blue"}
#         return [c[int(x.detach().numpy())] for (i, x) in enumerate(self.data.y)]


# # In[ ]:


# import networkx as nx
# import matplotlib.pyplot as plt

# dataset = ColonyDataset()
# G = torch_geometric.utils.convert.to_networkx(dataset.data)
# plt.figure(figsize=(12,12))
# nx.draw_networkx(G, pos=dataset.layout(), with_labels=False, alpha=0.5, node_color=dataset.node_color())


# In[ ]:


use_dataset = GridDataset
# use_dataset = ColonyDataset


# In[ ]:
import torch_geometric.utils as tg_utils

dataset = use_dataset()
data = dataset
# data = tg_utils.train_test_split_edges(data)


# In[ ]:


class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNEncoder, self).__init__()
        self.conv1 = torch_geometric.nn.GCNConv(
            in_channels, 4 * out_channels, cached=True
            ) 
        self.conv1b = torch_geometric.nn.GCNConv(
            4 * out_channels, 16 * out_channels, cached=True
            ) 
        self.conv1c = torch_geometric.nn.GCNConv(
            16 * out_channels, 32 * out_channels, cached=True
            ) 
        self.conv1d = torch_geometric.nn.GCNConv(
            32 * out_channels, 16 * out_channels, cached=True
            ) 
        self.conv1e = torch_geometric.nn.GCNConv(
            16 * out_channels, 4 * out_channels, cached=True
            ) 
        self.conv2 = torch_geometric.nn.GCNConv(
            4 * out_channels, out_channels, cached=True
            ) 

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv1b(x, edge_index).relu()
        x = self.conv1c(x, edge_index).relu()
        x = self.conv1d(x, edge_index).relu()
        x = self.conv1e(x, edge_index).relu()
        return self.conv2(x, edge_index)


# In[ ]:


def train():
    model.train()
    optimizer.zero_grad()
    z = model.encode(x, train_pos_edge_index)
    loss = model.recon_loss(z, train_pos_edge_index)
    loss.backward()
    optimizer.step()
    return float(loss)

def test(pos_edge_index, neg_edge_index):
    model.eval()
    with torch.no_grad():
        z = model.encode(x, train_pos_edge_index)
    return model.test(z, pos_edge_index, neg_edge_index)


# In[ ]:


epochs = 500
out_channels = 16

num_features = dataset.num_features
model = torch_geometric.nn.GAE(GCNEncoder(num_features, out_channels))
model = model.to(device)

x = data.x.to(device)
train_pos_edge_index = data.train_pos_edge_index.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


# In[ ]:


import copy

loss_hist = []
auc_hist = []
ap_hist = []
best_score = None
for epoch in range(1, epochs + 1):
    loss = train()
    auc, ap = test(data.test_pos_edge_index, data.test_neg_edge_index)
    loss_hist.append(loss)
    auc_hist.append(auc)
    ap_hist.append(ap)
    if best_score is None or best_score < ap:
        best_score = ap
        best_model = copy.deepcopy(model)
        print('Epoch: {:03d}, AUC: {:.4f}, AP: {:.4f}, Loss: {}'.format(epoch, auc, ap, loss))


# In[ ]:


import matplotlib.pyplot as plt

plt.title("GAE")
plt.plot(loss_hist, label="Loss")
plt.grid()
plt.legend()
plt.yscale('log')
plt.show()
plt.title("GAE")
plt.plot(auc_hist, label="AUC")
plt.plot(ap_hist, label="AP")
plt.grid()
plt.legend()
plt.show()


# In[ ]:


z = best_model.encode(x, train_pos_edge_index)
prob_adj = z @ z.T
prob_adj = prob_adj - torch.diagonal(prob_adj)
prob_adj


# In[ ]:


prob_adj_values = prob_adj.detach().cpu().numpy().flatten()
prob_adj_values.sort()
dataset = use_dataset()
threshold = prob_adj_values[-len(dataset.data.edge_attr)]
dataset.data.edge_index = (prob_adj >= threshold).nonzero(as_tuple=False).t()


# In[ ]:


import networkx as nx
import matplotlib.pyplot as plt

G = torch_geometric.utils.convert.to_networkx(dataset.data)
plt.figure(figsize=(12,12))
nx.draw_networkx(G, pos=dataset.layout(), with_labels=False, alpha=0.5, node_color=dataset.node_color())


# In[ ]:





# # Variable Auto Encoder

# 

# In[ ]:


class VariationalGCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VariationalGCNEncoder, self).__init__()
        self.conv1 = torch_geometric.nn.GCNConv(
            in_channels, 4 * out_channels, cached=True
            ) 
        self.conv1b = torch_geometric.nn.GCNConv(
            4 * out_channels, 16 * out_channels, cached=True
            ) 
        self.conv1c = torch_geometric.nn.GCNConv(
            16 * out_channels, 32 * out_channels, cached=True
            ) 
        self.conv1d = torch_geometric.nn.GCNConv(
            32 * out_channels, 16 * out_channels, cached=True
            ) 
        self.conv1e = torch_geometric.nn.GCNConv(
            16 * out_channels, 4 * out_channels, cached=True
            ) 
        self.conv_mu = torch_geometric.nn.GCNConv(
            4 * out_channels, out_channels, cached=True
            )
        self.conv_logstd = torch_geometric.nn.GCNConv(
            4 * out_channels, out_channels, cached=True
            )

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv1b(x, edge_index).relu()
        x = self.conv1c(x, edge_index).relu()
        x = self.conv1d(x, edge_index).relu()
        x = self.conv1e(x, edge_index).relu()
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)


# In[ ]:


def train():
    model.train()
    optimizer.zero_grad()
    z = model.encode(x, train_pos_edge_index)
    loss = model.recon_loss(z, train_pos_edge_index)
    loss = loss + (1 / data.num_nodes) * model.kl_loss()
    loss.backward()
    optimizer.step()
    return float(loss)

def test(pos_edge_index, neg_edge_index):
    model.eval()
    with torch.no_grad():
        z = model.encode(x, train_pos_edge_index)
    return model.test(z, pos_edge_index, neg_edge_index)


# In[ ]:


epochs = 500
out_channels = 16

num_features = dataset.num_features
model = torch_geometric.nn.VGAE(
    VariationalGCNEncoder(num_features, out_channels)
    )
model = model.to(device)

x = data.x.to(device)
train_pos_edge_index = data.train_pos_edge_index.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


# In[ ]:


import copy

loss_hist = []
auc_hist = []
ap_hist = []
best_score = None
for epoch in range(1, epochs + 1):
    loss = train()
    auc, ap = test(data.test_pos_edge_index, data.test_neg_edge_index)
    loss_hist.append(loss)
    auc_hist.append(auc)
    ap_hist.append(ap)
    if best_score is None or best_score < ap:
        best_score = ap
        best_model = copy.deepcopy(model)
        print('Epoch: {:03d}, AUC: {:.4f}, AP: {:.4f}, Loss: {}'.format(epoch, auc, ap, loss))


# In[ ]:


import matplotlib.pyplot as plt

plt.title("VGAE")
plt.plot(loss_hist, label="Loss")
plt.grid()
plt.legend()
plt.yscale('log')
plt.show()
plt.title("VGAE")
plt.plot(auc_hist, label="AUC")
plt.plot(ap_hist, label="AP")
plt.grid()
plt.legend()
plt.show()


# In[ ]:


z = best_model.encode(x, train_pos_edge_index)
prob_adj = z @ z.T
prob_adj = prob_adj - torch.diagonal(prob_adj)
prob_adj


# In[ ]:


prob_adj_values = prob_adj.detach().cpu().numpy().flatten()
prob_adj_values.sort()
dataset = use_dataset()
threshold = prob_adj_values[-len(dataset.data.edge_attr)]
dataset.data.edge_index = (prob_adj >= threshold).nonzero(as_tuple=False).t()


# In[ ]:


import networkx as nx
import matplotlib.pyplot as plt

G = torch_geometric.utils.convert.to_networkx(dataset.data)
plt.figure(figsize=(12,12))
nx.draw_networkx(G, pos=dataset.layout(), with_labels=False, alpha=0.5, node_color=dataset.node_color())


# In[ ]:




