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
#print('len(dataset):',len(dataset))
#print('dataset.num_classes',dataset.num_classes)
#print('dataset.num_node_features',dataset.num_node_features)
print('data[0]',dataset[0])