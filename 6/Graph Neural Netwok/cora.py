
import torch
from torch_geometric.datasets import Planetoid

dataset = Planetoid(root='/tmp/Cora',name='Cora')

print(len(dataset))