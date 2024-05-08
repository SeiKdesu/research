from torch_geometric.datasets import Planetoid

dataset = Planetoid(root='./data', name='Cora')

data = dataset[0]
print(data)
