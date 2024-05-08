from torch_geometric.datasets import TUDataset
dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES')
#>>> ENZYMES(600)
print(len(dataset))
#>>> 600
print(dataset.num_classes)
#>>> 6
print(dataset.num_node_features)
#>>> 3

data = dataset[0]
print(data)
#>>> Data(edge_index=[2, 168], x=[37, 3], y=[1])
print(data.is_undirected())
#>>> True