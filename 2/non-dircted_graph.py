import torch 
from torch_geometric.data import Data

edge_indx = torch.tensor([[0,1],
                          [1,0],
                          [1,2],
                          [2,1]],dtype=torch.long)
x = torch.tensor([[-1],[0],[1]],dtype=torch.float)

data = Data(x=x,edge_indx=edge_indx.t().contiguous())

print(data)