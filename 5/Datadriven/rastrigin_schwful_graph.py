import os 
import numpy as np
import torch 
from torch_geometric.data import Data
from rastrigin_and_schwefl import weight
import torch.nn.functional as F

edge_index=torch.tensor([
    [0,10],[0,11],[0,12],[0,13],[0,14],[0,15],[0,16],[0,17],
    [1,10],[1,11],[1,12],[1,13],[1,14],[1,15],[1,16],[1,17],
    [2,10],[2,11],[2,12],[2,13],[2,14],[2,15],[2,16],[2,17],
    [3,10],[3,11],[3,12],[3,13],[3,14],[3,15],[3,16],[3,17],
    [4,10],[4,11],[4,12],[4,13],[4,14],[4,15],[4,16],[4,17],
    [5,10],[5,11],[5,12],[5,13],[5,14],[5,15],[5,16],[5,17],
    [6,10],[6,11],[6,12],[6,13],[6,14],[6,15],[6,16],[6,17],
    [7,10],[7,11],[7,12],[7,13],[7,14],[7,15],[7,16],[7,17],
    [8,10],[8,11],[8,12],[8,13],[8,14],[8,15],[8,16],[8,17],
    [9,10],[9,11],[9,12],[9,13],[9,14],[9,15],[9,16],[9,17],
    [28,10],[28,11],[28,12],[28,13],[28,14],[28,15],[28,16],[28,17],
    [10,18],[10,19],[10,20],[10,21],[10,22],[10,23],
    [11,18],[11,19],[11,20],[11,21],[11,22],[11,23],
    [12,18],[12,19],[12,20],[12,21],[12,22],[12,23],
    [13,18],[13,19],[13,20],[13,21],[13,22],[13,23],
    [14,18],[14,19],[14,20],[14,21],[14,22],[14,23],
    [15,18],[15,19],[15,20],[15,21],[15,22],[15,23],
    [16,18],[16,19],[16,20],[16,21],[16,22],[16,23],
    [17,18],[17,19],[17,20],[17,21],[17,22],[17,23],
    [29,18],[29,19],[29,20],[29,21],[29,22],[29,23],
    [18,24],[18,25],[18,26],[18,27],
    [19,24],[19,25],[19,26],[19,27],
    [20,24],[20,25],[20,26],[20,27],
    [21,24],[21,25],[21,26],[21,27],
    [22,24],[22,25],[22,26],[22,27],
    [23,24],[23,25],[23,26],[23,27],
    [30,24],[30,25],[30,26],[30,27],
],dtype=torch.long)

weight,bias=weight()
edge_attr=weight
x=bias
y=torch.tensor([0,0,0,0,0,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2])


dataset=Data(x=x,edge_index=edge_index.t(),edge_attr=edge_attr,y=y,num_classes=3)
Data.train_mask=np.array([1 for i in range(len(y))])


print(dataset)
print('==============================================================')




from torch_geometric.nn.conv.gcn_conv import GCNConv

class Net(torch.nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        hidden_size = 5

        self.conv1=GCNConv(dataset.num_node_features,hidden_size)
        self.conv2=GCNConv(hidden_size,hidden_size)

        self.linear=torch.nn.Linear(hidden_size,dataset.num_classes)

    def forward(self,data):
        x = data.x
        edge_index = data.edge_index

        x = self.conv1(x,edge_index)
      
        x=F.relu(x)

        x=self.conv2(x,edge_index)
        x=F.relu(x)
        x=self.linear(x)
        return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model =Net()
model.to(device)
model.train()

optimizer=torch.optim.Adam(model.parameters(),lr=0.01)

loss_func = torch.nn.CrossEntropyLoss()

for epoch in range(100):
    optimizer.zero_grad()
    dataset.to(device)
    out = model(dataset)
    loss = loss_func(out,dataset.y)

    loss.backward()

    optimizer.step()

    print('Epoch %d | Loss: %.4f' % (epoch,loss.item()))

    model.eval()

    _,pred = model(dataset).max(dim=1)

print("結果",pred.cpu())
print("真値：",dataset.y.cpu())