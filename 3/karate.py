import torch
import torch.nn.functional as F

from torch_geometric.datasets import KarateClub
dataset = KarateClub()

print("グラフ数",len(dataset))
print("クラス数：",dataset.num_classes)

data = dataset[0]

def check_graph(data):
    print("グラフ構造:", data)
    print("グラフのキー: ", data.keys)
    print("ノード数:", data.num_nodes)
    print("エッジ数:", data.num_edges)
    print("ノードの特徴量数:", data.num_node_features)
    print("孤立したノードの有無:", data.contains_isolated_nodes())
    print("自己ループの有無:", data.contains_self_loops())
    print("====== ノードの特徴量:x ======")
    print(data['x'])
    print("====== ノードのクラス:y ======")
    print(data['y'])
    print("========= エッジ形状 =========")
    print(data['edge_index'])
 
check_graph(data)

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
    data.to(device)
    out = model(data)
    loss = loss_func(out,data.y)

    loss.backward()

    optimizer.step()

    print('Epoch %d | Loss: %.4f' % (epoch,loss.item()))

    model.eval()

    _,pred = model(data).max(dim=1)

print("結果",pred.cpu())
print("真値：",data.y.cpu())