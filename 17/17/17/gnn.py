import torch
from torch_geometric.nn import GATConv
from icecream import ic
import torch.nn.functional as F

# Define the Number of Clusters
num_clusters = 3
dim = 20
K = num_clusters
clusters = []
# Channel Parameters & GAE MODEL
in_channels = dim
hidden_channels = 40
out_channels = 3

# Transform Parameters
transform_set = True


# Epochs or the number of generation/iterations of the training dataset
# epoch and n_init refers to the number of times the clustering algorithm will run different initializations
epochs = 100
n = 1000

class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_size, out_channels):
      super(GCNEncoder, self).__init__()

      # GCN
      # self.conv1 = GCNConv(in_channels, hidden_size, cached=True) # cached only for transductive learning
      # self.conv2 = GCNConv(hidden_size, out_channels, cached=True) # cached only for transductive learning

      # SAGE
      # self.conv1 = SAGEConv(in_channels, hidden_channels, cached=True) # cached only for transductive learning
      # self.conv2 = SAGEConv(hidden_channels, out_channels, cached=True) # cached only for transductive learning

      # GAT
      self.in_head = 8
      self.out_head = 1
      hidden_channels2 = 100
      hidden_channels3=80
      hidden_channels4 =40
      hidden_channels5= 20
      self.conv1 = GATConv(in_channels, hidden_channels, heads=self.in_head, dropout=0.6)
      self.conv2 = GATConv(hidden_channels*self.in_head,hidden_channels2)
      self.conv3 = GATConv(hidden_channels2,hidden_channels3)
      self.conv4 = GATConv(hidden_channels3,hidden_channels4)
      self.conv5 = GATConv(hidden_channels4, hidden_channels5)#, heads=self.out_head, dropout=0.6)
      self.conv6 = GATConv(hidden_channels5, out_channels, concat=False)
      

    def forward(self, x, edge_index):
      x = self.conv1(x, edge_index).relu()
      x = F.dropout(x, p=0.6, training=self.training)
      x = self.conv2(x,edge_index).relu()
      x = self.conv3(x, edge_index).relu()
      x = self.conv4(x, edge_index).relu()
      x = self.conv5(x,edge_index).relu()
      x = self.conv6(x,edge_index)
      return x
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')