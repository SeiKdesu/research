import os 
import numpy as np
import torch 
from torch_geometric.data import Data

import torch.nn.functional as F
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import to_networkx
from rosenbrock_nn import weight
from torch import optim

# -*- coding: utf-8 -*-
"""GAT_Simplified_Graham,_Jessica_GAE_Clustering_Phase_1_KMeans.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/github/jegraham/1-GNN-Clustering/blob/main/GAT_Simplified_Graham%2C_Jessica_GAE_Clustering_Phase_1_KMeans.ipynb

# Simple K-Means GNN Implementation (Step 1)

This is the initial GNN implementation as referenced in WIDECOMM 2023 Paper Submission. Our implementation uses encoders and decoders with GAT, GCN, and GraphSAGE Layers. Parameters can be modified under the 'Testing Parameters' Section and will be implemented throughout the code.

## Import

### Import Libraries
"""
from datetime import datetime

# 現在の時刻を取得
current_time = datetime.now()
name = f'{current_time}_Spectural'
def visualize_graph(G, color,i):
    plt.figure(figsize=(3, 3))
    plt.xticks([])
    plt.yticks([])

    # ノードの範囲ごとに縦1列に配置するための位置を設定
    pos = {}

    # 各範囲ごとにノードを縦1列に並べる
    ranges = [[0, 1], [2, 3, 4, 5, 6, 7, 8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31], [32]]
    x_offset = 0  # X軸のオフセット

    # ノードを正しく配置するためにループを修正
    for r in ranges:
        for i, node in enumerate(r):
            pos[node] = (x_offset, -i)  # Y座標は負の値に設定
        x_offset += 1  # 次の列に移動

    # エッジの重みに基づいて太さを決定
    weights = nx.get_edge_attributes(G, 'weight')
    default_width = 1.0
    edge_widths = [weights[edge] if edge in weights else default_width for edge in G.edges()]

    # グラフを描画
    plt.figure(figsize=(8, 8))
    nx.draw_networkx_nodes(G, pos, node_size=700, node_color=color, cmap=plt.cm.rainbow)
    nx.draw_networkx_labels(G, pos)
    nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color='gray', arrows=True)

    # 画像を保存
    if(i==1):
        plt.savefig(f'acc_loss/{name}_rosencrok_teacher_.png')  # 保存

    plt.savefig(f'acc_loss/{name}_rosencrok_predict_.png')  # 保存


import os
import os.path as osp
import shutil
import pandas as pd
import random
import datetime

# libraries for the files in google drive
from pydrive.auth import GoogleAuth
# from google.colab import drive
# from pydrive.drive import GoogleDrive
# from google.colab import auth
# from oauth2client.client import GoogleCredentials

import torch
os.environ['TORCH'] = torch.__version__
print(torch.__version__)

# GPU Usage Guide - https://medium.com/@natsunoyuki/speeding-up-model-training-with-google-colab-b1ad7c48573e
if torch.cuda.is_available():
    device_name = torch.device("cuda")
else:
    device_name = torch.device('cpu')
print("Using {}.".format(device_name))


from collections import Counter
import matplotlib.pyplot as plt
import math
import networkx as nx
import numpy as np
from scipy.spatial.distance import cdist, squareform
from scipy import stats
from sklearn.cluster import KMeans, MeanShift, AffinityPropagation, FeatureAgglomeration, SpectralClustering, MiniBatchKMeans, Birch, DBSCAN, OPTICS, AgglomerativeClustering
from sklearn.impute import SimpleImputer
from sklearn.mixture import GaussianMixture
from sklearn.metrics import confusion_matrix, pairwise_distances, davies_bouldin_score, silhouette_score, calinski_harabasz_score, adjusted_rand_score, normalized_mutual_info_score
from torch_geometric.data import Data, InMemoryDataset, download_url
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import Planetoid, TUDataset
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, SAGEConv, GAE, GINConv, GATConv
from torch_geometric.utils import train_test_split_edges, to_networkx, from_networkx, to_dense_adj
from torch_geometric.transforms import NormalizeFeatures, ToDevice, RandomLinkSplit, RemoveDuplicatedEdges
import torch.nn.functional as F



"""### Import the Dataset

Process the Data Frame - Modified Code from - https://github.com/jegraham/csv_to_dataframe_to_graph/blob/master/.idea/csv_to_datadrame_conversion.py
"""

# from google.colab import files


"""## Testing Parameters"""

# Define the root directory where the dataset will be stored
root = './'
version = 'v1'
run_id = 'GAT_1000_k_50_dist_150_250_500_transform'

# File Path
folder_path = f'./results/{run_id}_{version}/'
os.makedirs(folder_path, exist_ok=True)


# Define the Number of Clusters
num_clusters = 4

K = num_clusters
clusters = []

# num_Infrastructure = 10 #The number of RSU and Towers in the Dataset (always at the start of the dataset)
# max_dist_tower = 500 #V2I
# max_dist_rsu = 250 #V2R
# max_dist = 150 #V2V

# Channel Parameters & GAE MODEL
in_channels = 1
hidden_channels = 20
out_channels = 1

# Transform Parameters
transform_set = True

# Optimizer Parameters (learning rate)
learn_rate = 0.0001

# Epochs or the number of generation/iterations of the training dataset
# epoch and n_init refers to the number of times the clustering algorithm will run different initializations
epochs = 500
n = 1000

"""# Run GNN

## InMemory Dataset

Convert Dataset to same format as Planetoid - https://pytorch-geometric.readthedocs.io/en/latest/tutorial/create_dataset.html
"""

count_0 = [0]*30
count_1 = [0]*30
count_2= [0] *30
count_3 = [0]*30

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
        hidden_channels2 = 10
        hidden_channels3=5
        hidden_channels4 =3
        self.conv1 = GATConv(in_channels, hidden_channels, heads=self.in_head, dropout=0.6)
        self.conv2 = GATConv(hidden_channels*self.in_head,hidden_channels2)
        self.conv3 = GATConv(hidden_channels2,hidden_channels3)
        self.conv4 = GATConv(hidden_channels3,hidden_channels4)
        self.conv5 = GATConv(hidden_channels4, out_channels, concat=False)#, heads=self.out_head, dropout=0.6)


    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x,edge_index).relu()
        x = self.conv3(x, edge_index).relu()
        x = self.conv4(x, edge_index).relu()
        x = self.conv5(x,edge_index)
        return x

def train(dt):
    model.train()
    optimizer.zero_grad()
    z = model.encode(dt.x, dt.pos_edge_label_index)
    loss = model.recon_loss(z, dt.pos_edge_label_index)
    loss.backward()
    optimizer.step()
    return float(loss)


def test(dt):
    model.eval()
    with torch.no_grad():
        z = model.encode(dt.x, dt.pos_edge_label_index)

    return model.test(z, dt.pos_edge_label_index, dt.neg_edge_label_index)
for ko in range(30):

    weight_edge,row_indices,col_indices=weight()
    # エッジのインデックス（ノード間の接続）を定義
    #row_indices = torch.tensor([0, 0, 0, 0,0,1,1,1,1,1,1,1,2,3,4,4,4,4,5,5,6,6,7,7,8,8,8,8,8,8,8,9,9,9,9,9,9,9,10,10,10,11,11,11,12,12,13,13,13,14,14,14,15,15,15,16,16,16,16,16,16,17,17,17,18,18,18,18,19,19,19,19,20,20,20,20,21,21,21,21,22,22,22,22,23,23,23,23,28,28,28,28,28,28,28,28,29,29,29,29,29,29,30,30,30,30])
    #col_indices = torch.tensor([13,14,15,16,17,10,11,12,13,14,15,17,15,16,10,11,13,15,13,17,16,17,11,14,10,11,13,14,15,16,17,10,11,12,14,15,16,17,19,20,21,19,20,21,22,23,18,22,23,18,22,23,19,20,21,18,19,20,21,22,23,19,20,21,24,25,26,27,24,25,26,27,24,25,26,27,24,25,26,27,24,25,26,27,24,25,26,27,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27])

    # 各エッジの重みを定義
    values = weight_edge
    #values =x
    # 行と列のインデックスを組み合わせて、疎行列のインデックスを作成
    print(row_indices.shape,col_indices.shape)
    tensor_data= torch.stack([row_indices, col_indices],dim=0)

    result = [[tensor_data[0, i].item(), tensor_data[1, i].item()] for i in range(tensor_data.shape[1])]
    edge_index=result
    print(result)
    edge_index=torch.tensor(edge_index)

    edge_attr=weight_edge

    np.random.seed(0000)
    x=np.random.rand(263,1)
    #x=np.zeros_like(a)
    x=torch.tensor(x,dtype=torch.float)
    #x=torch.tensor([[0],[0],[0],[0],[0],[1],[1],[1],[1],[1],[2],[2],[2],[2],[2],[2],[2],[2],[2],[2],[2],[2],[2],[2],[2],[2],[2],[2],[3],[3],[3]],dtype=torch.float)
    print(x)
    #y=torch.tensor([0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])
    # 配列のサイズを指定
    size = 263

    # テンソルをゼロで初期化
    y = torch.zeros(size, dtype=torch.long)

    # 条件に基づいてテンソルの値を設定
    y[0:10]=0
    y[10:20] = 1       # 71~72のインデックスは1
    y[20:30] = 2       # 63~74のインデックスは2

    y[31:263] = 3      # 63~74のインデックスは2


    print(y)


    # visualize_graph(G,color=dataset.y)


    print('==============================================================')

    dataset=Data(x=x,edge_index=edge_index.t(),edge_attr=edge_attr,y=y,num_classes=4)
    Data.train_mask=np.array([0 for i in range(30)])



    data = dataset
    print(dataset)
    G=to_networkx(dataset, to_undirected=True)
    # visualize_graph(G,color=dataset.y,i=1)
    # transform = RemoveDuplicatedEdges()
    # data = transform(data)


    transform = RandomLinkSplit(
        num_val=0.05,
        num_test=0.15,
        is_undirected=False,
        split_labels=True,
        add_negative_train_samples=True)

    train_data, val_data, test_data = transform(data)

    # Display Graphs
    print(f'Number of graphs: {len(dataset)}')
    print('dataset',dataset) ## dataset is vector with size 1 because we have one graph

    print(f'Number of features: {dataset.num_features}')
    print('------------')

    # Print information for initialization
    print('data', data)
    print('train data',train_data)
    print('valid data', val_data)
    print('test data', test_data)
    print('------------')

    # print(data.is_directed())

    """## Build Graph for Visualization

    ### Visualize Entire Data
    """


    G = to_networkx(data)
    G = G.to_directed()

    X = data.x[:,[0]].cpu().detach().numpy()
    pos = dict(zip(range(X[:, 0].size), X))


    # Draw the Graph
    # fig, ax = plt.subplots(figsize=(10, 10))
    # ax.scatter(X[:,0], data.y,s=20, color='grey')
    # nx.draw_networkx_nodes(G, pos, node_color='black', node_size=20, ax=ax)
    # nx.draw_networkx_edges(G, pos, edge_color='grey', ax=ax)
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # plt.savefig(f'{folder_path}{run_id}_{version}-initial-graph', format='eps', dpi=300)


    """### Define the Encoder
    Change the Encoder based on the type testing against
    """

  

    """### Define the Autoencoder


    """

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Initialize the Model
    model = GAE(GCNEncoder(in_channels, hidden_channels, out_channels))

    model = model.to(device)
    train_data = train_data.to(device)
    test_data = test_data.to(device)
    data_ = data.to(device)

    # Inizialize the Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr = learn_rate)
    print(model)


    auc_values=[]
    ap_values =[]

    best_auc = 0.0  # Track the best AUC value
    consecutive_epochs = 0  # Track the number of consecutive epochs with AUC not increasing
    best_ap = 0.0

    import matplotlib.pyplot as plt

    # 各エポックのlossとAUCの値を保存するリスト
    loss_values = []
    auc_values = []
    accuracy=[]
    best_label=[]
    best_loss= 100000000000000
    for epoch in range(1, epochs + 1):
        acc=0
        # 訓練データでのlossを取得
        loss = train(train_data)
    
        loss_values.append(loss)

        # テストデータでのAUCとAPを取得
        auc, ap = test(test_data)
        auc_values.append(auc)
        ap_values.append(ap)

        # 各エポックの結果を表示
        # print('Epoch: {:03d}, Loss: {:.4f}, AUC: {:.4f}, AP: {:.4f}'.format(epoch, loss, auc, ap))

        # 100エポックごとに表示
        # if (epoch % 100 == 0):
        print('Epoch: {:03d}, Loss: {:.4f}, AUC: {:.4f}, AP: {:.4f}'.format(epoch, loss, auc, ap))
        model.eval()
        with torch.no_grad():
            z = model.encode(data_.x, data_.edge_index)
        z = z.cpu().detach().numpy()
        

        
        # gnn_kmeans = KMeans(n_clusters=num_clusters, n_init=n).fit(z)
        # gnn_labels = gnn_kmeans.labels_

        # from sklearn import cluster
        # # SVMの分類器を訓練
        # spkm = cluster.SpectralClustering(n_clusters=num_clusters,affinity="rbf",assign_labels='discretize')
        # res_spkm = spkm.fit(z)
        # gnn_labels = res_spkm.labels_

        # from sklearn.cluster import AffinityPropagation
        # # SVMの分類器を訓練
        # spkm = AffinityPropagation()
        # res_spkm = spkm.fit(z)
        # gnn_labels = res_spkm.labels_

        from sklearn import cluster
        # SVMの分類器を訓練
        spkm = cluster.AgglomerativeClustering(n_clusters=num_clusters,metric='manhattan', linkage='complete')
        res_spkm = spkm.fit(z)
        gnn_labels = res_spkm.labels_

        # from sklearn import cluster
        # # SVMの分類器を訓練
        # spkm = cluster.DBSCAN()
        # res_spkm = spkm.fit(z)
        # gnn_labels = res_spkm.labels_

        if best_loss > loss:
            best_loss = loss
            best_label=gnn_labels
            best_epoch=epoch
        count=0
        for i in range(30): 
            if gnn_labels[i]==dataset.y[i]:
                count += 1
        acc=count/30
        
        accuracy.append(count/30)
        # print(count/2)
        # print(gnn_labels)
        if acc > 0.5:
            print('acc 50%',gnn_labels,epoch,acc)
            # break
        # Early stoppingの条件確認
        if (auc >= (best_auc - 0.01 * best_auc)) and (ap >= (best_ap - 0.01 * best_ap)):
            if (auc >= 0.8):
                best_auc = auc
                consecutive_epochs = 0
            if (ap >= 0.5):
                best_ap = ap
                consecutive_epochs = 0
            if (ap >= 0.5) and (auc >= 0.8):
                print("AUC and AP Over GOOD value")
                print(gnn_labels,epoch)
                break
        else:
            consecutive_epochs += 1

        if (consecutive_epochs >= 10):
            print('Early stopping: AUC and AP have not increased by more than 1% for 10 epochs.')
            print(gnn_labels,epoch)
            break
    # visualize_graph(G,color=best_label,i=0)
    print(epoch,gnn_labels,loss)
    print(best_label,best_epoch,best_auc,best_loss)


    for i in range(30): 
        if gnn_labels[i]== 0:
            print(i)
            count_0[i] += 1
        if gnn_labels[i]== 1:
            
            count_1[i] += 1
        if gnn_labels[i]== 2:
            count_2[i] += 1
        if gnn_labels[i] == 3:
            count_3[i] += 1

print(count_0,count_1,count_2,count_3)
# 訓練終了後にlossとAUCをプロット
plt.figure()




# Lossのプロット
plt.subplot(2, 1, 1)
plt.plot(range(1, len(loss_values) + 1), loss_values, label='loss')
plt.xlabel('Epoch')
plt.ylabel('LOss')
plt.title('Loss per Epoch')
plt.legend()
plt.savefig(f'acc_loss/{name}_rbf_loss.png')  # 保存
plt.close()




# from torch_geometric.nn.conv.gcn_conv import GCNConv

# class Net(torch.nn.Module):
#     def __init__(self):
#         super(Net,self).__init__()
#         hidden_size = 20
       
#         self.conv1=GCNConv(dataset.num_node_features,hidden_size)
#         self.conv2=GCNConv(hidden_size,25)
       
#         self.conv3=GCNConv(25,20)
#         self.conv4=GCNConv(20,10)

#         self.linear=torch.nn.Linear(10,dataset.num_classes)

#     def forward(self,data):
#         x = data.x
       
#         edge_index = data.edge_index

#         x = self.conv1(x,edge_index)
#         x=F.relu(x)

#         x=self.conv2(x,edge_index)
#         x=F.relu(x)
#         # x=self.conv7(x,edge_index)
#         # x=F.relu(x)
#         x=self.conv3(x,edge_index)
#         x=F.relu(x)
#         x=self.conv4(x,edge_index)
#         x=F.relu(x)
       
        
   
#         x=self.linear(x)
#         return x

# # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# # model =Net()
# # model.to(device)
# # dataset.to(device)
# # model.train()

# model =Net()
# model.to('cpu')
# dataset.to('cpu')
# model.train()
# optimizer=torch.optim.Adam(model.parameters(),lr=0.003)
# # scheduler = optim.lr_scheduler.CosineAnnealingLR(
# #     optimizer,
# #     T_max = 4000
# # )
# loss_func = torch.nn.CrossEntropyLoss()
# losses=[]
# acces=[]
# for epoch in range(1500):
#     optimizer.zero_grad()
#     dataset.to('cpu')
#     out = model(dataset)
#     loss = loss_func(out,dataset.y)
#     losses.append(loss)
#     loss.backward()

#     optimizer.step()

    

#     model.eval()

#     _,pred = model(dataset).max(dim=1)
#     # scheduler.step()
 
#     predict=pred.cpu()
#     data_y=dataset.y.cpu()
#     count=0
#     for i in range(len(predict)): 
#         if predict[i]==data_y[i]:
#             count += 1
#     acces.append(count/len(data_y))
#     print('Epoch %d | Loss: %.4f | ACC: %.4f' % (epoch,loss.item(),count/len(data_y)))
# print("結果：",predict)
# print("真値：",data_y)
# # visualize_graph(G,color=predict)
# # リスト内の各テンソルをdetachしてnumpy配列に変換
# losses_np = [loss.detach().numpy() for loss in losses]
# # acces_np=   [acc.detach().numpy() for acc in acces]
# import matplotlib.pyplot as plt
# plt.figure(figsize=(10, 5))
# plt.plot(losses_np, label='Training Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')

# plt.title('Loss vs. Epoch')
# plt.legend()
# plt.grid(True)
# plt.savefig('Graph Neural Network')
# plt.close()
# plt.figure(figsize=(10, 5))
# plt.plot(acces, label='Accracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accracy')
# # plt.ylim(0,0.8)
# plt.title('Accracy GNN')
# plt.legend()
# plt.grid(True)
# plt.savefig('GNN_ACC')
# plt.close()