import torch
from torch_geometric.data import Data
import numpy as np
from icecream import ic
from rosenbrock_nn import weight_return


def preprocesing(matrix):
    tmp = torch.tensor(matrix,dtype=torch.float)
    tmp = tmp.transpose(0,1)
    fl_tmp = tmp.flatten()
    fl_tmp = fl_tmp.unsqueeze(0)

    # 各要素をリストとして変換
    formatted_tensor = [[torch.tensor([value]) for value in row] for row in fl_tmp]
    formatted_tensor = torch.tensor(formatted_tensor)
    reshaped_tensor = formatted_tensor.flatten().unsqueeze(1)
    return reshaped_tensor

dim = 6

def load_my_dataset(use_degree_as_tag=False):

    src=[]
    dst=[]
    hidden1 = 12
    hidden2 = 6
    for j in range(dim):
        for i in range(hidden1):
            src.append(j)
    for j in range(dim ,hidden1+dim):
        for i in range(hidden2):
            src.append(j)

    # for i in range(dim,12+dim):
    #     src.append(i)

    for j in range(dim):
        for i in range(dim,hidden1+dim):
            dst.append(i)
    for j in range(hidden1):
        for i in range(hidden1+dim,hidden1+hidden2+dim):
            dst.append(i)
    # for i in range(12):
    #     dst.append(12+dim)
    ic(len(src))
    ic(len(dst))
    edge_index=torch.tensor([src,dst],dtype=torch.long)
    ic(edge_index.shape)
    node = hidden1+hidden2+dim
    unit_vectors = [[1 if i == j else 0 for i in range(node)] for j in range(node)]
    for vector in unit_vectors:
        ic(vector)
    ic(unit_vectors)
    x = torch.tensor(unit_vectors,dtype=torch.float)
    y_tmp=[]
    for i in range(3):
        y_tmp.append(0)
    for i in range(3,6):
        y_tmp.append(1)

    # for i in range(3):
    #     y_tmp.append(1)
    for i in range(hidden1+hidden2):
        y_tmp.append(2)

    y = torch.tensor(y_tmp)


    a = weight_return()
    ic(a[1].shape)
    formatted_tensor = []
    ic(preprocesing(a[0]))
    formatted_tensor.append(preprocesing(a[0]))
    formatted_tensor.append(preprocesing(a[1]))
    result_tensor = torch.cat(formatted_tensor,dim =0 )
    ic(result_tensor.shape)
    edge_attr = result_tensor
    dataset=Data(x=x,edge_index=edge_index,edge_attr=edge_attr,y=y,num_classes=4)
    Data.train_mask=np.array([0 for i in range(dim)])
    # 複数のグラフを持つリスト
    graphs = [dataset]
    
    num_classes = 2  # クラス数
    return dataset


def separate_mydata(graphs, fold_idx):
    # 例: 80%を訓練データ、20%をテストデータに分ける
    train_graphs = graphs[:int(len(graphs) * 0.8)]
    test_graphs = graphs[:int(len(graphs) * 0.2):]
    
    return train_graphs, test_graphs