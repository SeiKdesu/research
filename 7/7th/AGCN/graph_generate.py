import torch
from rastrigin_and_schwefl_bias_node import weight
import numpy as np
# ノードの数を設定
def graph_data_make():
        
    num_nodes = 31
    weight_edge,row_indices,col_indices=weight()
    # エッジのインデックス（ノード間の接続）を定義
    #row_indices = torch.tensor([0, 0, 0, 0,0,1,1,1,1,1,1,1,2,3,4,4,4,4,5,5,6,6,7,7,8,8,8,8,8,8,8,9,9,9,9,9,9,9,10,10,10,11,11,11,12,12,13,13,13,14,14,14,15,15,15,16,16,16,16,16,16,17,17,17,18,18,18,18,19,19,19,19,20,20,20,20,21,21,21,21,22,22,22,22,23,23,23,23,28,28,28,28,28,28,28,28,29,29,29,29,29,29,30,30,30,30])
    #col_indices = torch.tensor([13,14,15,16,17,10,11,12,13,14,15,17,15,16,10,11,13,15,13,17,16,17,11,14,10,11,13,14,15,16,17,10,11,12,14,15,16,17,19,20,21,19,20,21,22,23,18,22,23,18,22,23,19,20,21,18,19,20,21,22,23,19,20,21,24,25,26,27,24,25,26,27,24,25,26,27,24,25,26,27,24,25,26,27,24,25,26,27,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27])
    np.random.seed(1234)
    x=np.random.rand(31,1)
    #x=np.zeros_like(a)
    x=torch.tensor(x,dtype=torch.float)
    # 各エッジの重みを定義
    values = weight_edge
    #values =x
    # 行と列のインデックスを組み合わせて、疎行列のインデックスを作成
    indices = torch.stack([row_indices, col_indices],dim=0)
    #indices=indices.squeeze() 
    print(values.shape)
    #values=values.squeeze()
    # 疎行列を生成
    sparse_matrix = torch.sparse_coo_tensor(indices, values, (num_nodes, num_nodes))

    # 疎行列の情報を1つの辞書にまとめる
    # graph_info = {
    #     'indices': indices,
    #     'values': values,
    #     'size': sparse_matrix.size(),
    #     'nnz': sparse_matrix._nnz(),
    #     'layout': sparse_matrix.layout,
    #     'sparse_matrix': sparse_matrix  # 実際の疎行列も含める
    # }
    
    sparse_matrix = torch.tensor(sparse_matrix)
    # 結果を表示
    print('This is the shape.',sparse_matrix.shape)
   
    y=np.array([0,0,0,0,0,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,3,3,3])
    return indices,x,y,sparse_matrix
#graph_data_make()