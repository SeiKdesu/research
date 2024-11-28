import torch
from icecream import ic
from rosenbrock_nn import *
import torch 
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils import  to_networkx 
dim = 6

def visualize_graph(G,color):
    plt.figure(figsize=(3, 3))
    plt.xticks([])
    plt.yticks([])

    # ノードの範囲ごとに縦1列に配置するための位置を設定
    pos = {}

    # 各範囲ごとにノードを縦1列に並べる
    ranges = [list(range(dim)),list(range(dim,hidden1+dim)),list(range(hidden1+dim,hidden1+hidden2+dim))]
    ic(ranges)
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
        plt.savefig(f'acc_loss/rosencrok_teacher_.png')  # 保存

    plt.savefig(f'acc_loss/rosencrok_predict_.png')  # 保存
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
ic(dataset)

G = to_networkx(dataset)
G = G.to_directed()
visualize_graph(G,color=dataset.y)
