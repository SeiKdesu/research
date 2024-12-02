import torch
from torch_geometric.data import Data
import networkx as nx
import matplotlib.pyplot as plt
import torch
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from icecream import ic
from rosenbrock_nn import *
# サンプルデータ（前述のコードから取得）
# ノード評価値
x , y ,num_nodes  = train_data()
ic(x)
threshold = 0.5
edge_index = []
for i in range(num_nodes):
    for j in range(num_nodes):
        if i != j and abs(y[i] - y[j]) <= threshold:
            edge_index.append([i, j])
edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

# グラフデータ
data = Data(x=x, edge_index=edge_index, y=y)

# NetworkXグラフに変換
G = nx.Graph()
for i in range(data.num_nodes):
    G.add_node(i, y=data.y[i].item())  # ノード評価値を属性に追加

for i, j in data.edge_index.t().tolist():
    G.add_edge(i, j)

# ノード位置を設定
pos = nx.spring_layout(G)  # スプリングレイアウト

# ノードの色を評価値で表現
node_colors = [G.nodes[i]['y'] for i in G.nodes]

# グラフ描画
plt.figure(figsize=(8, 6))
nodes = nx.draw_networkx_nodes(
    G, pos, node_color=node_colors, cmap=plt.cm.viridis, node_size=500
)
nx.draw_networkx_edges(G, pos, edge_color='gray')
nx.draw_networkx_labels(G, pos, font_size=10)

# カラーバーを追加
cbar = plt.colorbar(nodes)
cbar.set_label('Node Evaluation (y)', fontsize=12)
plt.title("Graph Visualization")
plt.savefig('graph.png')