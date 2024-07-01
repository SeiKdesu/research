
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

G=nx.Graph()
G.add_node('1')
G.add_node('2')
G.add_node('3')
G.add_node('4')
G.add_node('5')
G.add_node('6')

G.add_edge('1','2',weight=1)
G.add_edge('1','3',weight=1)
G.add_edge('1','4',weight=1)
G.add_edge('1','5',weight=1)
G.add_edge('2','4',weight=1)
G.add_edge('3','6',weight=1)
G.add_edge('6','5',weight=1)
weights=nx.get_edge_attributes(G,'weight').values()
nx.draw(G,with_labels=True,edge_color=weights)
plt.savefig('powerpoint.pdf')
degree_centers = nx.degree_centrality(G)
print('次数中心性',sorted(degree_centers.items(), key=lambda x: x[1], reverse=True)[:5])

close_centers = nx.closeness_centrality(G)
print('近傍中心性',sorted(close_centers.items(), key=lambda x: x[1], reverse=True)[:5])

between_centers = nx.betweenness_centrality(G)
print('媒介中心性',sorted(between_centers.items(), key=lambda x: x[1], reverse=True)[:5])

eigen_centers = nx.eigenvector_centrality_numpy(G)
print('固有ベクトル中心性',sorted(eigen_centers.items(), key=lambda x: x[1], reverse=True)[:5])