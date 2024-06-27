
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

G=nx.Graph()
G.add_node('X')
G.add_node('A')
G.add_node('B')
G.add_node('C')

G.add_edge('X','A',weight=2)
G.add_edge('X','B',weight=1)
G.add_edge('A','C',weight=1)
G.add_edge('B','C',weight=1)
weights=nx.get_edge_attributes(G,'weight').values()
nx.draw(G,with_labels=True,edge_color=weights)
plt.savefig('function1.pdf')
degree_centers = nx.degree_centrality(G)
print('次数中心性',sorted(degree_centers.items(), key=lambda x: x[1], reverse=True)[:5])

close_centers = nx.closeness_centrality(G)
print('近傍中心性',sorted(close_centers.items(), key=lambda x: x[1], reverse=True)[:5])

between_centers = nx.betweenness_centrality(G)
print('媒介中心性',sorted(between_centers.items(), key=lambda x: x[1], reverse=True)[:5])

eigen_centers = nx.eigenvector_centrality_numpy(G)
print('固有ベクトル中心性',sorted(eigen_centers.items(), key=lambda x: x[1], reverse=True)[:5])