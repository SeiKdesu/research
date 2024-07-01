
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
plt.close
G=nx.Graph()
G.add_node('A')
G.add_node('B')
G.add_node('C')
G.add_node('D')
G.add_node('E')
G.add_node('F')

G.add_edge('A','B',weight=3)
G.add_edge('B','C',weight=2)
G.add_edge('B','D',weight=1)
G.add_edge('B','E',weight=5)
G.add_edge('B','F',weight=1)
'''
node_feature_dict={
    'X1':[3],
    'X2':[2],
    'X3':[1],
    'Y1':[1],
    'Y2':[2],
    'Y3':[3]
}'''
weights=nx.get_edge_attributes(G,'weight').values()
nx.draw(G,with_labels=True,edge_color=weights)
plt.savefig('function4.pdf')
degree_centers = nx.degree_centrality(G)
print('次数中心性',sorted(degree_centers.items(), key=lambda x: x[1], reverse=True)[:8])

close_centers = nx.closeness_centrality(G)
print('近傍中心性',sorted(close_centers.items(), key=lambda x: x[1], reverse=True)[:8])

between_centers = nx.betweenness_centrality(G)
print('媒介中心性',sorted(between_centers.items(), key=lambda x: x[1], reverse=True)[:8])

eigen_centers = nx.eigenvector_centrality_numpy(G)
print('固有ベクトル中心性',sorted(eigen_centers.items(), key=lambda x: x[1], reverse=True)[:8])