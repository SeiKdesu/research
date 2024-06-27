
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
plt.close
G=nx.Graph()
G.add_node('X1')
G.add_node('X2')
G.add_node('X3')
G.add_node('Y1')
G.add_node('Y2')
G.add_node('Y3')

G.add_edge('X1','X2',weight=1)
G.add_edge('X2','Y1',weight=1)
G.add_edge('X3','Y1',weight=1)
G.add_edge('Y2','X3',weight=1)
G.add_edge('Y2','Y3',weight=1)

node_feature_dict={
    'X1':[3],
    'X2':[2],
    'X3':[1],
    'Y1':[1],
    'Y2':[2],
    'Y3':[3]
}
weights=nx.get_edge_attributes(G,'weight').values()
nx.draw(G,with_labels=True,edge_color=weights)
plt.savefig('function3.pdf')
degree_centers = nx.degree_centrality(G)
print('次数中心性',sorted(degree_centers.items(), key=lambda x: x[1], reverse=True)[:8])

close_centers = nx.closeness_centrality(G)
print('近傍中心性',sorted(close_centers.items(), key=lambda x: x[1], reverse=True)[:8])

between_centers = nx.betweenness_centrality(G)
print('媒介中心性',sorted(between_centers.items(), key=lambda x: x[1], reverse=True)[:8])

eigen_centers = nx.eigenvector_centrality_numpy(G)
print('固有ベクトル中心性',sorted(eigen_centers.items(), key=lambda x: x[1], reverse=True)[:8])