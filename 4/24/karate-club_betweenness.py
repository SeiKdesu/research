import networkx as nx
import matplotlib.pyplot as plt

G=nx.karate_club_graph()

output = nx.communicability_betweenness_centrality(G)
max_output=max(list(output.keys()),key=lambda val:output[val])
print("betweenness",str(max_output))
degree_centers = nx.degree_centrality(G)
print('次数中心性',sorted(degree_centers.items(), key=lambda x: x[1], reverse=True)[:8])

close_centers = nx.closeness_centrality(G)
print('近傍中心性',sorted(close_centers.items(), key=lambda x: x[1], reverse=True)[:8])

between_centers = nx.betweenness_centrality(G)
print('媒介中心性',sorted(between_centers.items(), key=lambda x: x[1], reverse=True)[:8])

eigen_centers = nx.eigenvector_centrality_numpy(G)
print('固有ベクトル中心性',sorted(eigen_centers.items(), key=lambda x: x[1], reverse=True)[:8])