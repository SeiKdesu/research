import networkx as nx
import matplotlib.pyplot as plt

G=nx.karate_club_graph()

output=nx.eigenvector_centrality_numpy(G)

max_output=max(list(output.keys()),key=lambda val:output[val])
print("betweenness",str(max_output))