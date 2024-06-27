import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

G= nx.karate_club_graph()

between_cent=nx.communicability_betweenness_centrality(G)

node_size=[10000*size for size in between_cent.values()]
plt.figure(figsize=(20,20))
nx.draw_networkx(G,node_size=node_size)
plt.savefig('nodesize.pdf')