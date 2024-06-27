import networkx as nx
import matplotlib.pyplot as plt

G=nx.karate_club_graph()
pos=nx.spring_layout(G)

color_list=[0 if G.node[i]["club"] == "Mr.Hi" else 1 for i in G.nodes()]
nx.draw_networkx(G,pos,node_color=color_list,cmap=plt.cm.RdYlBu)

plt.savefig('karate_Mr.Hi.pdf')