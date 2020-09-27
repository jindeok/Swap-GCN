## Deletion module practice
## Not yet implemented on the project


import networkx as nx 
import numpy as np
import random as rnd
from random import sample


def GraphMasking(Graph, method = 'blockwise', walklength = 1, spreadpwr = 1, startpos = (0,0), portion = 0.1):
    
    nodelist = list(Graph.nodes)
    edgelist = list(Graph.edges)
    
    
    # 딜리션 리스트 추출 numMask개
    if method == 'random sampling':
        
        numMask = int(len(nodelist) * portion) #deletion할 개수 선택
        delnodelist = sample(list(nodelist), numMask)
        Graph.remove_nodes_from(delnodelist)
        #print(delnodelist)
        
    # node가 (x,y) 로 표현되기떄문에, delnodelist가 remove 되지 않아. grid에서 순차적으로 ordering하는방법이 필요할 듯 싶다.
    elif method == 'blockwise':
        
        W_vi = []
        W_vi.append(startpos)
        
        for i in range(walklength-1):
            
            samplecand = list(G[W_vi[i]]) #sample된 후보들
            if spreadpwr <= len(samplecand):
                sampled = rnd.sample(list(samplecand), spreadpwr)  
            else:
                sampled = rnd.sample(list(samplecand), len(samplecand))
            W_vi.extend(sampled)      
            
        delnodelist = list(W_vi)   
        Graph.remove_nodes_from(delnodelist)
        
    elif method == 'random edge sampling':
        
        numMask = int(len(edgelist) * portion)
        deledgelist = sample(list(edgelist), numMask)
        Graph.remove_edges_from(deledgelist)

    return Graph

#G = nx.grid_2d_graph(3,3)
#nx.draw(G)
#a1 = nx.adj_matrix(G)
#a2 = nx.to_numpy_matrix(G)
#G1 = GraphMasking(G, method = 'blockwise',  startpos = (1,2), walklength = 1, spreadpwr = 1)
#G2 = GraphMasking(G, method = 'random sampling', portion = 0.1)

#G3 = GraphMasking(G, method= 'random edge sampling', portion = 0.1)

#nx.draw(G3)
#a2 = nx.adj_matrix(G3)
#print(a2)