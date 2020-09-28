#create data for swap model
import networkx as nx 
import numpy as np
from Masking import GraphMasking
from scipy.optimize import linear_sum_assignment
from functools import reduce
import copy
def graphs_to_matrix(graphs, mode = "random", Is_randomstart = True):
    
    mat_list = []
    Adj_list = []
    G_list = []

    for i in graphs:     
        mat = nx.to_numpy_matrix(i)
        
        if mode == "BFS":
            
            if Is_randomstart == True:
                start_idx = np.random.randint(mat.shape[0])
            else:
                start_idx = 0
            #x_idx = np.array(bfs_seq(i, start_idx))
            #mat = mat[np.ix_(x_idx, x_idx)]
            j = nx.convert_node_labels_to_integers(i, first_label=0, ordering='default', label_attribute=None)
            bfs_tree = nx.algorithms.traversal.bfs_tree(j, start_idx)
            mat = mat[np.ix_(bfs_tree,bfs_tree)]
            mat = np.triu(mat, k = 1)
            mat = np.matrix(mat)
            mat_list.append(mat.flatten())  
            Adj_list.append(mat)
            G_list.append(j)
            
        elif mode == "random":

            j = nx.convert_node_labels_to_integers(i, first_label=0, ordering='default', label_attribute=None)
            node_mapping = dict(zip(j.nodes(), sorted(j.nodes(), key=lambda k: np.random.random())))
            j = nx.relabel_nodes(j, node_mapping)
            Adj = nx.to_numpy_matrix(j, nodelist = sorted(j.nodes()))       
            mat_list.append(Adj.flatten())
            Adj_list.append(Adj)
            G_list.append(j)
            
            
        elif mode == "consistent order":
            mat_list.append(mat.flatten())  
            Adj_list.append(mat)
            G_list.append(i)
            
    
    return mat_list, Adj_list, G_list


        

def create_graphs(graphtype = "grid"):
    
    if graphtype == "grid":
        graphs = []
        for i in range(3,4):
            for j in range(3,4):
                graphs.append(nx.grid_2d_graph(i,j))                

    if graphtype == "trigrid":
        graphs = []
        for i in range(3,4):
            for j in range(3,4):
                graphs.append(nx.triangular_lattice_graph(i,j))
                
    if graphtype == "b-a":
        graphs = []
        for i in range(10,11):
            for j in range(2,3): # j should be lower tahn i ( j = # of edges , i = # of nodes )
                graphs.append(nx.barabasi_albert_graph(i,j))
                
    if graphtype == "Karate":
        graphs = []
        graphs.append(nx.karate_club_graph())                
        
                
    return graphs

def GenerateMaskedPair(X_train, X_train_copy, delportion):
    
    Y1_train = []
    Y2_train = []
    
    for i in X_train:
        Gm1 = GraphMasking(i, method= 'random edge sampling', portion = delportion)
        Y1_train.append(Gm1)
        
        
    for j in X_train_copy:
        Gm2 = GraphMasking(j,  method= 'random edge sampling', portion = delportion)
        Y2_train.append(Gm2)        
        
    
    return Y1_train, Y2_train


def SwapDataloader(Y1,Y2, maxnode):

    Y1_train_prev = graphs_to_matrix(Y1)
    Y2_train_prev = graphs_to_matrix(Y2) # return flatten matrix
    #To be delteted
    
    list_for_check_maxsize = [] # for checking maxnum of input graphs
    
        
    for i in Y1_train_prev:
        temp = i.T
        list_for_check_maxsize.append(temp)    
    
   ## zero padding preprocessing
    Y1_train = []
    Y2_train = []
    
    for i in range(len(Y1_train_prev)):    
        
        zeropad = np.zeros((maxnode**2-len(Y1_train_prev[i].T),1))
        Y1_element = np.concatenate((Y1_train_prev[i].T, zeropad))
        Y2_element = np.concatenate((Y2_train_prev[i].T, zeropad))
        Y1_train.append(Y1_element.T)
        Y2_train.append(Y2_element.T)
        
    return Y1_train, Y2_train
        


def generate_rndperm(nodenum):
    
    perm_idx_arr = np.random.permutation(nodenum)
    perm_arr = np.zeros((nodenum,nodenum))
    for i in range(nodenum):
        perm_arr[i,perm_idx_arr[i]] = 1
    
    return perm_arr
    
def matching(matrix):

  def hungarian(x):
    if x.ndim == 2:
      x = np.reshape(x, [1, x.shape[0], x.shape[1]])
    sol = np.zeros((x.shape[0], x.shape[1]), dtype=np.int32)
    for i in range(x.shape[0]):
      sol[i, :] = linear_sum_assignment(-x[i, :])[1].astype(np.int32)
    return sol

  #listperms = tf.py_func(hungarian, [matrix_batch], tf.int32)
  listperms = hungarian(matrix)
  
  return listperms


def zeropad_for1d(adjlist, maxnumnode):
    ## zero padding preprocessing
    Y1_train = []
    
    for i in range(len(adjlist)):
        
        zeropad = np.zeros((maxnumnode**2-len(adjlist[i].T),1))
        Y1_element = np.concatenate((adjlist[i].T,zeropad))
        Y1_train.append(Y1_element.T)
        
    return Y1_train

def maxnode(graphset):    
    temp = len(reduce(lambda w1, w2: w1 if len(w1)>len(w2) else w2, graphset)) 
    return int(np.sqrt(temp))

def extract_masking_set(A1,A2):
    
    disc = A1-A2
    disc2 = A1-A2  
    disc[disc > 0] = 0
    theta1 = np.abs(disc)
    disc2[disc2 < 0] = 0
    theta2 = disc2    

    return 1-theta1, 1-theta2

def PredPruning(result, thres):
    
    adj_result=[]
    for i in result:
        if i <= thres:
            adj_result.append(0)
        elif i >= thres:
            adj_result.append(1)
    adj_result = np.array(adj_result)

    return adj_result
