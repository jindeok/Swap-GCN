import concurrent.futures
from datetime import datetime
from functools import partial
import numpy as np
import networkx as nx
import os
import pickle as pkl
import subprocess as sp
import time
import mmd

PRINT_TIME = False

def degree_stats(graph_ref_list, graph_pred_list):
    ''' Compute the distance between the degree distributions of two unordered sets of graphs.
    Args:
      graph_ref_list, graph_target_list: two lists of networkx graphs to be evaluated
    '''
    sample_ref = []
    sample_pred = []
    # in case an empty graph is generated
    graph_pred_list_remove_empty = [G for G in graph_pred_list if not G.number_of_nodes() == 0]

    prev = datetime.now()
    for i in range(len(graph_ref_list)):
        degree_temp = np.array(nx.degree_histogram(graph_ref_list[i]))
        sample_ref.append(degree_temp)
    for i in range(len(graph_pred_list_remove_empty)):
        degree_temp = np.array(nx.degree_histogram(graph_pred_list_remove_empty[i]))
        sample_pred.append(degree_temp)
    print("len for ref, pred:",len(sample_ref),",",len(sample_pred))
    
    mmd_dist = mmd.compute_mmd(sample_ref, sample_pred, kernel=mmd.gaussian_emd)
    
    elapsed = datetime.now() - prev
    if PRINT_TIME:
        print('Time computing degree mmd: ', elapsed)
        
    return mmd_dist