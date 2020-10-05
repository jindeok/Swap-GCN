## Swap training Implementation -Pytorch version

import networkx as nx 
import numpy as np
import random as rnd
from random import sample
from random import randrange
from Masking import GraphMasking
from createdata import *
import pandas as pd
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from numpy import arange
from functools import reduce
from perm_swapmodel import ReconNet,GCN,sinkhorn

import time
import seaborn as sns
import copy
from stats import degree_stats
np.random.seed(1)
exp_iter = 1
mmd_refer = []
mmd_model = []
for k in range(exp_iter):
        
    #####################################################
    ## create data, data processing for model training ##
    #####################################################
    graph_name = 'grid'
    
    X_train = create_graphs(graphtype = graph_name)
    X_ref = copy.deepcopy(X_train)
    X_train_copy = copy.deepcopy(X_train)
    
    delportion = 0.2
    
    Y1,Y2 = GenerateMaskedPair(X_train, X_train_copy, delportion)
    
    A1_list, A2_list, Pgt_list = graph_pairs_to_mat(Y1,Y2)
     
    maxnumnode = maxnode(A1_list)
    
    ############################
    ## create model instances ##
    ############################
    
    feature = np.identity(maxnumnode).astype('float32')
    ts_feature= torch.from_numpy(feature)
    permmodel = GCN(A1_list[0], feature.shape[1], [256, 256, 128]) #A1_list[0] -> 가장 큰 A1으로 대체 추후
    
    #Setting loss, optimizer
    criterion = torch.nn.MSELoss(reduction='sum')
    permoptimizer = torch.optim.Adam(permmodel.parameters(), lr=0.5e-3)
    
    
    ####training##########
    epochs = 1200
    
    for t in range(epochs):
        for i in range(len(A1_list)):
            #만약 그래프 1개로 쓸거면 이하 부분은 반복문 밖으로 빼기
            A1_np = np.array(A1_list[i])
            A1_np = A1_np.astype('float32')
            A2_np = np.array(A2_list[i])
            A2_np= A2_np.astype('float32') #이부분 나중에 루프 밖으로 빼기
            
            A1 = torch.from_numpy(A1_np) 
            A2 = torch.from_numpy(A2_np)    
            
            #proxy training    
    
            outgcn = permmodel(A1_np, ts_feature)
            P_hat = sinkhorn(outgcn, n_iters= 20, temp= 0.01)       
            temp = torch.matmul(P_hat, A1)   
            P_hat_inv = torch.transpose(P_hat, 0, 1)
            #permuted A1
            A1_prime = torch.matmul(temp, P_hat_inv) 
            #optimizer
            loss_perm1 = criterion(A1_prime, A2)
            
            ## hot wo adopt anchor nodes for permutation learning?
            P_gt =torch.from_numpy(Pgt_list[i].astype('float32'))
            one = torch.ones(1)
            loss_anchor = criterion(P_hat[0,3], one) + criterion(P_hat[3,2], one) + criterion(P_hat[7,1], one)
            
            loss_perm = loss_perm1 + loss_anchor
            
            permoptimizer.zero_grad()                
            loss_perm.backward()
            permoptimizer.step()
    
            if t % 300 == 0:
                print( t, "perm loss:",loss_perm.item())
                
    
    ####test the results##########
    #for only 1 graph -> to be revised
    # actual permutation
    P_hat_np = P_hat.clone().detach().numpy()            
    prune_permutation = matching(P_hat_np)  # hungarian assignment algorithm
    test_p = np.zeros((A1_np.shape[0],A1_np.shape[0]))
    for i in range(len(prune_permutation[0])):
        temp = prune_permutation[0][i]
        test_p[i][temp] = 1
    # apply actual permutation
    A1_prime_np = A1_prime.clone().detach().numpy()
    A1_prime_np = np.dot(np.dot(test_p,A1),test_p.T)
    
    # G_test = nx.from_numpy_array(A1_prime_np)
    # nx.draw(G_test)
        
            
    theta1, theta2 = extract_masking_set(A1_prime_np,A2_np)        
    impute1 = 1 - theta1
    impute2 = 1 - theta2
    
    A1_comp = A1_prime_np + impute1
    A2_comp = A2_np + impute2
    
    G_test = []
    G_test.append(nx.from_numpy_array(A1_comp))
       
    mmd_comp = degree_stats(X_ref, Y1)
    mmd_refer.append(mmd_comp)
    print("ref_mmd is:",mmd_comp)
    mmd_score = degree_stats(X_ref,G_test)
    mmd_model.append(mmd_score)
    print("completion degree mmd score is : " , mmd_score )
    
print("-----------------------------------------------------")  
print("deletion portion : {}% ".format(delportion*100))
print("AVG ref_mmd over iter {} is: {}".format( exp_iter, sum(mmd_refer)/exp_iter))    
print("AVG model_mmd over iter {} is: {}".format( exp_iter, sum(mmd_model)/exp_iter))
nx.draw(G_test[0])

