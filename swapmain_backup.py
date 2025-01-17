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



#####################################################
## create data, data processing for model training ##
#####################################################
graph_name = 'grid'

X_train = create_graphs(graphtype = graph_name)
X_ref = copy.deepcopy(X_train)
X_train_copy = copy.deepcopy(X_train)

Y1,Y2 = GenerateMaskedPair(X_train, X_train_copy, delportion = 0.2)
Y1_train_prev,A1_list,G1_list = graphs_to_matrix(Y1)
Y2_train_prev,A2_list,G2_list = graphs_to_matrix(Y2) 

list_for_check_maxsize = [] # for checking maxnum of input graphs
for i in Y1_train_prev:
    temp = i.T
    list_for_check_maxsize.append(temp)    
    
maxnumnode = maxnode(list_for_check_maxsize)

Y1_train = zeropad_for1d(Y1_train_prev, maxnumnode)
Y2_train = zeropad_for1d(Y2_train_prev, maxnumnode)


############################
## create model instances ##
############################
D_in, H1, H2, D_out = maxnumnode**2, 128, 128, maxnumnode**2

recmodel = ReconNet(D_in, H1, H2, D_out)

feature = np.identity(maxnumnode).astype('float32')
ts_feature= torch.from_numpy(feature)
permmodel = GCN(A1_list[0], feature.shape[1], [64, 32, 32]) #A1_list[0] -> 가장 큰 A1으로 대체 추후

#Setting loss, optimizer
criterion = torch.nn.MSELoss(reduction='sum')
#criterion = torch.nn.CrossEntropyLoss(reduction='sum') #argument?
recoptimizer = torch.optim.Adam(recmodel.parameters(), lr=1e-4)
permoptimizer = torch.optim.Adam(permmodel.parameters(), lr=1e-3)


####training##########
epochs_swap = 2000
epochs_prox = 1000
alpha = 0.5 #loss hyperparam
for t in range(epochs):
    for i in range(len(Y1_train)):
        #만약 그래프 1개로 쓸거면 이하 부분은 반복문 밖으로 빼기
        A1_np = np.array(A1_list[i])
        A1_np = A1_np.astype('float32')
        A2_np = np.array(A2_list[i])
        A2_np= A2_np.astype('float32') #이부분 나중에 루프 밖으로 빼기
        
        A1 = torch.from_numpy(A1_np) 
        A2 = torch.from_numpy(A2_np)    
        
        A1_fl = Y1_train[i].flatten()
        A1_fl = A1_fl.astype('float32')
        A2_fl = Y2_train[i].flatten()
        A2_fl = A2_fl.astype('float32')
        A1_fl = torch.from_numpy(A1_fl)
        A2_fl = torch.from_numpy(A2_fl)
        
        if t == 0:
            outgcn = permmodel(A1_np, ts_feature)
            A1_prime = A1 
            A1_prime_np = A1_prime.clone().detach().numpy() 
            A1_prime_fl = A1_prime.flatten()
            
        #proxy training with nested loop
        if t % 200 == 199:     
            for j in range(epochs_prox): #인풋이 여러개가 되면 바뀌어야할듯. 지금은 1개마다 epochs_prox씩 학습됨.
                #forward - infer P , apply to A
                A1_prime_np = A1_prime.clone().detach().numpy()
                outgcn = permmodel(A1_prime_np, ts_feature)
                P_hat = sinkhorn(outgcn, n_iters= 20, temp= 0.01)       
                temp = torch.matmul(P_hat, A1)   
                P_hat_inv = torch.transpose(P_hat, 0, 1)
                #permuted A1
                A1_prime = torch.matmul(temp, P_hat_inv) 
                #optimizer
                loss_perm = criterion(A1_prime, A2)                
                permoptimizer.zero_grad()                
                loss_perm.backward()
                permoptimizer.step()
                #log
                if j % 200 == 0:
                   print("iter:{}, proxy loss:{}, ".format(j,loss_perm.item()))
            # actual permutation
            P_hat = P_hat.clone().detach().numpy()   
            A1_prime_np = A1_prime.clone().detach().numpy() 
            prune_permutation = matching(P_hat)  # hungarian assignment algorithm
            test_p = np.zeros((A1_np.shape[0],A1_np.shape[0]))
            for i in range(len(prune_permutation[0])):
                temp = prune_permutation[0][i]
                test_p[i][temp] = 1
            # apply actual permutation
            A1_prime_np = np.dot(np.dot(test_p,A1_prime_np),test_p.T)
            A1_prime_fl = A1_prime_np.flatten()
            A1_prime_fl = A1_prime_fl.astype('float32')
            A1_prime_fl = torch.from_numpy(A1_prime_fl)
        
        # feed forward      
                  
        # outgcn = permmodel(A1_prime_np, ts_feature)
        # # permuted new A1
        # P_hat = sinkhorn(outgcn, n_iters= 20, temp= 0.01)       
        # temp = torch.matmul(P_hat, A1)   
        # P_hat_inv = torch.transpose(P_hat, 0, 1)
        # A1_prime = torch.matmul(temp, P_hat_inv) #permuted A1
        
        # A1_prime_np = A1_prime.clone().detach().numpy()        
        
        
        rec_pred1 = recmodel(A1_prime_fl) #reconstructed Ai
        rec_pred2 = recmodel(A2_fl)   
              
        theta1, theta2 = extract_masking_set(A1_prime_np,A2_np)
        theta1 = theta1.flatten()
        theta1 = torch.from_numpy(theta1)
        theta2 = theta2.flatten()
        theta2 = torch.from_numpy(theta2)
     
        # 손실을 계산하고 출력합니다.
        loss_self = criterion(rec_pred1*theta1 + rec_pred2*theta2, A1_fl + A2_fl)
        loss_swap = criterion(rec_pred2*theta1 + rec_pred1*theta2, A1_fl + A2_fl)
        loss_swap_sum = loss_swap + alpha*loss_self

        if t % 100 == 0:
            print(t, loss_swap_sum.item())
            
        # 변화도를 0으로 만들고, 역전파 단계를 수행하고, 가중치를 갱신합니다.
        recoptimizer.zero_grad()
        loss_swap_sum.backward()
        recoptimizer.step()
        

####test the results##########
#for only 1 graph -> to be revised
rec_pred1 = rec_pred1.clone().detach().numpy()  
rec_pruned = PredPruning(rec_pred1, thres = 0.5)
results = np.reshape(rec_pruned, (A1_np.shape[0],A1_np.shape[0]))
G_test = nx.from_numpy_array(results)
nx.draw(G_test)
    
        
        
        