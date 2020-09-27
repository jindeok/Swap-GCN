from random import randrange
import networkx as nx
from networkx import to_numpy_matrix, degree_centrality, betweenness_centrality, shortest_path_length
import copy
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import numpy as np

from numpy import arange
from functools import reduce

import dgl
import dgl.function as fn
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph




class ReconNet(torch.nn.Module):
    
    def __init__(self, D_in, H1, H2, D_out):
        super(ReconNet, self).__init__()
        
        self.linear1 = nn.Sequential(nn.Linear(D_in, H1), nn.ReLU())
        self.linear2 = nn.Sequential(nn.Linear(H1, H2), nn.ReLU())
        self.linear3 = nn.Sequential(nn.Linear(H2, D_out), nn.Sigmoid())

    def forward(self, x):

        h_relu = self.linear1(x)
        h2_relu = self.linear2(h_relu)
        y_pred = self.linear3(h2_relu)
        
        return y_pred
 
#dgl implementation GCN
#gcn_msg = fn.copy_src(src='h', out='m')
#gcn_reduce = fn.sum(msg='m', out='h')
#   
#class GCNLayer(nn.Module):
#    def __init__(self, in_feats, out_feats):
#        super(GCNLayer, self).__init__()
#        self.linear = nn.Linear(in_feats, out_feats)
#
#    def forward(self, g, feature):
#        # Creating a local scope so that all the stored ndata and edata
#        # (such as the `'h'` ndata below) are automatically popped out
#        # when the scope exits.
#        with g.local_scope():
#            g.ndata['h'] = feature
#            g.update_all(gcn_msg, gcn_reduce)
#            h = g.ndata['h']
#            return self.linear(h)
#        
#class GCN(nn.Module):
#    def __init__(self, Adj):
#        super(GCN, self).__init__()
#        self.gcnlayer1 = GCNLayer(Adj.shape[0], 64)
#        self.gcnlayer2 = GCNLayer(64, 32)
#        self.gcnlayer3 = GCNLayer(32, 32)
#
#    def forward(self, g, features):
#        x = F.relu(self.gcnlayer1(g, features))
#        x = self.gcnlayer2(g, x)
#        x = self.gcnlayer3(g, x)
#        
#        return x
        
        
####################################    
###### matrix  multiplication GCN 
###################################
        #  원본
# class GcnLayer(nn.Module):
#     def __init__(self, A, in_units, out_units, activation='relu'):
#         super(GcnLayer, self).__init__()
#         I = np.eye(*A.shape)
#         A_hat = A.copy() + I

#         D = np.sum(A_hat, axis=0)
#         D_inv = D ** -0.5
#         D_inv = np.diag(D_inv)

#         self.A_hat = (D_inv * A_hat * D_inv).astype('float32')
#         self.A_hat = torch.tensor(self.A_hat)

#         self._fc = nn.Sequential(nn.Linear(in_units, out_units), nn.Tanh())

#     def forward(self, x):
#         x = torch.matmul(self.A_hat, x)
#         x = self._fc(x)
#         return x
        
# class GCN(nn.Module):
#     def __init__(self, A, input_dim, hidden_dim_list):
#         super(GCN, self).__init__()
#         self.input_dim = input_dim
#         self.hidden_dim_list = hidden_dim_list
#         self.gcn1 = GcnLayer(A, self.input_dim, hidden_dim_list[0])
#         self.gcn2 = GcnLayer(A, hidden_dim_list[0], hidden_dim_list[1])
#         self._fc = nn.Sequential(nn.Linear(hidden_dim_list[1], A.shape[0]), nn.Sigmoid())
#         #self._fc = nn.Sequential(nn.Linear(hidden_dim_list[2], A.shape[0]), nn.Softmax(dim=1))

#     def forward(self, x):
#         x = self.gcn1(x)
#         x = self.gcn2(x)
#         x = self._fc(x)
#         return x

 
class GcnLayer(nn.Module):
    def __init__(self, in_units, out_units, activation='relu'):
        super(GcnLayer, self).__init__()

        self._fc = nn.Sequential(nn.Linear(in_units, out_units), nn.Tanh())

    def forward(self, A, x):
        
        I = np.eye(*A.shape)
        A_hat = A.copy() + I

        D = np.sum(A_hat, axis=0)
        D_inv = D ** -0.5
        D_inv = np.diag(D_inv)

        self.A_hat = (D_inv * A_hat * D_inv).astype('float32')
        self.A_hat = torch.tensor(self.A_hat)
        
        x = torch.matmul(self.A_hat, x)
        x = self._fc(x)
        return x


class GCN(nn.Module):
    def __init__(self, max_A, input_dim, hidden_dim_list):
        super(GCN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim_list = hidden_dim_list
        self.gcn1 = GcnLayer(self.input_dim, hidden_dim_list[0])
        self.gcn2 = GcnLayer(hidden_dim_list[0], hidden_dim_list[1])
        self._fc = nn.Sequential(nn.Linear(hidden_dim_list[1], max_A.shape[0]), nn.Sigmoid())
        #self._fc = nn.Sequential(nn.Linear(hidden_dim_list[2], A.shape[0]), nn.Softmax(dim=1))

    def forward(self, A, x):
        x = self.gcn1(A, x)
        x = self.gcn2(A, x)
        x = self._fc(x)
        return x




    
def sinkhorn(log_alpha, n_iters = 20, temp = 0.01):
    # torch version
    log_alpha = log_alpha / temp
    n = log_alpha.size()[1]
    log_alpha = log_alpha.view(n, n)

    for i in range(n_iters):

        log_alpha = log_alpha - (torch.logsumexp(log_alpha, dim=1, keepdim=True)).view(n, 1)
        log_alpha = log_alpha - (torch.logsumexp(log_alpha, dim=0, keepdim=True)).view(1, n)
        
    return torch.exp(log_alpha)