## Swap training Implementation

import networkx as nx 
import numpy as np
import random as rnd
from random import sample
from random import randrange
import tensorflow as tf
from Masking import GraphMasking
#import keras
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, ZeroPadding1D
import math
from numba import jit
from numpy import arange
from functools import reduce

#Generate Randomtheta when Proxy theta training
def RandomThetaGenerator(pred):
    
    pred = PredPruning(pred, 0.6)
    numnode = int(math.sqrt(len(pred.T)))    
    
    predProxy = pred.reshape(numnode,numnode)
    G = nx.from_numpy_matrix(predProxy)
    I = np.ones((numnode,numnode))
    
    randportion = rnd.uniform(0,1) # It is verified that without matching portion, it still works.
    #randportion =0.1
    Gprox = GraphMasking(G, method = 'random edge sampling', portion = randportion)
    
    GproxMat = nx.to_numpy_matrix(Gprox)
    randtheta = I - GproxMat    
    
    return randtheta.flatten()


def RstToBinary(result, numnode):    

    adj_result=[]
    for i in result[0]:
        if i <= 0:
            adj_result.append(0)
        elif i >= 1:
            adj_result.append(1)
        else:            
            a = np.random.binomial(n=1, p= i, size=1)
            adj_result.append(a[0])
            
    adj_result = np.array(adj_result)
    adj_result = adj_result.reshape(numnode,numnode)
    w = np.triu(adj_result, k = 1)
    result = w + w.T    
            
    return result


def PredPruning(result, thres):
    
    adj_result=[]
    for i in result[0]:
        if i <= thres:
            adj_result.append(0)
        elif i >= thres:
            adj_result.append(1)
    adj_result = np.array(adj_result)

    return adj_result



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

def maxnode(graphset):    
    temp = len(reduce(lambda w1, w2: w1 if len(w1)>len(w2) else w2, graphset)) 
    return int(np.sqrt(temp))



''''''''''''
'''model '''
''''''''''''

#Swap training model
tf.keras.backend.set_floatx('float64')
class model:
    
    def __init__(self, a, b, r, lr, S): # S for checking max.node size
        xavier=tf.keras.initializers.glorot_uniform   
        
        self.maxnumnode = maxnode(S)
        print("maxnumnode for the model instance is : ", self.maxnumnode)
        
        self.learning_rate = lr
        
        self.l1=tf.keras.layers.Dense(self.maxnumnode**2, kernel_initializer=xavier, activation=tf.nn.leaky_relu,input_shape=[1])
        self.l2=tf.keras.layers.Dense(128,kernel_initializer=xavier,activation=tf.nn.leaky_relu)
        self.l3=tf.keras.layers.Dense(64,kernel_initializer=xavier,activation=tf.nn.leaky_relu)
        self.out=tf.keras.layers.Dense(self.maxnumnode**2, kernel_initializer=xavier, activation=tf.nn.sigmoid)
        
        self.lprox1=tf.keras.layers.Dense(self.maxnumnode**2,kernel_initializer=xavier,activation=tf.nn.leaky_relu,input_shape=[1])
        self.lprox2=tf.keras.layers.Dense(128,kernel_initializer=xavier,activation=tf.nn.leaky_relu)
        self.lprox3=tf.keras.layers.Dense(64,kernel_initializer=xavier,activation=tf.nn.leaky_relu)
        self.outprox=tf.keras.layers.Dense(self.maxnumnode**2,kernel_initializer=xavier,activation=tf.nn.sigmoid)       
       
        self.lr_schedule = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=self.learning_rate,decay_steps=100,decay_rate=0.9)
        self.train_op = tf.keras.optimizers.Adam(learning_rate= self.lr_schedule)
        
        #Training hyperparameter for loss fuction
        self.alpha = a
        self.beta = b
        self.gamma = r 
        self.loss = []
        
        
        
        
    # Running the model
    def run(self,X):
        boom=self.l1(X)
        boom1=self.l2(boom)
        boom2=self.l3(boom1)
        boom3=self.out(boom2)
        return boom3
    
    def runtheta(self,Y):
        boom=self.lprox1(Y)
        boom1=self.lprox2(boom)
        boom2=self.lporx3(boom1)
        boom3=self.outprox(boom2)
        
        return boom3
      
    #Custom loss fucntion
    def get_loss(self,Y1,Y2):
        
        # non-proxy part of the model
        input1=self.l1(Y1)
        h1=self.l2(input1)
        h1_2=self.l3(h1)
        pred1=self.out(h1_2)
        
        input2=self.l1(Y2)
        h2=self.l2(input2)
        h2_2=self.l3(h2)
        pred2=self.out(h2_2)
        
        #generate random theta
        self.thetaprox1 = RandomThetaGenerator(pred1.numpy()) #매 epoch마다 다른 theta
        self.thetaprox2 = RandomThetaGenerator(pred2.numpy())
        
        
        # proxy part of the model
        
        #Estimation of theta1
        inputprox1=self.lprox1(pred1)
        hprox1=self.lprox2(inputprox1)
        hprox1_2=self.lprox3(hprox1)
        predprox1=self.outprox(hprox1_2)
                
        #Estimation of theta2
        inputprox2=self.lprox1(pred2)
        hprox2=self.lprox2(inputprox2)
        hprox2_2=self.lprox3(hprox2)
        predprox2=self.outprox(hprox2_2)
        #predprox2 =Makingtheta(predprox2)
        
        #swap loss , self loss
        swap_loss = tf.square(Y2 - tf.multiply(pred1,predprox2)) + tf.square(Y1 - tf.multiply(pred2, predprox1))        
        self_loss = tf.square(Y1 - tf.multiply(pred1,predprox1)) + tf.square(Y2 - tf.multiply(pred2, predprox2))
        
        #prox loss
        proxx_loss = tf.square(pred1 - self.run(tf.multiply(pred1,self.thetaprox1))) + tf.square(pred2 - self.run(tf.multiply(pred2,self.thetaprox2)))
        proxtheta_loss = tf.square(self.thetaprox1 - predprox1) + tf.square(self.thetaprox2 - predprox2)
        
        totloss = tf.reduce_mean(swap_loss + self.alpha*self_loss + self.beta*proxx_loss + self.gamma*proxtheta_loss, axis=-1)
       
        return totloss 
      
    # get gradients
    def get_grad(self,Y1,Y2):
        with tf.GradientTape() as tape:
            tape.watch(self.l1.variables)
            tape.watch(self.l2.variables)
            tape.watch(self.l3.variables)
            tape.watch(self.out.variables)   
            
            L = self.get_loss(Y1,Y2)
            g1 = tape.gradient(L, [self.l1.variables[0],self.l1.variables[1],self.l2.variables[0],self.l2.variables[1],self.l3.variables[0],self.l3.variables[1],self.out.variables[0],self.out.variables[1]])
                    
            L_append = L.numpy()
            self.loss.append(L_append[0])
            
        return g1

    def get_prox_grad(self,Y1,Y2):
        with tf.GradientTape() as tape2:
            tape2.watch(self.lprox1.variables)
            tape2.watch(self.lprox2.variables)
            tape2.watch(self.lprox3.variables)
            tape2.watch(self.outprox.variables)
            L = self.get_loss(Y1,Y2)
            g2 = tape2.gradient(L, [self.lprox1.variables[0],self.lprox1.variables[1],self.lprox2.variables[0],self.lprox2.variables[1],self.lprox3.variables[0],self.lprox3.variables[1],self.outprox.variables[0],self.outprox.variables[1]])

        return g2
    
    # perform gradient descent
    def network_learn(self,Y1,Y2):
        
        g1 = self.get_grad(Y1,Y2)
        g2 = self.get_prox_grad(Y1,Y2)
        
        self.train_op.apply_gradients(zip(g2, [self.lprox1.variables[0],self.lprox1.variables[1],self.lprox2.variables[0],self.lprox2.variables[1],self.lprox3.variables[0],self.lprox3.variables[1],self.outprox.variables[0],self.outprox.variables[1]]))       
        self.train_op.apply_gradients(zip(g1, [self.l1.variables[0],self.l1.variables[1],self.l2.variables[0],self.l2.variables[1],self.l3.variables[0],self.l3.variables[1],self.out.variables[0],self.out.variables[1]]))



