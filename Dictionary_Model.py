# Simple pytorch implementation of Dictionary Learning based on stochastic gradient descent
#
# June 2018
# Jeremias Sulam


import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np
import time
import pdb

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


####################################
    ## Dict. Learning ##
####################################

class DictLearn(nn.Module):
    def __init__(self,m):
        super(DictLearn, self).__init__()

        self.W = nn.Parameter(torch.randn(28*28, m, requires_grad=False))
        
        # normalization
        self.W.data = NormDict(self.W.data)
        
    def forward(self, Y, SC, K):
        
        # normalizing Dict
        self.W.requires_grad_(False)
        self.W.data = NormDict(self.W.data)
        
        # Sparse Coding
        if SC == 'IHT':
            Gamma,residual, errIHT = IHT(Y,self.W,K)
        elif SC == 'fista':
            Gamma,residual, errIHT = FISTA(Y,self.W,K)
        else: print("Oops!")
        
        
        # Reconstructing
        self.W.requires_grad_(True)
        X = torch.mm(Gamma,self.W.transpose(1,0))
        
        # sparsity
        NNZ = np.count_nonzero(Gamma.cpu().data.numpy())/Gamma.shape[0]
        return X, Gamma, errIHT
        

        
#--------------------------------------------------------------
#         Auxiliary Functions
#--------------------------------------------------------------

def hard_threshold_k(X, k):
    Gamma = X.clone()
    m = X.data.shape[1]
    a,_ = torch.abs(Gamma).data.sort(dim=1,descending=True)
    T = torch.mm(a[:,k].unsqueeze(1),torch.Tensor(np.ones((1,m))).to(device))
    mask = Variable(torch.Tensor((np.abs(Gamma.data.cpu().numpy())>T.cpu().numpy()) + 0.)).to(device)
    Gamma = Gamma * mask
    return Gamma#, mask.data.nonzero()

#--------------------------------------------------------------


def soft_threshold(X, lamda):
    #pdb.set_trace()
    Gamma = X.clone()
    Gamma = torch.sign(Gamma) * F.relu(torch.abs(Gamma)-lamda)
    return Gamma.to(device)


#--------------------------------------------------------------


def IHT(Y,W,K):
    
    c = PowerMethod(W)
    eta = 1/c
    Gamma = hard_threshold_k(torch.mm(Y,eta*W),K)    
    residual = torch.mm(Gamma, W.transpose(1,0)) - Y
    IHT_ITER = 50
    
    norms = np.zeros((IHT_ITER,))

    for i in range(IHT_ITER):
        Gamma = hard_threshold_k(Gamma - eta * torch.mm(residual, W), K)
        residual = torch.mm(Gamma, W.transpose(1,0)) - Y
        norms[i] = np.linalg.norm(residual.cpu().numpy(),'fro')/ np.linalg.norm(Y.cpu().numpy(),'fro')
    
    return Gamma, residual, norms


#--------------------------------------------------------------


def FISTA(Y,W,lamda):
    
    c = PowerMethod(W)
    eta = 1/c
    FISTA_ITER = 20
    norms = np.zeros((FISTA_ITER,))
    # print(c)
    # plt.spy(Gamma); plt.show()
    # pdb.set_trace()
    
    Gamma = soft_threshold(torch.mm(Y,eta*W),lamda)
    Z = Gamma.clone()
    Gamma_1 = Gamma.clone()
    t = 1
    
    for i in range(FISTA_ITER):
        Gamma_1 = Gamma.clone()
        residual = torch.mm(Z, W.transpose(1,0)) - Y
        Gamma = soft_threshold(Z - eta * torch.mm(residual, W), lamda/c)
        
        t_1 = t
        t = (1+np.sqrt(1 + 4*t**2))/2
        #pdb.set_trace()
        Z = Gamma + ((t_1 - 1)/t * (Gamma - Gamma_1)).to(device)
        
        norms[i] = np.linalg.norm(residual.cpu().numpy(),'fro')/ np.linalg.norm(Y.cpu().numpy(),'fro')
    
    return Gamma, residual, norms


#--------------------------------------------------------------

def NormDict(W):
    Wn = torch.norm(W, p=2, dim=0).detach()
    W = W.div(Wn.expand_as(W))
    return W

#--------------------------------------------------------------

def PowerMethod(W):
    ITER = 100
    m = W.shape[1]
    X = torch.randn(1, m).to(device)
    for i in range(ITER):
        Dgamma = torch.mm(X,W.transpose(1,0))
        X = torch.mm(Dgamma,W)
        nm = torch.norm(X,p=2)
        X = X/nm
    
    return nm

#--------------------------------------------------------------


def showFilters(W,ncol,nrows):
    p = int(np.sqrt(W.shape[0]))+2
    Nimages = W.shape[1]
    Mosaic = np.zeros((p*ncol,p*nrows))
    indx = 0
    for i in range(ncol):
        for j in range(nrows):
            im = W[:,indx].reshape(p-2,p-2)
            im = (im-np.min(im))
            im = im/np.max(im)
            Mosaic[ i*p : (i+1)*p , j*p : (j+1)*p ] = np.pad(im,(1,1),mode='constant')
            indx += 1
            
    return Mosaic

