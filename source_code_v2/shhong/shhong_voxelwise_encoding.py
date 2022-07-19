# This code is for building voxel-wise neural encoding models
# 
# Data: The video-fMRI dataset are available online: 
# https://engineering.purdue.edu/libi/lab/Resource.html.
# 
# Environment requirement:  
# This code was developed under Red Hat Enterprise Linux environment.
#
# Reference: 
# Wen H, Shi J, Zhang Y, Lu KH, Cao J, and Liu Z. (2017) Neural Encoding
# and Decoding with Deep Learning for Dynamic Natural Vision. Cerebral
# Cortex, In press.
#
# Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). 
# Imagenet classification with deep convolutional neural networks.
# In Advances in neural information processing systems (pp. 1097-1105).
#

# inputs:
#   - Y: Nt-by-Nc matrix, each column is a regressor (mean = zero)
#   - X: Nt-by-Nv matrix, each column is the response time series (mean=zero) of a voxel
#   - lamb_da: a vector, a candidate set of regularization parameters
#   - nfold: a scalar, the number of folds to do cross validation
# outputs:
#   - W: Nc-by-Nv matrix, each column is the optimal encoding weights of a voxel
#   - Rmat: Nv*(#lamb_da)*nfold array, validation accuracy (correlation)
#   - Lamb_da: optimal regularization parameters

# History
# v1.0 (original version) --2017/09/17

# compiled by sh Hong

import numpy as np

from subfunctions.shhong_amri_sig_corr import amri_sig_corr

# Training voxel-wise encoding models
def voxelwise_encoding(Y, X, lamb_da, nfold):

    # validation: nfold cross validation    
    Nt = Y.shape[0] # number of total time points
    Nc = Y.shape[1] # number of components
    dT = Nt/nfold # number of time points in a fold
    T = Nt - dT # number of time points for training
    Nv = X.shape[1] # number of voxels
    
    Rmat = np.zeros((Nv, len(lamb_da), nfold), dtype='f4')
    
    print('Validating regularization parameters ..')
    for nf in range(nfold):
        idx_t = np.ndarray(Nt)
        idx_v = (nf-1)*dT+np.ndarray(nf*dT)
        idx_t[idx_v] = []
        
        Y_t = Y[idx_t,:]
        Y_v = Y[idx_v,:]
        YTY = Y_t.conj().T * Y_t
        X_v = X[idx_v,:]
        X_t = X[idx_t,:]
        
        for k in range(len(lamb_da)):
            print(f'fold: {nf}, lamb_da: {k}')
            lmb = lamb_da(k)
            M = np.linalg.lstsq(YTY + lmb * T * np.eye(Nc), Y_t.conj().T) 

            R = np.zeros(Nv,1)
            v1 = 1
            while (v1 <= Nv):
                v2 = min(v1+4999,Nv)    
                W = M*X_t[:,v1:v2]
                X_vp = Y_v * W # predicted
                X_vg = X_v[:,v1:v2] # ground truth
                R[v1:v2] = amri_sig_corr(X_vg, X_vp, 'mode', 'auto')
                v1 = v2+1
            
            Rmat[:,k,nf] = R   
    
    
    # choose optimal regularization parameters
    _, Lambda = max(np.mean(Rmat,3),[],2)
    Lambda = lamb_da(Lambda)
    
    # training with optimal regulariztion paramters
    print('Training encoidng models with optimal regularization parameters ..')
    YTY = Y.conj().T * Y
    W = np.zeros(Nc, Nv, 'single')

    for k in range(len(lamb_da)):
        print(f'Progress: {k/len(lamb_da)*100}%')
        lmb = lamb_da(k)
        M = np.linalg.lstsq(YTY + lmb * Nt * np.eye(Nc), Y.conj().T) 
        voxels = (Lambda==lmb)
        W[:,voxels] = M*X[:,voxels]

    
    return W, Rmat, Lambda
