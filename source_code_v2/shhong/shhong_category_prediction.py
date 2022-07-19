# This code is for predicting the categories from fmri responses

# Data: The video-fMRI dataset are available online: 
# https://engineering.purdue.edu/libi/lab/Resource.html.

# Environment requirement:  
# This code was developed under Red Hat Enterprise Linux environment.

# Reference: 
# Wen H, Shi J, Zhang Y, Lu KH, Cao J, and Liu Z. (2017) Neural Encoding
# and Decoding with Deep Learning for Dynamic Natural Vision. Cerebral
# Cortex, In press.

# Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). 
# Imagenet classification with deep convolutional neural networks.
# In Advances in neural information processing systems (pp. 1097-1105).


# inputs:
#   - Y: Nt-by-Nc matrix, each row is the semantic represenation in the
#       dimension-reduced space. Nt is the number of volumes. 
#       See AlexNet_feature_processing_encoding.m
#   - X: Nt-by-Nv matrix, each row is the cortical representation in the
#       visual cortex.
#   - lamb_da: a vector, a candidate set of regularization parameters
#   - nfold: a scalar, the number of folds to do cross validation
# outputs:
#   - W: Nc-by-Nv matrix, 
#   - Rmat: Nv*(#lamb_da)*nfold array, validation accuracy (correlation)
#   - Lambda: optimal regularization parameters
#   - q: optimal number of principal components to keep.

## History
# v1.0 (original version) --2017/09/17

# compiled by sh Hong

## fMRI-based Categorization 

import numpy as np
import math

from subfunctions.shhong_amri_sig_corr import amri_sig_corr


def category_prediction(X, Y, lamb_da, nfold):
    
    # Load principal components
    dataroot = '/path/to/alexnet_feature_maps/'
    with open(f'{dataroot}AlexNet_feature_maps_pca_layer7.mat') as f:
        B = np.load(f, 'B')

    Yo = Y*B.conj().T*math.sqrt(B.shape(0)) # transform Y back to the original semantic space
    
    # validation: nfold cross validation    
    Nt = Y.shape[0] # number of total time points
    Nc = Y.shape[1] # number of components
    dT = Nt/nfold # number of time points in a fold
    T = Nt - dT # number of time points for training

    Rmat = np.zeros(Nc, len(lamb_da), nfold, dtype='f4')
    
    print('Validating parameters ..')
    for nf in range(nfold):
        idx_t = np.array(range(1,Nt))
        idx_v = np.array(range((nf-1)*dT+1,nf*dT))
        idx_t[idx_v] = []
        
        Y_t = Y[idx_t,:]
        Yo_v = Yo[idx_v,:] # Ground truth in the original space
        X_v = X[idx_v,:]
        X_t = X[idx_t,:]
        XXT = X_t*X_t.conj().T
        
        # validate the regularization parameter
        for k in range(len(lamb_da)):
            print(f'fold: {nf} lamb_da: {k}')
            lmb = lamb_da(k)
            W = X_t.conj().T * np.linalg.lstsq(XXT + lmb*T*np.eye(T), Y_t)
            
            Y_vp = X_v * W # predicted
            # validate the number of componets
            for q in range(Nc):
                print(f'fold:  {nf}, lamb_da: {k}, componets: {q}')
                Yo_vp = Y_vp[:,1:q]*B[:,1:q].conj().T*math.sqrt(B.shape(0)) 
                R = amri_sig_corr(Yo_v.conj().T, Yo_vp.conj().T, 'mode', 'auto')
                Rmat[q,k,nf] = np.mean(R)


    R = np.mean(Rmat,3)
    _, idx = max(R[:])
    [q,lmbidx] = ind2sub(R.shape,idx)
    Lambda = lamb_da(lmbidx)
    
    # Train optimal model
    W = X.conj().T*(np.linalg.lstsq(X*X.conj().T + lmb*Nt*np.eye(Nt), Y)) # decoder in dimension-reduced space            
    Wo = W*np.B[:,1:q].conj().T*math.sqrt(B.shape[0])  # decoder in the original semantic space


    return Wo, W, Lambda, q
