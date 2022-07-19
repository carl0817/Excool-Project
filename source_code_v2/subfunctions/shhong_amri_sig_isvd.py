# inputs:
#   X: p-by-n, p is dimension, n is number of samples. X = U*S*V'
#   U0: p-by-k0, k0 is the number of principle components
#   S0: k0-by-k0, standard deviation
#   percp: percentage of variance to keep, in range (0 1]
#
# outputs:
#   U: p-by-k, k is the number of principle components
#   S: k-by-k, standard deviation
#   k: updated number of components by keeping percp*100# variance

# example:
# X0 = randn(1000,500)
# [U0,S0] = svd(X0,0)
# 
# X = randn(1000,200)
# [U,S] = amri_sig_isvd(X,'var',0.99, 'init',{U0,S0})
# 
# [U1, S1] = amri_sig_isvd([X0 X],'var',0.99)
# 
# junk0 = diag(S)
# junk1 = diag(S1)

# reference:
# Zhao, H., Yuen, P. C., & Kwok, J. T. (2006).
# A novel incremental principal component analysis and its application for face recognition. 
# IEEE Transactions on Systems, Man, and Cybernetics, Part B (Cybernetics), 36(4), 873-886.


# history
# 0.01 - HGWEN - 09/14/2017 - original file
# 0.02 - HGWEN - 11/16/2017 - change "amri_sig_svd" to "svd" in line 74

# compiled by sh Hong

import numpy as np
import os



# def nargin(*args, **kwargs):

#     nArg = len(args)
#     nKwArg = len(kwargs)

#     output = nArg + nKwArg

#     return output

# svd updating
def amri_sig_isvd(X, *varargin):

    # # check inputs
    # if nargin < 1:

    #     eval('help amri_sig_isvd')

    #     return
    

    # defaults
    percp = 0
    init_flag = 0

    # Keywords
    for iter in range(0, len(varargin), 2): 

        Keyword = varargin[iter]
        Value   = varargin[iter+1]

        if Keyword == 'var':

            percp = Value 

        elif Keyword == 'init':

            U0 = Value[1]
            S0 = Value[2]
            init_flag = 1

        else:

            print(f'WARNING: amri_sig_isvd(): unknown keyword {Keyword}')
            
        
    

    # updating svd
    if init_flag == 1:

        A = X - U0*(U0.conj().T*X)
        [Q,R] = np.linalg.qr(A,0)

        k = U0.shape[1]
        k0 = k
        r = R.shape[0]
        
        [U, S, _] = np.linalg.svd([S0, U0.conj().T*X, np.zeros((r,k)), R], 0)
        U = [U0, Q]*U

    else:
        [U, S, _] = np.linalg.svd(X,0) # S.shape = (14400, 14400)
        S = np.diag(S)
        k = min(X.shape[0],X.shape[1])
    

    if percp>0:
        v = np.diag(S)**2
        vs = np.cumsum(v)
        vs = vs/vs[-1]
        k = np.nonzero(vs>percp)[0][0]
        if os.path.isfile('k0','var'):
            k = max(k,k0)
        
    

    U = U[:, 0:k]
    S = S[0:k, 0:k]

    return U, S, k

