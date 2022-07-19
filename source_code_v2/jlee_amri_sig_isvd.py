# Generated with SMOP  0.41
# from libsmop import *
import numpy as np
import scipy
# amri_sig_isvd_copy.m

    # inputs:
#   X: p-by-n, p is dimension, n is number of samples. X = U*S*V';
#   U0: p-by-k0, k0 is the number of principle components
#   S0: k0-by-k0, standard deviation
#   percp: percentage of variance to keep, in range (0 1];
    
    # outputs:
#   U: p-by-k, k is the number of principle components
#   S: k-by-k, standard deviation
#   k: updated number of components by keeping percp*100# variance;
    
    # example:
# X0 = randn(1000,500);
# [U0,S0] = svd(X0,0);
# 
# X = randn(1000,200);
# [U,S] = amri_sig_isvd(X,'var',0.99, 'init',{U0,S0});
# 
# [U1, S1] = amri_sig_isvd([X0 X],'var',0.99);
# 
# junk0 = diag(S);
# junk1 = diag(S1);
    
    # reference:
# Zhao, H., Yuen, P. C., & Kwok, J. T. (2006).
# A novel incremental principal component analysis and its application for face recognition. 
# IEEE Transactions on Systems, Man, and Cybernetics, Part B (Cybernetics), 36(4), 873-886.
    
    ## history
# 0.01 - HGWEN - 09/14/2017 - original file
# 0.02 - HGWEN - 11/16/2017 - change "amri_sig_svd" to "svd" in line 74
    
    ## svd updating
    
# @function
def amri_sig_isvd(X=None,varargin=None,*args,**kwargs):
    varargin = amri_sig_isvd.varargin
    nargin = amri_sig_isvd.nargin

    # check inputs
    if nargin < 1:
        eval('help amri_sig_isvd')
        return U,S,k
    
    # defaults
    percp=0
    init_flag=0
    
    for iter in np.arange(1,(varargin,2),2).reshape(-1):
        Keyword=varargin[iter]
        Value=varargin[iter + 1]
        if (Keyword,'var').lower():
            percp=Value
# amri_sig_isvd_copy.m:51
        else:
            if (Keyword,'init').lower():
                U0=Value[1]
                S0=Value[2]
                init_flag=1
            else:
                print('amri_sig_isvd(): unknown keyword ', Keyword)
    
    # updating svd
    if init_flag == 1:
        A=X - np.dot(U0,(np.dot(U0.T,X)))
        Q,R=scipy.linalg.qr(A,0,nargout=2)

        k=(U0,2).shape
        k0=k
        r=(R,1).shape
        U,S=np.linalg.svd(np.concat([[S0,np.dot(U0.T,X)],[np.zeros(r,k),R]]),0,nargout=2)
        U=np.dot(np.concat([U0,Q]),U)

    else:
        U,S=np.linalg.svd(X,0,nargout=2)
        k=min((X,1).shape,(X,2).shape)
    
    if percp > 0:
        v=np.diag(S) ** 2
        vs=np.cumsum(v)
        vs=vs / vs[-1]
        k=np.argwhere(vs > percp,1,'first')
        if ('k0','var'):
            k=max(k,k0)
    
    U=U(np.arange(),np.arange(1,k))
    S=S(np.arange(1,k),np.arange(1,k))
    return U,S,k
    
if __name__ == '__main__':
    pass
    