##
# amri_sig_corr
#    returns a p-by-p matrix containing the pairwise linear correlation 
#    coefficient between each pair of columns in the n-by-p matrix A, 
#    or return correlation verctor of coresponding columns of A and B, 
#    or return correlation matrix of the pairwise columns of A and B.

# Usage
#   [R,pval] = amri_sig_corr(A)
#   [R,pval] = amri_sig_corr(A,B)

# Inputs
#   A: n-by-p data matrix
#   B: n-by-p input matrix

# Keywords:
#   mode: 'Auto' or 'Cross'. 'Auto' mode returns the the correlation
#   coefficient of the coresponding columns of matrix A and B. 'Cross' mode
#   returns the cross correlation matrix of A and B.

# Output
#   R: p-by-p correlation matrix or a vector of length n
#   pval: p value for Pearson correlation

# Version 
#  1.04
## DISCLAIMER AND CONDITIONS FOR USE:
#     This software is distributed under the terms of the GNU General Public
#     License v3, dated 2007/06/29 (see http://www.gnu.org/licenses/gpl.html).
#     Use of this software is at the user's OWN RISK. Functionality is not
#     guaranteed by creator nor modifier(s), if any. This software may be freely
#     copied and distributed. The original header MUST stay part of the file and
#     modifications MUST be reported in the 'MODIFICATION HISTORY'-section,
#     including the modification date and the name of the modifier.

## MODIFICATION HISTORY
# 1.01 - 07/06/2010 - ZMLIU - compute correlation between two input vectors
#        16/11/2011 - JAdZ  - v1.01 included in amri_eegfmri_toolbox v0.1
# 1.02 - 10/18/2013 - HGWEN - for two input matrix A and B, calculate the 
#                     correlation of the coresponding columns of A and B.
# 1.03 - 04/20/2015 - HGWEN - convect input vectors into column vectors.
# 1.04 - 07/28/2015 - HGWEN - return p-value for Pearson Correlation

import numpy as np
import math

def nargin(*args, **kwargs):

    nArg = len(args)
    nKwArg = len(kwargs)

    output = nArg + nKwArg

    return output


def amri_sig_corr(A,B,mode='cross',df=0):


    ##
    if (nargin() > 1) and (len(B)!=0):

        if len(A.shape) == 1:

            A = A[:]
        
        if len(B.shape) == 1:

            B = B[:]
        
        if all(A.shape== B.shape) == 0 and (mode == 'auto') == 1:
            
            print('ERROR: amri_sig_corr(): A and B must have the same size.')
        
        p = A[1].size
        n = A[0].size

        for i in range(p):

            A[:,i]=A[:,i] - np.mean(A[:,i])
            nn = np.linalg.norm(A[:,i])

            if nn > 0:

                A[:,i]=A[:,i]/nn
                
        
        
        for i in B[1].size:

            B[:,i]=B[:,i]-np.mean(B[:,i])
            nn = np.linalg.norm(B[:,i])

            if nn> 0:

                B[:,i]=B[:,i]/nn
            
        
        if (nargin() > 1) and mode == 'auto':

            R = sum(A*B,1)

        elif (nargin() > 1) and mode == 'cross':
            
            R = A.conj().T * B
        
        
        tval = R*math.sqrt(n-2)/math.sqrt(1-R**2)
        pval=2.*(1-tcdf(abs(tval),n-2))
        return


    p = A[1].size
    n = A[0].size
    for i in range(p):
        A[:,i]=A[:,i]-np.mean(A[:,i])
        nn = np.linalg.norm(A[:,i])
        if nn>0:
            A[:,i]=A[:,i]/nn
        

    R=np.matmul(A.conj().T, A)
    if df > 0:
        n = df

    tval = R*math.sqrt(n-2)/math.sqrt(1-R**2)
    pval=2.*(1-tcdf(abs(tval),n-2))


    return R, pval