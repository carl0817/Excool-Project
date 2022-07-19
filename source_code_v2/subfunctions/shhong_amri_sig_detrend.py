# amri_sig_detrend
# remove polynormial functions from the input time series
#
# Version 0.01

# History
# 0.01 - 06/03/2014 - ZMLIU - don't call amri_sig_nvr
#                           - noted that detrend may distort the signal

# compiled by sh Hong 

import numpy as np

def nargin(*args, **kwargs):

    nArg = len(args)
    nKwArg = len(kwargs)

    output = nArg + nKwArg

    return output

#
def amri_sig_detrend(its, polyorder):

    if nargin<1:
        eval('help amri_sig_detrend')
        return


    if nargin<2:
        polyorder=1


    polyorder = round(polyorder)
    if polyorder<0:
        polyorder=0


    [nr,nc]=its.shape
    its=its[:]
    its=its-np.mean(its)

    if polyorder>0:
        nt=len(its)
        

        poly=np.zeros(nt,polyorder+1)

        for i in range(polyorder+1):

            poly[:,i]=np.arange(1, nt+1)**(i-1)
            poly[:,i]=poly[:,i]/np.linalg.norm(poly[:,i])
        
        p=np.linalg.lstsq(np.array(poly), np.array(its))
        trend=np.array(poly)@p
        ots=its-trend
        
    #     poly=zeros(nt,polyorder)
    #     for i=1:polyorder
    #         poly(:,i)=(1:nt).^i
    #         poly(:,i)=poly(:,i)./norm(poly(:,i))
    #     end
    #     ots=amri_sig_nvr(its,poly)
    
        ots=ots.reshape(nr, nc, order='F').copy()

    else:

        ots=its


    return ots