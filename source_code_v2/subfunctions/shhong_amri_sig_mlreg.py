# This code is for estimating the feature maps from fmri response
# 
# Data: The video-fMRI dataset are available online: 
# https://engineering.purdue.edu/libi/lab/Resource.html.
# 
# Environment requirement:  
# This code was developed under Red Hat Enterprise Linux environment.
#
# inputs:
#   - X: Nt-by-dimV matrix, each column is a regressor.
#   - Y: Nt-by-dimH matrix, each column is the response time series
#   - M: Cost weighting matrix (optional). Has the same size as Y. 
#   - x_v: validation data, each column is a regressor.
#   - y_x: validation data, each column is the response time series
#   - m_v: validation data. Cost weighting matrix.
#   - para: contains intial model.
#   - opts: training settings.

# outputs:
#   - para: contains trained model.


# History
# v1.0 (original version) --2017/09/13

# Linear regression model

# compiled by shhong

import numpy as np
import math

def amri_sig_mlreg(X, Y, M, x_v, y_v, m_v, para, opts):
    # defaults:
    InitialMomentum = 0.5     # momentum for first #InitialMomentumIter iterations
    FinalMomentum = 0.9       # momentum for remaining iterations
    lamb_da = 0.05       # L1 regularization parameter
    InitialMomentumIter = 20
    MaxIter = 50
    DropOutRate1 = 0.3 # input dropout rate
    DropOutRate2 = 0 # output dropout rate
    StepRatio = 5e-4
    MinStepRatio = 1e-4
    BatchSize = 0
    disp_flag = 0
    IterNum = 10
    savefilename = []
    SparsityCost = 0

    # read parameters
    if(exist('opts','var')):
        if( isfield(opts,'MaxIter') ):
            MaxIter = opts.MaxIter
        
        if( isfield(opts,'InitialMomentum') ):
            InitialMomentum = opts.InitialMomentum
        
        if( isfield(opts,'InitialMomentumIter') ):
            InitialMomentumIter = opts.InitialMomentumIter
        
        if( isfield(opts,'FinalMomentum') ):
            FinalMomentum = opts.FinalMomentum
        
        if( isfield(opts,'lamb_da') ):
            lamb_da = opts.lamb_da
        
        if( isfield(opts,'SparsityCost') ):
            SparsityCost = opts.SparsityCost
        
        if( isfield(opts,'DropOutRate1') ):
            DropOutRate1 = opts.DropOutRate1
        
        if( isfield(opts,'DropOutRate2') ):
            DropOutRate2 = opts.DropOutRate2
        
        if( isfield(opts,'StepRatio') ):
            StepRatio = opts.StepRatio
        
        if( isfield(opts,'MinStepRatio') ):
            MinStepRatio = opts.MinStepRatio
        
        if( isfield(opts,'BatchSize') ):
            BatchSize = opts.BatchSize
        
        if( isfield(opts,'disp_flag') ):
            disp_flag = opts.disp_flag
        
        if( isfield(opts,'IterNum') ):
            IterNum = opts.IterNum
        
        if( isfield(opts,'savefilename') ):
            savefilename = opts.savefilename
        

    

    # initialize some parameters
    [num, dimV] = X.shape
    dimH = Y.shape(1)

    if( BatchSize <= 0 ):
        BatchSize = num
    

    deltaW = np.zeros(dimV, dimH)

    # start training
    unitMomentum = ((FinalMomentum - InitialMomentum)/(num/BatchSize*InitialMomentumIter))
    momentum = InitialMomentum
    errmat = np.zeros(1, math.floor(num/BatchSize)*MaxIter)
    corrmat = np.zeros(1, math.floor(num/BatchSize)*MaxIter)
    k = 1
    for iter in range(MaxIter):
        # train one interation
        ind = randperm(num)
        N = math.floor(num/BatchSize)
        for batch in range(0,N*BatchSize,BatchSize):
            fprintf(1,'epoch %d batch %d\r',iter,math.ceil(batch/BatchSize)) 

            # set momentum
            if (iter <= InitialMomentumIter):
                momentum = momentum + unitMomentum
            
            # select one batch data
            bind = ind(batch : min([batch+BatchSize-1,num]))
            x = np.array(X[bind,:])
            y = np.array(Y[bind,:])
            m = np.array(M[bind,:])
            # perform Dropout on inputs
            if(DropOutRate1 > 0):
                cMat1 = (np.random.rand(BatchSize,dimV)>DropOutRate1)
                x = x*cMat1
            
            # perform Dropout on outpouts
            if(DropOutRate2 > 0):
                cMat2 = (np.random.rand(BatchSize,dimH)>DropOutRate2)
                y = y*cMat2
            

            # Compute the weights update
            z = x*para.W
            dW = (2/BatchSize)*x.conj().T*((y-z)*m)
            deltaW = momentum * deltaW + (1-momentum)*StepRatio*(dW-(SparsityCost/BatchSize)*x.conj().T*(math.exp(z)) \
                -(lamb_da)*((para.W>0)*2-1))

            # Update the network weights
            para.W = para.W + deltaW

            if len(x_v) != 0:
                z_v = x_v*para.W
                err = ((y_v-z_v)*m_v)**2
                rmse = math.sqrt(sum(err[:]) / err.size)
                errmat[k] = rmse 
                r = amri_sig_corr(y_v[:],z_v[:])
                corrmat[k] = r 
                k = k + 1
            
            if disp_flag and (len(x_v) != 0) and math.remainder(math.ceil(batch/BatchSize),3) == 1:
                figure(101)
                subplot(1,2,1) hist(para.W[:],100)
                subplot(1,2,2) hist(z_v[:],100)
                title('Histogram of W and estimated FMap') 
                drawnow
            
            if disp_flag and (len(x_v) != 0):         
                figure(102) plot([1:k-1]/N, errmat[1:k-1])
                title('Estimation error') drawnow
            
            if disp_flag and (len(x_v) != 0):         
                figure(103) plot([1:k-1]/N, corrmat[1:k-1])
                title('Estimation accuracy (correlation)') drawnow
            
        

        if (math.remainder(iter,IterNum)==0) and (len(savefilename)):
            para.opts = opts
            para.errmat = errmat(math.floor(num/BatchSize):math.floor(num/BatchSize):end)
            para.corrmat = corrmat(math.floor(num/BatchSize):math.floor(num/BatchSize):end)
            save([savefilename,'_iter',str(math.ceil(iter)), '.mat'], 'para')
        

        if StepRatio > MinStepRatio:
            StepRatio = StepRatio*0.96
        
    
    
    para.opts = opts # training setting
    para.errmat = errmat(math.floor(num/BatchSize):math.floor(num/BatchSize):end) # validation root of meam square error
    para.corrmat = corrmat(math.floor(num/BatchSize):math.floor(num/BatchSize):end) # validation correlation
    

return para