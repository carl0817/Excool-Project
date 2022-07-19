# This code is for processing the CNN features extracted from videos
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
# Zha, H., & Simon, H. D. (1999). 
# On updating problems in latent semantic indexing. 
# SIAM Journal on Scientific Computing, 21(2), 782-791.
# 
# Zhao, H., Yuen, P. C., & Kwok, J. T. (2006). 
# A novel incremental principal component analysis and its application for face recognition. 
# IEEE Transactions on Systems, Man, and Cybernetics, Part B (Cybernetics), 36(4), 873-886.
#
# Wen, H., Shi, J., Chen, W., & Liu, Z. (2017). 
# Transferring and Generalizing Deep-Learning-based Neural Encoding Models across Subjects. 
# bioRxiv, 171017.

# History
# v1.0 (original version) --2017/09/17

#%%
# compiled by sh Hong

import numpy as np
import os
from scipy import signal
import h5py
import sys

# sys.path.append('/local_raid1/03_user/sunghyoung/01_project/03_Encoding_decoding/01_NeuralEncodingDecoding/sourcecode/source_code_v2/source_code_v2/subfunctions')
from subfunctions.shhong_amri_sig_isvd import amri_sig_isvd
# from subfunctions.bcha_spm_hrf import bcha_spm_hrf


# calculate the temporal mean and standard deviation of the feature time series of CNN units
# CNN layer labels
layername = ['/conv1','/conv2','/conv3','/conv4','/conv5','/fc6','/fc7','/fc8']
# dataroot = '/path/to/alexnet_feature_maps/'
dataroot = '/local_raid1/03_user/sunghyoung/01_project/03_Encoding_decoding/01_NeuralEncodingDecoding/sourcecode/source_code_v2/source_code_v2/feature_extracted/'

#%%
# t=0

# # calculate the temporal mean 
# for lay in range(1, len(layername) + 1):
#     N = 0
#     for seg in range(1, 19):

#         print(f'Layer: {layername[lay-1]}; Seg: {seg}')
#         secpath = f'{dataroot}AlexNet_feature_maps_seg{seg}.h5'

#         if os.path.isfile(secpath):

#             h5file = h5py.File(secpath, 'r')
#             lay_feat = h5file[layername[lay-1] + '/data']                     
#             dim = lay_feat.shape # convolutional layers: #kernel*fmsize1*fmsize2*#frames in matlab; ex) [55,55,96,14400] in matlab, (14400,96,55,55) in python
#             if seg == 1:
#                 lay_feat_mean = np.zeros((dim[1:]))
            
#             lay_feat_mean = lay_feat_mean + sum(lay_feat, len(dim))
#             N = N  + dim[0]


#     lay_feat_mean = lay_feat_mean/N

#     if t==0:

#         with h5py.File(f'{dataroot}AlexNet_feature_maps_avg.h5' , 'w') as f:
            
#             grp = f.create_group(layername[lay-1])
#             dset = grp.create_dataset(name='data', data=lay_feat_mean, shape=lay_feat_mean.shape, dtype='f4')

#         t += 1

#     else:
        
#         with h5py.File(f'{dataroot}AlexNet_feature_maps_avg.h5' , 'a') as f:

#             grp = f.create_group(layername[lay-1])
#             dset = grp.create_dataset(name='data', data=lay_feat_mean, shape=lay_feat_mean.shape, dtype='f4')

# #%%
# # calculate the temporal standard deviation
# t = 0

# for lay in range(1,len(layername) + 1):

#     h5file = h5py.File(f'{dataroot}AlexNet_feature_maps_avg.h5', 'r')
#     lay_feat_mean = h5file[layername[lay-1] + '/data']
#     N = 0
#     for seg in range(1, 19):
#         print(f'Layer: {layername[lay-1]}; Seg: {seg}')
#         secpath = f'{dataroot}AlexNet_feature_maps_seg{seg}.h5'
#         if os.path.isfile(secpath):
#             lay_feat = h5py.File(secpath, 'r')
#             lay_feat = lay_feat[layername[lay-1] + '/data']
#             lay_feat = np.array(lay_feat)
#             lay_feat_mean = np.array(lay_feat_mean)
#             lay_feat = np.subtract(lay_feat, lay_feat_mean)
#             lay_feat = lay_feat**2
#             dim = lay_feat.shape
#             if seg == 1:
#                 lay_feat_std = np.zeros((dim[1:]))
           
#             lay_feat_std = lay_feat_std + sum(lay_feat,len(dim))
#             N = N  + dim[0]
 

#     lay_feat_std = np.sqrt(lay_feat_std/(N-1))
#     lay_feat_std[lay_feat_std==0] = 1

#     if t==0:

#         with h5py.File(f'{dataroot}AlexNet_feature_maps_std.h5' , 'w') as f:
            
#             grp = f.create_group(layername[lay-1])
#             dset = grp.create_dataset(name='data', data=lay_feat_mean, shape=lay_feat_mean.shape, dtype='f4')

#         t += 1

#     else:
        
#         with h5py.File(f'{dataroot}AlexNet_feature_maps_std.h5' , 'a') as f:

#             grp = f.create_group(layername[lay-1])
#             dset = grp.create_dataset(name='data', data=lay_feat_std, shape=lay_feat_std.shape, dtype='f4')

# # Reduce the dimension of the AlexNet features for encoding models
# # Dimension reduction by using principal component analysis (PCA)
# # Here provides two ways to compute the PCAs. If the memory is big enough
# # to directly calculate the svd, then use the method 1, otherwise use the
# # SVD-updating algorithm (Zha and Simon, 1999 Zhao et al., 2006 Wen et al. 2017).

# #%%
# # # # # # # # # # Direct SVD # # # # # # # # # # # # # #
# # This requires big memory to compute. It's better to reduce the
# # frame rate of videos or use the SVD-updating algorithm.

# for lay in range(1, len(layername) + 1):
#     h5file = h5py.File(f'{dataroot}AlexNet_feature_maps_avg.h5', 'r')
#     lay_feat_mean = h5file[layername[lay-1] + '/data']

#     h5file = h5py.File(f'{dataroot}AlexNet_feature_maps_std.h5', 'r')
#     lay_feat_std = h5file[layername[lay-1] + '/data']

#     lay_feat_mean = lay_feat_mean[:]
#     lay_feat_std = lay_feat_std[:]
    
#     # Concatenating the feature maps across training movie segments. Ensure that 
#     # the memory is big enough to concatenate all movie segments. Otherwise, 
#     # use the SVD-updating algorithm. 
    
#     for seg in range(1, 19):
#         print(f'Layer: {layername[lay-1]}; Seg: {seg}')
#         secpath = f'{dataroot}AlexNet_feature_maps_seg{seg}.h5'
            
#         h5file = h5py.File(f'{dataroot}AlexNet_feature_maps_seg{seg}.h5', 'r')
#         lay_feat = h5file[layername[lay-1] + '/data']
#         dim = lay_feat.shape
#         Nu = np.prod(dim[1:]) # number of units
#         Nf = dim[0] # number of frames # 14400 in matlab
#         lay_feat = np.array(lay_feat)
#         lay_feat = lay_feat.reshape(Nu, Nf, order='F').copy() # Nu*Nf
        
# #         lay_feat = lay_feat(:,1:3:end) # downsample if encounter memory issue
# #         Nf = size(lay_feat,2)
        
#         if seg == 1:
#             lay_feat_cont = np.zeros((Nu, Nf*18), dtype='f4')
       
#         lay_feat_cont[: , (seg-1)*Nf: seg*Nf] = lay_feat
    
    
#     # standardize the time series for each unit  
#     lay_feat_cont = lay_feat_cont - lay_feat_mean
#     lay_feat_cont = lay_feat_cont/lay_feat_std
#     lay_feat_cont[np.isnan(lay_feat_cont)] = 0 # assign 0 to nan values
    
#     #[B, S] = svd(lay_feat_cont,0)
#     if lay_feat_cont.shape[0] > lay_feat_cont.shape[1]:
#         R = lay_feat_cont.conj().T * lay_feat_cont / lay_feat_cont.shape[0]
#         [U,S] = np.linalg.svd(R)
#         s = np.diag(S)
        
#         # keep 99% variance
#         ratio = np.cumsum(s)/sum(s) 
#         Nc = np.nonzero(ratio>0.99)[0][0] # number of components
        
#         S_2 = np.diag(1/np.sqrt(s[1:Nc])) # S.^(-1/2)
#         B = lay_feat_cont*(U[:,1:Nc]*S_2/np.sqrt(lay_feat_cont.shape[0]))
#         # I = B'*B # check if I is an indentity matrix
        
#     else:
#         R = lay_feat_cont*lay_feat_cont.conj().T
#         [U,S] = np.linalg.svd(R)
#         s = np.diag(S)
        
#         # keep 99% variance
#         ratio = np.cumsum(s)/sum(s) 
#         Nc = np.nonzero(ratio>0.99)[0][0] # number of components
        
#         B = U[:,0:Nc]
    

#     # save principal components
#     with open(f'{dataroot}AlexNet_feature_maps_pca_layer{lay}.mat', 'w') as f:
#         np.save(f, B)
#         np.save(f, s)
#     # np.save(f'{dataroot}AlexNet_feature_maps_pca_layer{lay}.mat', 'B', 's', '-v7.3')

#%%
# # # # # # # # # # SVD-updating algorithm # # # # # # # # #
Niter = 2 # number of iteration to compute principle component

for lay in range(1, len(layername) + 1):
    h5file = h5py.File(f'{dataroot}AlexNet_feature_maps_avg.h5', 'r')
    lay_feat_mean = h5file[layername[lay-1] + '/data']

    h5file = h5py.File(f'{dataroot}AlexNet_feature_maps_std.h5', 'r')
    lay_feat_std = h5file[layername[lay-1] + '/data']
    
    k0 = 0
    percp = 0.99 # explain 99# of the variance of every movie segments
    for iter in  range(1, Niter + 1):
        for seg in range(1, 19):
            print(f'Layer: {lay}, Seg: {seg}, Comp: {k0}')
            secpath = f'{dataroot}AlexNet_feature_maps_seg{seg}.h5'
            if os.path.isfile(secpath):
                h5file = h5py.File(secpath, 'r')
                lay_feat = h5file[layername[lay-1] + '/data'] 
                lay_feat = np.array(lay_feat)
                lay_feat_mean = np.array(lay_feat_mean)
                lay_feat = lay_feat - lay_feat_mean
                lay_feat = lay_feat/lay_feat_std
                lay_feat[np.isnan(lay_feat)] = 0 # assign 0 to nan values
                
                dim = lay_feat.shape
                lay_feat = lay_feat.reshape(np.prod(dim[1:]),dim[0], order='F').copy()
                if (seg == 1) and (iter == 1):
                    [B, S, k0] = amri_sig_isvd(lay_feat, 'var', percp)
                else:
                    [B, S, k0] = amri_sig_isvd(lay_feat, 'var',  percp, 'init', [B,S])
                 
            else:
               print("file doesn't exist") 
  
    s = np.diag(S)
    
    # save principal components
    with open(f'{dataroot}AlexNet_feature_maps_svd_layer{lay}.mat', 'wb') as f:
        np.save(f, B)
        np.save(f, s)


# #%%
# # Processed the dimension-reduced 
# # CNN layer labels
# layername = {'/conv1','/conv2','/conv3','/conv4','/conv5','/fc6','/fc7','/fc8'}
# dataroot = '/local_raid1/03_user/sunghyoung/01_project/03_Encoding_decoding/01_NeuralEncodingDecoding/sourcecode/source_code_v2/source_code_v2/feature_extracted/'

# # The sampling rate should be equal to the sampling rate of CNN feature
# # maps. If the CNN extracts the feature maps from movie frames with 30
# # frames/second, then srate = 30. It's better to set srate as even number
# # for easy downsampling to match the sampling rate of fmri (2Hz).
# srate = 30 

# # Here is an example of using pre-defined hemodynamic response function
# # (HRF) with positive peak at 4s.
# p  = [5, 16, 1, 1, 6, 0, 32]
# hrf = bcha_spm_hrf(1/srate,p)
# hrf = hrf[:]
# # figure plot(0:1/srate:p(7),hrf)

# #%%
# # Dimension reduction for CNN features
# for lay in range(1, len(layername) + 1):
#     h5file = h5py.File(f'{dataroot}AlexNet_feature_maps_avg.h5', 'r')
#     lay_feat_mean = h5file[layername[lay-1] + '/data']

#     h5file = h5py.File(f'{dataroot}AlexNet_feature_maps_std.h5', 'r')
#     lay_feat_std = h5file[layername[lay-1] + '/data']

#     # with open(f'{dataroot}AlexNet_feature_maps_pca_layer{lay}.mat') as f:
#     with open(f'{dataroot}AlexNet_feature_maps_svd_layer{lay}.mat') as f:
#         B = np.load(f, 'B')
    
#     # Dimension reduction for testing data
#     t = 0
    
#     for seg in range(1, 19):
#         print(['Layer: ', str(lay),' Seg: ',str(seg)])
#         secpath = f'{dataroot}AlexNet_feature_maps_seg{seg}.h5'
#         if os.path.isfile(secpath,'file')==2:
#             h5file = h5py.File(secpath, 'r')
#             lay_feat = h5file[layername[lay-1] + '/data']  
#             lay_feat = lay_feat - lay_feat_mean
#             lay_feat = lay_feat / lay_feat_std
#             lay_feat[np.isnan(lay_feat)] = 0 # assign 0 to nan values
            
#             dim = lay_feat.shape
#             lay_feat = lay_feat.reshape((np.prod(dim[1:]),dim[0]), order='F').copy()
#             Y = lay_feat.conj().T * B / np.sqrt(B.shape[0]) # Y: #time-by-#components
            
#             ts = signal.convolve2d(hrf,Y) # convolude with hrf
#             ts = ts[4*srate+1:4*srate+Y.shape[0],:]
#             ts = ts[srate+1:2*srate:-1,:] # downsampling to match fMRI

#             if t==0:

#                 with h5py.File(f'{dataroot}AlexNet_feature_maps_pcareduced_seg{seg}.h5' , 'w') as f:
                    
#                     grp = f.create_group(layername[lay-1])
#                     dset = grp.create_dataset(name='data', data=ts, shape=ts.shape, dtype='f4')

#                 t += 1

#             else:
                
#                 with h5py.File(f'{dataroot}AlexNet_feature_maps_pcareduced_seg{seg}.h5' , 'a') as f:

#                     grp = f.create_group(layername[lay-1])
#                     dset = grp.create_dataset(name='data', data=ts, shape=ts.shape, dtype='f4')

    
#     # Dimension reduction for testing data
#     t = 0

#     for seg in range(5):
#         print(f'Layer: {lay}, Test: {seg}')
#         secpath = f'{dataroot}AlexNet_feature_maps_test{seg}.h5'
#         if os.path.isfile(secpath,'file')==2:
#             h5file = h5py.File(secpath, 'r')
#             lay_feat = h5file[layername[lay-1] + '/data']  
#             lay_feat = lay_feat - lay_feat_mean
#             lay_feat = lay_feat / lay_feat_std
#             lay_feat[np.isnan(lay_feat)] = 0 # assign 0 to nan values
            
#             dim = lay_feat.shape
#             lay_feat = lay_feat.reshape(np.prod(dim[1:]),dim(0), order='F').copy()
#             Y = lay_feat.conj().T * B / np.sqrt(B.shape[0]) # Y: #time-by-#components
            
#             ts = signal.convolve2d(hrf,Y) # convolude with hrf
#             ts = ts[4*srate+1:4*srate+Y.shape[0],:]
#             ts = ts[srate+1:2*srate:-1,:] # downsampling

#             if t==0:

#                 with h5py.File(f'{dataroot}AlexNet_feature_maps_pcareduced_test{seg}.h5' , 'w') as f:
            
#                     grp = f.create_group(layername[lay-1])
#                     dset = grp.create_dataset(name='data', data=ts, shape=ts.shape, dtype='f4')

#                 t += 1

#             else:
                
#                 with h5py.File(f'{dataroot}AlexNet_feature_maps_pcareduced_test{seg}.h5' , 'a') as f:

#                     grp = f.create_group(layername[lay-1])
#                     dset = grp.create_dataset(name='data', data=ts, shape=ts.shape, dtype='f4')

# #%%
# # Concatenate the dimension-reduced CNN features of training movies
# # CNN layer labels
# layername = ['/conv1','/conv2','/conv3','/conv4','/conv5','/fc6','/fc7','/fc8']
# dataroot = '/local_raid1/03_user/sunghyoung/01_project/03_Encoding_decoding/01_NeuralEncodingDecoding/sourcecode/source_code_v2/source_code_v2/feature_extracted/'

# t = 0

# for lay in range(1, len(layername) + 1):
    
#     for seg in range(1, 19):
#         print(['Layer: ', str(lay),' Seg: ',str(seg)])
#         secpath = f'{dataroot}AlexNet_feature_maps_pcareduced_seg{seg}.h5'     
#         h5file = h5py.File(secpath, 'r')
#         lay_feat = h5file[layername[lay-1] + '/data'] #time-by-#components
#         dim = lay_feat.shape
#         Nf = dim[0] # number of frames
#         if seg == 1:
#            Y = np.zeros([Nf*18, dim(2)],'single') 
        
#         Y[(seg-1)*Nf + 1 : seg*Nf, :] = lay_feat

#     if t==0:

#         with h5py.File(f'{dataroot}AlexNet_feature_maps_pcareduced_concatenated.h5' , 'w') as f:
    
#             grp = f.create_group(layername[lay-1])
#             dset = grp.create_dataset(name='data', data=Y, shape=Y.shape, dtype='f4')

#         t += 1

#     else:
        
#         with h5py.File(f'{dataroot}AlexNet_feature_maps_pcareduced_concatenated.h5' , 'a') as f:

#             grp = f.create_group(layername[lay-1])
#             dset = grp.create_dataset(name='data', data=Y, shape=Y.shape, dtype='f4')