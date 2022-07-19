## This code is for processing the CNN features extracted from videos
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

## History
# v1.0 (original version) --2017/09/17

import numpy as np
import scipy
import subfunctions
import h5py

## Process the AlexNet features for bivariate analysis to relate CNN units to brain voxels
# CNN layer labels
layername = ['/conv1','/conv2','/conv3','/conv4','/conv5','/fc6','/fc7','/fc8']

# The sampling rate should be equal to the sampling rate of CNN feature
# maps. If the CNN extracts the feature maps from movie frames with 30
# frames/second, then srate = 30. It's better to set srate as even number
# for easy downsampling to match the sampling rate of fmri (2Hz).
srate = 30

# Here is an example of using predefined hemodynamic response function
# (HRF) with positive peak at 4s.
p  = [5, 16, 1, 1, 6, 0, 32]
### addpath(genpath('/local_raid1/03_user/sunghyoung/01_project/03_Encoding_decoding/01_NeuralEncodingDecoding/sourcecode/source_code_v2/source_code_v2/subfunctions/'))
hrf = bcha_spm_hrf(1/srate, p)
## hrf = hrf(:)
# figure; plot(0:1/srate:p(7),hrf);

dataroot = '/local_raid1/03_user/sunghyoung/01_project/03_Encoding_decoding/01_NeuralEncodingDecoding/stimuli/video_fmri_dataset/stimuli/'

# Training movies
for seg in range(0, 18, 1):
    secpath = f'{dataroot}AlexNet_feature_maps_seg{str(seg+1)}.h5'
    
    for lay in range(len(layername)):
        print(f'Seg: {str(seg+1)}; Layer: {layername[lay]}')
        # info = h5info(secpath);
        lay_feat = h5py.File(secpath, 'r').get(layername[lay]+'/data')
        dim = lay_feat.shape
        Nu = np.prod((dim[0], dim[1], dim[2]), axis=0) # number of units
        Nf = dim[-1] # number of frames
        lay_feat = lay_feat.reshape((Nu,Nf)) # Nu*Nf
        if lay+1 < len(layername):
            lay_feat = np.log10(lay_feat + 0.01) # log-transformation except the last layer
        # ts = conv2(hrf,lay_feat') # convolude with hrf
        ts = scipy.signal.convolve2d(in1=hrf, in2=lay_feat.T)
        ts = ts[4*srate+1:4*srate+Nf+1, :]
        ts = ts[srate+1::2*srate, :].T # downsampling
        ts = ts.reshape((dim[0], dim[1], dim[2], 240))
        
        # check time series
        # figure;plot(squeeze(ts(25,25,56,:)));

        # h5create([dataroot,'AlexNet_feature_maps_processed_seg', num2str(seg),'.h5'],[layername{lay},'/data'],...
        #    [size(ts)],'Datatype','single');
        # h5write([dataroot,'AlexNet_feature_maps_processed_seg', num2str(seg),'.h5'], [layername{lay},'/data'], ts);
        hf = h5py.File(f'{dataroot}AlexNet_feature_maps_processed_seg{str(seg+1)}.h5', 'w')
        hf.create_dataset(f'{layername[lay]}/data', data=ts)
        hf.close()
        

# Testing movies
for test in range(5):
    secpath = f'{dataroot}AlexNet_feature_maps_test{str(test+1)}.h5'
    for lay in range(len(layername)):
        print(f'Test: {str(test+1)}; Layer: {layername[lay]}')
        lay_feat = h5py.File(secpath, 'r').get(layername[lay]+'/data')
        dim = lay_feat.shape
        Nu = np.prod((dim[0], dim[1], dim[2]), axis=0) # number of units
        Nf = dim[-1] # number of frames
        lay_feat = lay_feat.reshape((Nu,Nf)) # Nu*Nf
        if lay+1 < len(layername):
            lay_feat = np.log10(lay_feat + 0.01) # log-transformation except the last layer
        ts = scipy.signal.convolve2d(in1=hrf, in2=lay_feat.T)
        ts = ts[4*srate+1:4*srate+Nf+1, :]
        ts = ts[srate+1::2*srate, :].T # downsampling
        ts = ts.reshape((dim[0], dim[1], dim[2], 240))

        # h5create([dataroot,'AlexNet_feature_maps_processed_test', num2str(test),'.h5'],[layername{lay},'/data'],...
        #    [size(ts)],'Datatype','single')
        # h5write([dataroot,'AlexNet_feature_maps_processed_test', num2str(test),'.h5'], [layername{lay},'/data'], ts)
        hf = h5py.File(f'{dataroot}AlexNet_feature_maps_processed_test{str(test+1)}.h5', 'w')
        hf.create_dataset(f'{layername[lay]}/data', data=ts)
        hf.close()