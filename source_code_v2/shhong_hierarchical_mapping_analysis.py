# This code is for retinotopic analysis by using CNN
# 
# Data: The raw and preprocessed fMRI data in NIFTI and CIFTI formats are
# available online: https://engineering.purdue.edu/libi/lab/Resource.html.
# This code focuses on the processing of the fMRI data on the cortical
# surface template (CIFTI format).
#
# Reference: 
# Wen H, Shi J, Zhang Y, Lu KH, Cao J, and Liu Z. (2017) Neural Encoding
# and Decoding with Deep Learning for Dynamic Natural Vision. Cerebral
# Cortex, In press.

# History
# v1.0 (original version) --2017/09/14

# compiled by sh Hong

import numpy as np
import h5py
import sys

sys.path.append('/local_raid1/03_user/sunghyoung/01_project/03_Encoding_decoding/01_NeuralEncodingDecoding/sourcecode/source_code_v2/source_code_v2/subfunctions')

import subfunctions

# Concatenate CNN activation time series in the 1st layer across movie segments
# CNN layer labels
layername = ['/conv1','/conv2','/conv3','/conv4','/conv5','/fc6','/fc7','/fc8']
dataroot = '/path/to/alexnet_feature_maps/'

# Training movies
for lay in range(1,len(layername)+1):
    for seg in range(1, 19):
        print(f'Seg: {seg}')
        secpath = f'{dataroot}AlexNet_feature_maps_processed_seg{seg}.h5'      
        # info = h5info(secpath)
        h5file = h5py.File(secpath, 'r')
        lay_feat = h5file[layername[lay-1] + '/data']  
        dim = lay_feat.shape
        Nf = dim(0) # number of frames
        if seg == 1:
           lay_feat_concatenated = np.zeros([dim[1:end-1],Nf*18],'single') 
        
        if lay <= 5:
            lay_feat_concatenated[:,:,:,(seg-1)*Nf+1:seg*Nf] = lay_feat
        else:
            lay_feat_concatenated[:,(seg-1)*Nf+1:seg*Nf] = lay_feat
        
    

    # check time series
    # figure;plot(squeeze(lay_feat_concatenated(25,25,56,:)));

    with open(f'{dataroot}AlexNet_feature_maps_processed_layer{lay}_concatenated.mat', 'w') as f:
        np.save(f, lay_feat_concatenated)



# Load fmri responses
fmripath = '/path/to/subject1/fmri/'

with open(f'{fmripath}training_fmri.mat') as f:
    fmri = np.load(f, 'fmri') # from movie_fmri_processing.m

fmri_avg = (fmri.data1+fmri.data2) / 2 # average across repeats
fmri_avg = fmri_avg.reshape(fmri_avg.shape[0], fmri_avg.shape[1]*fmri_avg.shape[2], order = 'F').copy()


# Map hierarchical CNN features to brain
# Correlating all the voxels to all the CNN units is time-comsuming.
# Here is the analysis given some example voxels in the visual cortex.
# select voxels
voxels = [21892, 21357, 21885, 51456, 22778, 53919 43797, 54301]

for lay in range(1, len(layername)+1):

    with open(f'{dataroot}AlexNet_feature_maps_processed_layer{lay}_concatenated.mat') as f:
        lay_feat_concatenated = np.load(f, 'lay_feat_concatenated')

    dim = lay_feat_concatenated.shape
    Nu = np.prod(dim[1:], axis=0)# number of units
    Nf = dim(-1) # number of time points
    lay_feat_concatenated = lay_feat_concatenated.reshape(Nu, Nf, order = 'F').copy()

    Rmat = np.zeros([Nu,len(voxels)])
    k1 = 1
    while k1 <= len(voxels):
       print(f'Layer: {lay}, Voxel: {k1}')
       k2 = min(len(voxels), k1+100)
       R = subfunctions.amri_sig_corr(lay_feat_concatenated, fmri_avg[voxels[k1:k2],:])
       Rmat[:,k1:k2] = R
       k1 = k2+1
    
    with open(f'{fmripath}cross_corr_fmri_cnn_layer{lay}.mat', 'w') as f:

        np.save(f, Rmat)


lay_corr = np.zeros(len(layername),len(voxels))
for lay in range(1, len(layername)+1):
    print(f'Layer: {lay}')

    with open(f'{fmripath}cross_corr_fmri_cnn_layer{lay}.mat') as f:

        Rmat = np.load(f, 'Rmat')

    lay_corr[lay,:] = max(Rmat,[],1)


with open(f'{fmripath}cross_corr_fmri_cnn.mat', 'w') as f:

        np.save(f, lay_corr)

# Assign layer index to each voxel
[~,layidx] = max(lay_corr,[],1)


# Display correlation profile for example voxels
figure(100)
for v in 1 : len(voxels):
    plot(1:len(layername),lay_corr(:,v)','--o')
    title(v)
    pause


