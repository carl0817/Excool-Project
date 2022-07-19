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
# modified by bc Ha

# from sys import last_type
import numpy as np
import subfunctions.bcha_amri_sig_corr as amri_sig_corr
import subfunctions.bcha_struct_to_dict as loadmat_struct_to_dict
import h5py
import pandas as pd
import scipy.io as sio
# Concatenate CNN activation time series in the 1st layer across movie segments
# CNN layer labels
layername = ['/conv1','/conv2','/conv3','/conv4','/conv5','/fc6','/fc7','/fc8']
dataroot = '/local_raid1/03_user/sunghyoung/01_project/03_Encoding_decoding/01_NeuralEncodingDecoding/sourcecode/source_code_v2/source_code_v2/feature_extracted/'

# Training movies
for lay in range(0, len(layername)): #  from lay 1 to lay 8
    for seg in range(0, 18): # from seg 1 to seg 18
        print('Seg:', str(seg+1))
        secpath = f'{dataroot}AlexNet_feature_maps_processed_seg{str(seg+1)}.h5'   
        # info = h5info(secpath)
        lay_feat = h5py.File(secpath, 'r').get(layername[lay]+'/data')
        # lay_feat: extracted feature maps in each layer and segment
        dim = lay_feat.shape # [55 55 96 240] in matlab
        # (240, 96, 55, 55) in python
        # (# of frames,  )
        # Nf = dim[-1] # number of frames
        Nf = dim[0] # number of frames

        if seg == 0: # initializing lay_feat_concatenated in each layer
            if lay <= 4: # from layer 1 to 5
                # lay_feat_concatenated = np.zeros((dim[0], dim[1], dim[2], Nf*18),'single')
                lay_feat_concatenated = np.zeros((Nf*18, dim[1], dim[2], dim[3]),'single')
            else: # from layer 6 to 8
                lay_feat_concatenated = np.zeros((Nf*18, dim[1]),'single')

        if lay <= 4: # from layer 1 to 5
            lay_feat_concatenated[seg*Nf:(seg+1)*Nf,:,:,:] = lay_feat
        else: # from layer 6 to 8
            lay_feat_concatenated[seg*Nf:(seg+1)*Nf,:] = lay_feat

    print(f'lay{str(lay+1)} done')
        

    # check time series
    # figure;plot(squeeze(lay_feat_concatenated(25,25,56,:)));

    #save(f'{dataroot}AlexNet_feature_maps_processed_layer{str(lay)}_concatenated.h5 'lay_feat_concatenated')
    #store = pd.HDFStore(f'{dataroot}AlexNet_feature_maps_processed_layer{str(lay)}_concatenated.h5')
    #store.put()
    np.save(f'{dataroot}AlexNet_feature_maps_processed_layer{str(lay+1)}_concatenated.npy', lay_feat_concatenated)

# Load fmri responses (for subject 1 only!)
fmripath = '/local_raid1/03_user/sunghyoung/01_project/03_Encoding_decoding/01_NeuralEncodingDecoding/subject1/video_fmri_dataset/subject1/fmri/'
#load(f'{fmripath}training_fmri.mat'],'fmri') # from movie_fmri_processing.m

#fmri = sio.loadmat(f'{fmripath}training_fmri.mat')
# load struct mat file and change it to python dictionary
fmri = loadmat_struct_to_dict.loadmat(f'{fmripath}training_fmri.mat')

fmri_avg = (np.array(fmri['fmri']['data1'])+np.array(fmri['fmri']['data2'])) / 2 # average across repeats
fmri_avg = fmri_avg.reshape(fmri_avg.shape[0], fmri_avg.shape[1]*fmri_avg.shape[2], order='F').copy()
# reshaped into (59412,(240*18))


# Map hierarchical CNN features to brain
# Correlating all the voxels to all the CNN units is time-comsuming.
# Here is the analysis given some example voxels in the visual cortex.
# select voxels
voxels = [21892, 21357, 21885, 51456, 22778, 53919, 43797, 54301]

for lay in range(0, len(layername)):
    lay_feat_concatenated = np.load(f'{dataroot}AlexNet_feature_maps_processed_layer{str(lay+1)}_concatenated.npy')
    dim = lay_feat_concatenated.shape
    # Nu = np.prod((dim[0], dim[1], dim[2]), axis=0) # number of units
    if lay <= 4: # from layer 1 to 5
        Nu = dim[1]*dim[2]*dim[3] # number of units in case of layer 1-5
    else: # from layer 6 to 8
        Nu = dim[1] # number of units in case of layer 6-8
    Nf = dim[0] # number of time points
    lay_feat_concatenated = lay_feat_concatenated.reshape(Nu, Nf, order = 'F').copy()

    Rmat = np.zeros((Nu,len(voxels)))
    k1 = 1
    while k1 <= len(voxels):
       print(f'Layer: {str(lay+1)} Voxel: {str(k1)}')
       k2 = min(len(voxels), k1+100)
       R = amri_sig_corr.amri_sig_corr(lay_feat_concatenated.transpose(), fmri_avg[voxels[k1-1:k2],:].transpose())
       Rmat[:,k1-1:k2] = R
       k1 = k2+1
    
    # save([fmripath, 'cross_corr_fmri_cnn_layer',str(lay),'.mat'], 'Rmat', '-v7.3')
    np.save(f'{fmripath}cross_corr_fmri_cnn_layer{str(lay+1)}.npy', Rmat)


lay_corr = np.zeros(len(layername),len(voxels))
for lay in range(0, len(layername)):
    print(f'Layer: {str(lay+1)}')
    # load([fmripath, 'cross_corr_fmri_cnn_layer',str(lay),'.mat'],'Rmat')
    Rmat = np.load(f'{fmripath}cross_corr_fmri_cnn_layer{str(lay+1)}.npy')
    lay_corr[lay,:] = max(Rmat,[],1)


#save([fmripath, 'cross_corr_fmri_cnn.mat'], 'lay_corr')
np.save(f'{fmripath}cross_corr_fmri_cnn.npy', lay_corr)

# Assign layer index to each voxel
'''[~,layidx] = max(lay_corr,[],1)'''


# Display correlation profile for example voxels
'''
figure(100)
for v in range(0, len(voxels)):
    plot(1:len(layername),lay_corr(:,v)','--o')
    title(v+1)
    pause
'''

