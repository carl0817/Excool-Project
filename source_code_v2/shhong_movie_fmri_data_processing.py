# This code is for processing the BOLD fMRI response to natural movies
# 
# Data: The raw and preprocessed fMRI data in NIFTI and CIFTI formats are
# available online: https://engineering.purdue.edu/libi/lab/Resource.html.
# This code focuses on the processing of the fMRI data on the cortical
# surface template (CIFTI format).
# 
# Environment requirement: Install the workbench toolbox published by Human
# Connectome Project (HCP). It is public available on 
# https://www.humanconnectome.org/software/connectome-workbench. 
# This code was developed under Red Hat Enterprise Linux environment.
#
# Reference: 
# Wen H, Shi J, Zhang Y, Lu KH, Cao J, and Liu Z. (2017) Neural Encoding
# and Decoding with Deep Learning for Dynamic Natural Vision. Cerebral
# Cortex, In press.

# History
# v1.0 (original version) --2017/09/13

# compiled by shhong

import numpy as np
import sys

sys.path.append('/local_raid1/03_user/sunghyoung/01_project/03_Encoding_decoding/01_NeuralEncodingDecoding/sourcecode/source_code_v2/source_code_v2/subfunctions')

import subfunctions

# Process fMRI data (CIFTI) for an example segment
fmripath = '/path/to/fmri/' # path to the cifti files
filename = 'seg1/cifti/seg1_1_Atlas.dtseries.nii'
cii = ciftiopen([fmripath, filename],'wb_command')

# For training data (with prefixed 'seg') and the first testing data
# (i.e. test1), we disregarded the first volume and the the last 4 volumes, 
# reducing the number of volumes from 245 to 240 volumes (i.e. 8mins, TR=2s). 
# For other testing data, we disregarded the first 2 volumes and the last 4
# volumes, reducing volume# from 246 to 240.
if cii.cdata.shape[1] == 245:
    st = 2
else:
    st = 3

# Mask out the vertices in subcortical areas 
Nv = 59412# number of vertices on the cortical surface
lbl = np.zeros(cii.cdata.shape[0],1)
lbl[1:Nv] = 1
lbl = (lbl == 1)
data = np.array(cii.cdata[lbl,st:end-4])

# Remove the 4th order polynomial trend for each voxel
for i in 1 : data.shape[0]:
    data[i,:] = subfunctions.amri_sig_detrend(data[i,:],4)

# Standardization: remove the mean and divide the standard deviation
data = data - np.mean(data,2)
data = data / np.std(data,[],2)

# Check the time series
figure
for i in  1 : 100 : 10000:
    plot(data[i,:])
    pause


# Put all training segments together
Nv = 59412 # number of vertices on the cortical surface
Nt = 240  # number of volumes
Ns = 18 # number of training movie segments

fmripath = '/path/to/fmri/' # path to the cifti files
fmri.data1 = np.zeros(Nv,Nt,Ns,'single')# use single to save memory  
fmri.data2 = np.zeros(Nv,Nt,Ns,'single')
Rmat = np.zeros(Nv,Ns) 

for seg in range(Ns):
    print(['segment: ',str(seg)])
    filename1 = ['seg',str(seg),'/cifti/seg', str(seg),'_1_Atlas.dtseries.nii']
    filename2 = ['seg',str(seg),'/cifti/seg', str(seg),'_2_Atlas.dtseries.nii']
    
    cii1 = ciftiopen([fmripath, filename1],'wb_command')
    cii2 = ciftiopen([fmripath, filename2],'wb_command')

    # For training data (with prefixed 'seg') and the first testing data
    # (i.e. test1), we disregarded the first volume and the the last 4 volumes, 
    # reducing the number of volumes from 245 to 240 volumes (i.e. 8mins, TR=2s). 
    # For other testing data, we disregarded the first 2 volumes and the last 4
    # volumes, reducing volume# from 246 to 240.
    if cii.cdata.shape[1] == 245:
        st = 2
    else:
        st = 3
    

    # Mask out the vertices in subcortical areas 
    lbl = np.zeros(cii1.cdata.shape[0],1)
    lbl[1:Nv] = 1
    lbl = (lbl == 1)
    data1 = np.array(cii1.cdata[lbl,st:end-4])
    data2 = np.array(cii2.cdata[lbl,st:end-4])

    # Remove the 4th order polynomial trend for each voxel
    for i in 1 : data1.shape[0]:
        data1[i,:] = subfunctions.amri_sig_detrend(data1[i,:],4)
        data2[i,:] = subfunctions.amri_sig_detrend(data2[i,:],4)

    # Standardization: remove the mean and divide the standard deviation
    data1 = data1 - np.mean(data1,2)
    data1 = data1 / np.std(data1,[],2)
    
    data2 = data2 - np.mean(data2,2)
    data2 = data2 / np.std(data2,[],2)
    
    fmri.data1[:,:,seg] = data1
    fmri.data2[:,:,seg] = data2
    
    # calculate reproducibility to check the data
    R = subfunctions.amri_sig_corr(data1',data2','mode','auto')
    Rmat[:,seg] = R[:]

# save fmri
with open(f'{fmripath}training_fmri.mat', 'w') as f:
    np.save(f, fmri)
    np.save(f, Rmat)

# Put all testing segments together
Nv = 59412 # number of vertices on the cortical surface
Nt = 240  # number of volumes
fmripath = '/path/to/fmri/' # path to the cifti files
fmritest.test1 = np.zeros(Nv,Nt,10,'single')
fmritest.test2 = np.zeros(Nv,Nt,10,'single')
fmritest.test3 = np.zeros(Nv,Nt,10,'single')
fmritest.test4 = np.zeros(Nv,Nt,10,'single')
fmritest.test5 = np.zeros(Nv,Nt,10,'single')

for seg in range(1, 6):
    for rep in range(1, 11):
        print(['segment: ',str(seg),' repeat: ', str(rep)])
        filename = ['test',str(seg),'/cifti/test', str(seg),'_',str(rep),'_Atlas.dtseries.nii']
        cii = ciftiopen([fmripath, filename],'wb_command')

        # For training data (with prefixed 'seg') and the first testing data
        # (i.e. test1), we disregarded the first volume and the the last 4 volumes, 
        # reducing the number of volumes from 245 to 240 volumes (i.e. 8mins, TR=2s). 
        # For other testing data, we disregarded the first 2 volumes and the last 4
        # volumes, reducing volume# from 246 to 240.
        if cii.cdata.shape[1] == 245:
            st = 2
        else:
            st = 3

        # Mask out the vertices in subcortical areas 
        lbl = np.zeros(cii.cdata.shape[0],1)
        lbl[1:Nv] = 1
        lbl = (lbl == 1)
        data = np.array(cii.cdata(lbl,st:end-4))

        # Remove the 4th order polynomial trend for each voxel
        for i in 1 : data.shape[0]:
            data[i,:] = subfunctions.amri_sig_detrend(data[i,:],4)

        # Standardization: remove the mean and divide the standard deviation
        data = data - np.mean(data,2)
        data = data / np.std(data,[],2)
        
        if seg == 1:
            fmritest.test1[:,:,rep] = data
        elif seg == 2:
            fmritest.test2[:,:,rep] = data    
        elif seg == 3:
            fmritest.test3[:,:,rep] = data        
        elif seg == 4:
            fmritest.test4[:,:,rep] = data
        elif seg == 5:
            fmritest.test5[:,:,rep] = data

# save fmritest
with open(f'{fmripath}testing_fmri.mat', 'w') as f:
    np.save(f, fmritest)
