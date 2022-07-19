# This code is for evaluating the reproducibility of the response to the same movie stimuli
#
# Reference: 
# Wen H, Shi J, Zhang Y, Lu KH, Cao J, and Liu Z. (2017) Neural Encoding
# and Decoding with Deep Learning for Dynamic Natural Vision. Cerebral
# Cortex, In press.

# history
# v1.0 (original version) --2017/09/13

# compiled by sh Hong

# Reproducibility analysis
# Load fMRI responses during watching the movie for the first and second time
# Nv: number of voxels, Nv = 59412.
# Nt: number of volumes for one movie segment, Nt = 240.
# Ns: number of movie segments, Ns = 18.

import numpy as np
import sys

sys.path.append('/local_raid1/03_user/sunghyoung/01_project/03_Encoding_decoding/01_NeuralEncodingDecoding/sourcecode/source_code_v2/source_code_v2/subfunctions')

import subfunctions

with open('/path/to/subject1/fmri/training_fmri.mat') as f:

   fmri = np.load(f, 'fmri') # from movie_fmri_processing.m
  
# Calculate the voxelwise correlation between the responses to the same movie for the
# first time and the second time.
Rmat = np.zeros(Nv, Ns) 

for seg in range(Ns):

   dt1 = fmri.data1[:,:,seg]
   dt2 = fmri.data2[:,:,seg]
   Rmat[:,seg] = subfunctions.amri_sig_corr(dt1',dt2','mode','auto') 


# Fisher's r-to-z-transformation
Zmat = subfunctions.amri_sig_r2z(Rmat)
