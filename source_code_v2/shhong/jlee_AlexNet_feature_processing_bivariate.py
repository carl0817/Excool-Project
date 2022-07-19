# %%
import nibabel as nib
from nilearn import glm
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
from scipy.io import loadmat 
import h5py, datetime
from scipy.signal import convolve

# %%
layername = ['/conv1','/conv2','/conv3','/conv4','/conv5','/fc6','/fc7','/fc8']

# %%
srate = 30
p  = [5, 16, 1, 1, 6, 0, 32]
tr = 1/srate


# %% [markdown]
# # hrf
# reference: https://github.com/poldracklab/poldracklab-base/blob/master/fmri/spm_hrf.py

# %%
import scipy.stats
import numpy as N

def spm_hrf(TR,p=[5,16,1,1,6,0,32]):
    """ An implementation of spm_hrf.m from the SPM distribution
Arguments:
Required:
TR: repetition time at which to generate the HRF (in seconds)
Optional:
p: list with parameters of the two gamma functions:
                                                     defaults
                                                    (seconds)
   p[0] - delay of response (relative to onset)         6
   p[1] - delay of undershoot (relative to onset)      16
   p[2] - dispersion of response                        1
   p[3] - dispersion of undershoot                      1
   p[4] - ratio of response to undershoot               6
   p[5] - onset (seconds)                               0
   p[6] - length of kernel (seconds)                   32
"""

    p=[float(x) for x in p]

    fMRI_T = 16.0

    TR=float(TR)
    dt  = TR/fMRI_T
    u   = N.arange(p[6]/dt + 1) - p[5]/dt
    hrf=scipy.stats.gamma.pdf(u,p[0]/p[2],scale=1.0/(dt/p[2])) - scipy.stats.gamma.pdf(u,p[1]/p[3],scale=1.0/(dt/p[3]))/p[4]
    good_pts=N.array(range(N.int(p[6]/TR)))*int(fMRI_T)
    hrf=hrf[list(good_pts)]    
    hrf = hrf/N.sum(hrf);
    return hrf

# %%
poldrack_hrf = spm_hrf(tr,p)
print(poldrack_hrf.shape)

poldrack_hrf = np.expand_dims(poldrack_hrf, -1)
print(poldrack_hrf.shape)

print(poldrack_hrf.dtype)
poldrack_hrf = np.array(poldrack_hrf, dtype=np.float32)
print(poldrack_hrf.dtype)

# # %%
# mat_file = loadmat('./hrf_variable.mat')
# mat_hrf = np.squeeze(mat_file['hrf'])
# print(mat_hrf.shape)
# plt.plot(mat_hrf)

# # %%
# print(poldrack_hrf.shape)
# plt.plot(poldrack_hrf)

# %% [markdown]
# # run_bivariate.m_training

# %%
dataroot = '/local_raid3/03_user/jungmin/02_data/Encoding_decoding/encoding_analyzing/02_Goal-Driven-DL/01_NeuralEncodingDecoding/sourcecode/source_code_v2/source_code_v2/feature_extracted/'
saveroot = '/local_raid3/03_user/jungmin/02_data/Encoding_decoding/encoding_analyzing/02_Goal-Driven-DL/01_NeuralEncodingDecoding/sourcecode/source_code_v2/source_code_v2/jlee_feature_extracted/'

# %%
def print_time():
    now = datetime.datetime.now()
    print("{}Y-{}M-{}D, {}H-{}m-{}s".format(now.year, now.month, now.day, now.hour, now.minute, now.second))

# %%
for seg in range(1,19):
    secpath = dataroot + 'AlexNet_feature_maps_seg'+ str(seg)+'.h5'

    for lay in range(0,len(layername)):
        print('Seg: ',  str(seg) , '; Layer: ',layername[lay])
        print_time()
        lay_feat = h5py.File(secpath, 'r')
        lay_feat = lay_feat[layername[lay] + '/data']
        lay_feat = lay_feat.value
        dim = lay_feat.shape
        print(dim)
        Nu = np.prod(dim[1:])
        print(Nu)
        Nf = dim[0]
        print(Nf)
        lay_feat = np.reshape(lay_feat,(Nf,Nu))
        print(lay_feat.shape) 
        print_time()
        if lay < len(layername):   
            lay_feat = np.log10(lay_feat + 0.01)
        print("Log10 applied!")
        print_time()     
        
        ts = convolve(poldrack_hrf, lay_feat)
        print("Convold Done!")
        print_time()
        ts = ts[4*srate+1:4*srate+Nf, :]
        down_sample_idx = np.arange(srate+1, len(ts), srate*2)
        ts = ts[down_sample_idx]
        print("Down sampled ts: {}".format(ts.shape))
        print_time()
        
        foldpath = saveroot+'AlexNet_feature_maps_processed_seg' + str(seg)+'.h5'
        f = h5py.File(foldpath, 'w')
        g = f.create_group(layername[lay])
        data = g.create_dataset("data", data=ts)
        print("Saved: {}".format(foldpath))
        print_time()
        f.close()

# %%
# Checking
# load_w = h5py.File(foldpath, 'r')
# load_w[layername[lay] + '/data'].value.shape


# %% [markdown]
# # run_bivariate.m_testing

# %%
for test in range(1,6):
    secpath = dataroot + 'AlexNet_feature_maps_test'+ str(test)+'.h5'

    for lay in range(0,len(layername)):
        print('Test: ',  str(test) , '; Layer: ',layername[lay])
        print_time()
        lay_feat = h5py.File(secpath, 'r')
        lay_feat = lay_feat[layername[lay] + '/data']
        lay_feat = lay_feat.value
        dim = lay_feat.shape
        print(dim)
        Nu = np.prod(dim[1:])
        print(Nu)
        Nf = dim[0]
        print(Nf)
        lay_feat = np.reshape(lay_feat,(Nf,Nu))
        print(lay_feat.shape) 
        print_time()     
        if lay < len(layername):   
            lay_feat = np.log10(lay_feat + 0.01)
        print("Log10 applied!")
        print_time()     

        ts = convolve(poldrack_hrf, lay_feat)
        print("Convold Done!")
        print_time()
        ts = ts[4*srate+1:4*srate+Nf, :]
        down_sample_idx = np.arange(srate+1, len(ts), srate*2)
        ts = ts[down_sample_idx]
        print("Down sampled ts: {}".format(ts.shape))
        print_time()

        foldpath = saveroot+'AlexNet_feature_maps_processed_test' + str(test)+'.h5'
        f = h5py.File(foldpath, 'w')
        g = f.create_group(layername[lay])
        data = g.create_dataset("data", data=ts)
        print("Saved: {}".format(foldpath))
        print_time()
        f.close()
        
