import numpy as np
from bcha_spm_Gpdf import bcha_spm_Gpdf

def bcha_spm_hrf(RT, p=[6,16,1,1,6,0,32]):
    # Returns a hemodynamic response function
    # FORMAT [hrf,p] = spm_hrf(RT,[p])
    # RT   - scan repeat time
    # p    - parameters of the response function (two gamma functions)
    #
    #                                                     defaults
    #                                                    (seconds)
    #   p(1) - delay of response (relative to onset)         6
    #   p(2) - delay of undershoot (relative to onset)      16
    #   p(3) - dispersion of response                        1
    #   p(4) - dispersion of undershoot                      1
    #   p(5) - ratio of response to undershoot               6
    #   p(6) - onset (seconds)                               0
    #   p(7) - length of kernel (seconds)                   32
    #
    # hrf  - hemodynamic response function
    # p    - parameters of the response function
    #__________________________________________________________________________
    # Copyright (C) 2008 Wellcome Trust Centre for Neuroimaging

    # Karl Friston
    # $Id: spm_hrf.m 3716 2010-02-08 13:58:09Z karl $


    # global parameter
    #--------------------------------------------------------------------------
    try:
        fMRI_T = spm_get_defaults('stats.fmri.t')
    except:
        fMRI_T = 16

    # default parameters
    #--------------------------------------------------------------------------
    # do not need to use nargin in python,
    # because we can define optional arguments in python
    # directly in the function definition

    # modelled hemodynamic response function - {mixture of Gammas}
    #--------------------------------------------------------------------------
    dt  = RT/fMRI_T
    # u   = [0:(p[6]/dt)] - p[5]/dt
    u   = {*range(0, p[6]/dt+1, 1)}
    hrf = bcha_spm_Gpdf(u,p[0]/p[2],dt/p[2]) - bcha_spm_Gpdf(u,p[1]/p[3],dt/p[3])/p[4]
        # subtraction of two gamma functions to get HRF
    # hrf = hrf([0:(p[6]/RT)]*fMRI_T + 1)
    hrf = np.array(hrf)
    hrf = hrf[{*range(0, p[6]/RT+1, 1)}*fMRI_T + 1]
    hrf = hrf.transpose/sum(hrf)

    return [hrf,p]