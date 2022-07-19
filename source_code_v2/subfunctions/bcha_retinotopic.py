## Calculate the eccentricity and polar angle of a location given its index in 2D image space
# inputs:
#   idx: indeces of locations in the 2D image
import numpy as np

def retinotopic(idx,s):
    if not s:
        s = 55 # size of the 2D image
    
    idx = np.array(idx)
    ct = round(s/2) # distance to the center location
    retin = np.zeros((idx.shape[0]/2, 2)) # the first column is eccentricity, the second is polar angle 
    for i in range(retin.shape[0]):
        num = sum(idx[(i)*2,:]>0)
        xy = idx[(i)*2:(i+1)*2, 0:num]
        xy = np.array(xy) - np.array([[ct],[ct]])
        xy[1,:] = -xy[1,:]
        d = np.sqrt(sum(np.square(xy))) # relative distance to the screen center

        # eccentricity
        retin[i,0] = np.arctan(np.mean(d)*0.1854/ct) # mean eccentricity of locations

        # quantify polar angle: use sin value with respect to the vertical axis 
        c = xy[0,:] / d
        c[np.isnan(c)] = []
        retin[i,1] =   np.mean(c)
    return retin