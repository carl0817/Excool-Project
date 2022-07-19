import numpy as np

def amri_sig_r2z(r):
 
    # Fisher's r-to-z-transformatiom
    # z =.5.*np.log((1+r)./(1-r)) ; matlab version
    z = np.multiply(0.5, np.log(np.divide(1+r, 1-r)))

    return z