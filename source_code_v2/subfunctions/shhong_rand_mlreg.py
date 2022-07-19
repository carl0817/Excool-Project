# compiled by sh Hong

import numpy as np

def rand_mlreg(dimV, dimH, init, sigma):

    if init.lower() == 'normal':

        para.W = sigma*np.random.randn(dimV, dimH)

    else:
        para.W = (2*np.random.rand(dimV, dimH)-1)/(dimV+dimH)


    para.b = np.zeros(1, dimH)

return para