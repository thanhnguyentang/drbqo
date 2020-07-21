"""A convenient wrapper for various inner-optimization for acquisition functions. 

# Thanh Tang Nguyen <thanhnt@deakin.edu.au>, <nguyent2792@gmail.com>
"""

import numpy as np 
from DIRECT import solve

def minimize(func, 
        bounds, 
        maxiter=15000, 
        n_restarts_optimizer=10
        ):
    """
    # Arguments
        func: if `approx_grad`=0, `func` returns function values and gradients. Otherwise, it returns only function values. `
    
    Note: the backend minimizer evaluates `func` at each single input;thus `func` does not have to be able to perform sample-wise evaluation. 
    When `func` cannot be evaluated at a batch of samples, simply set `n_warmup` = 0. 
    """
    def real_func(x, user_data):
        return func(x), 0
    xmins = []
    fmins = []
    for i in range(n_restarts_optimizer):
        xmin, fmin, _ =  solve(real_func, l=bounds[:,0], u=bounds[:,1], eps=1e-4, maxf=2000, \
                        maxT=6000 if maxiter is None else maxiter, 
                        algmethod=0, fglobal=-1e100, fglper=0.01, volper=-1.0,
                        sigmaper=-1.0, logfilename='DIRresults.txt', user_data=None)
        xmins.append(xmin)
        fmins.append(fmin)
    ind = np.argmin(fmins)
    return xmins[ind], fmins[ind]
    
