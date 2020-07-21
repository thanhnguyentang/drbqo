"""A convenient wrapper for various inner-optimization for acquisition functions. 

# Thanh Tang Nguyen <thanhnt@deakin.edu.au>, <nguyent2792@gmail.com>
"""

import numpy as np 
import scipy.optimize 

def minimize(func, 
        bounds, 
        approx_grad=0, 
        maxiter=15000, 
        n_warmup=100000, 
        n_restarts_optimizer=10, 
        initializer=None, 
        x_init=None, 
        random_state=None):
    """
    # Arguments
        func: if `approx_grad`=0, `func` returns function values and gradients. Otherwise, it returns only function values. `
    
    Note: the backend minimizer evaluates `func` at each single input;thus `func` does not have to be able to perform sample-wise evaluation. 
    When `func` cannot be evaluated at a batch of samples, simply set `n_warmup` = 0. 
    """
    if initializer is None:
        initializer = random_sampler
    if n_warmup > 0:
        X0 = latin_hypercube_sampler(n_warmup, bounds.shape[0], bounds, random_state)
        if approx_grad:
            ys = func(X0)
        else: 
            ys, _ = func(X0)
        xmin = X0[ys.argmin()]
        fmin = ys.min()
    else:
        xmin = None
        fmin = np.inf
        # Actual optimize
    X0 = latin_hypercube_sampler(n_restarts_optimizer, bounds.shape[0], bounds, random_state)
    if n_warmup > 0:
        X0 = np.vstack((xmin.reshape(1,-1), X0))
    if x_init is not None:
        X0 = np.vstack((x_init.reshape(1,-1), X0))
    for j, x0 in enumerate(X0):
        if maxiter is not None:
            _xmin, _fmin, d = scipy.optimize.fmin_l_bfgs_b(func, x0.reshape(1,-1), approx_grad=approx_grad, bounds=bounds, maxiter=maxiter)
        else:
            _xmin, _fmin, d = scipy.optimize.fmin_l_bfgs_b(func, x0.reshape(1,-1), approx_grad=approx_grad, bounds=bounds)

        if _fmin < fmin:
            fmin = _fmin
            xmin = _xmin
    xmin = np.clip(xmin, bounds[:,0], bounds[:,1])
    return xmin, fmin 

def random_sampler(n=1, indim=1, bounds=None, rng=None):
    rng = ensure_rng(rng)
    if bounds is None:
        bounds = np.zeros((indim, 2))
        bounds[:,1] = 1. 
    return rng.uniform(bounds[:,0], bounds[:,1], (n, indim))

def latin_hypercube_sampler(n=1, indim=1, bounds=None, rng=None):
    """Latin Hypercube sampling. 

    Samples in random sampling are independent of each other while they are not in latin hypercube sampling. 
    Samples from latin hypercube sampling better cover the input space. 
    """
    rng = ensure_rng(rng)
    if bounds is None:
        bounds = np.zeros((indim, 2))
        bounds[:,1] = 1. 
    # Divide each dimension into `n` equal intervals
    hypercubes = np.linspace(bounds[:,0], bounds[:,1], n+1)
    
    l = hypercubes[:-1,:].reshape(-1,)
    u = hypercubes[1:,:].reshape(-1,)
    _x = rng.uniform(l,u, (1, indim*n)).reshape(n, indim)
    x = _x
    for j in range(indim):
        x[:,j] = _x[rng.permutation(n), j]
    return x

def ensure_rng(random_state=None):
    """
    Creates a random number generator based on an optional seed.  This can be
    an integer or another random state for a seeded rng, or None for an
    unseeded rng.
    """
    if random_state is None:
        random_state = np.random.RandomState()
    elif isinstance(random_state, int):
        random_state = np.random.RandomState(random_state)
    else:
        assert isinstance(random_state, np.random.RandomState)
    return random_state