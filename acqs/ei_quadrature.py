import sys
# sys.path.insert(0, '/home/thanhnt/DeakinML/GPyOpt')

import numpy as np 
from scipy.stats import norm
import copy 
from copy import deepcopy
import os 
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import GPy 
from optimizer.lbfgs import minimize as lbfgs_minimize  

SMALL_POSITIVE_EPS = 1e-30

def ei_selection_routine(model, X_init, Y_init, horizon, objective_func, x_bound, **argv):
    """
    Args:
        model: GPyOpt.models.quadrature.GaussianFiniteQuadrature
    """
    print('=== Start EI selection routine ===')
    xi = argv['xi']

    y_min = np.min(Y_init)
    y_max = np.max(Y_init)

    X = np.copy(X_init)
    Y = (np.copy(Y_init) - y_min) / (y_max - y_min)

    model_copied = deepcopy(model)
    report_points = []
    report_values = []
    for t in range(horizon):
        print('t=%d'%(t))
        xt = ei_suggest(model_copied, x_bound, xi)
        yt = objective_func.evaluate(xt.reshape(-1)) # Objective evaluated at (d,)
        yt = (yt - y_min) / (y_max - y_min)
        fmax = model_copied.get_fmax()
        print('EI recommended point: ', xt)
        print('Best value so far: ', fmax)
        
        X = np.vstack((X, xt.reshape(1,-1)))
        Y = np.vstack((Y, yt))
        # model_copied.updateModel(X, Y) 
        
        model_backup = deepcopy(model_copied)
        try:
            model_copied.updateModel(X, Y)
        except np.linalg.LinAlgError as exc:
            print('np.linalg.LinAlgError happens when fitting GP:')
            print('GP MLE parameters:', model_copied.model.param_array)
            print('X_train: ', model_copied.model.X)
            print('Y_train: ', model_copied.model.Y)
            
            print('Reuse the last GP hyperparameters!')
            model_copied = deepcopy(model_backup)
            model_copied.model.set_XY(X, Y)
        print('GP MLE parameters:', model_copied.model.param_array)
        report_point, report_value = model_copied.get_q_report()
        print('Report_point: ', report_point, 'Report_value: ', report_value)
        report_points.append(report_point)
        report_values.append(report_value)
    return X, report_points, report_values

def ei_suggest(model, x_bound, xi):
    xt = ei_suggest_x(model, x_bound, xi)
    wt = suggest_explorative_w(model, xt)
    return np.hstack((xt, wt))

def suggest_explorative_w(model, x):
    # Based on the highest posterior variance. 
    x = x.reshape(1,-1)
    w_domain = model.w_domain
    xw = np.hstack((np.kron(x, np.ones((w_domain.shape[0], 1))), np.kron(np.ones((x.shape[0], 1)), w_domain))) 
    m, v = model.model.predict(xw, full_cov=False, include_likelihood=True)
    v = np.clip(v, 0, None)
    ind = np.argmax(v)
    return w_domain[ind:ind+1,:].reshape(1,-1)

def ei_suggest_x(model, x_bound, xi):
    def obj_func(x):
        if x.ndim <= 1:
            x = x.reshape(1,-1)
        f_acq = compute_ei(model, x, xi)
        return -f_acq

    xmin, yval = lbfgs_minimize(obj_func, 
        x_bound, 
        approx_grad=1, 
        maxiter=15000, 
        n_warmup=0, 
        n_restarts_optimizer=30
        )
    xt = xmin.reshape(1,-1)
    return xt

def compute_ei(model, x, xi):
    m, s = model.predict(x)
    fmax = model.get_fmax()
    s = np.clip(s, SMALL_POSITIVE_EPS, None)
    u = (m - fmax - xi)/s
    phi = norm.pdf(u)
    Phi = norm.cdf(u)
    f_acq = s * (u * Phi + phi) 
    return f_acq 

# def compute_ei_with_grad(model, x, xi):
#     m, s, dmdx, dsdx = model.predict_withGradients(x)
#     fmax = model.get_fmax()
#     s = np.clip(s, SMALL_POSITIVE_EPS, None)
#     u = (m - fmax - xi)/s
#     phi = norm.pdf(u)
#     Phi = norm.cdf(u)
#     f_acq = s * (u * Phi + phi) 
#     df_acqu = dsdx * phi + Phi * dmdx
#     return f_acq, df_acqu 