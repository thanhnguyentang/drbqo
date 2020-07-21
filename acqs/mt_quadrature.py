"""
Multi-task: sConditioned on x, we then choose the task that yields the highest single-task expected
improvement.
https://papers.nips.cc/paper/5086-multi-task-bayesian-optimization.pdf 
"""
import sys
# sys.path.insert(0, '/home/thanhnt/DeakinML/GPyOpt')

import numpy as np 
from scipy.stats import norm
from copy import deepcopy
import os 
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import GPy 
from optimizer.lbfgs import minimize as lbfgs_minimize  

SMALL_POSITIVE_EPS = 1e-30

def mt_selection_routine(model, X_init, Y_init, horizon, objective_func, x_bound, **argv):
    """
    Args:
        model: GPyOpt.models.quadrature.GaussianFiniteQuadrature 
        with mt kernel 

        w_domain: [0,1,...,k-1]
    """
    print('=== Start MT selection routine ===')
    xi = argv['xi']

    y_min = np.min(Y_init)
    y_max = np.max(Y_init)

    X = np.copy(X_init)
    Y = (np.copy(Y_init) - y_min) / (y_max - y_min)

    model_copied = deepcopy(model)
    report_points = []
    report_values = []
    curr_values = []
    for y_init in Y_init:
        curr_values.append(y_init)
    for t in range(horizon):
        print('t=%d'%(t))
        xt = mt_suggest(model_copied, x_bound, xi)
        yt = objective_func.evaluate_mt(xt.reshape(-1)) # Objective evaluated at (d,)
        curr_values.append(np.copy(yt))
        yt = (yt - y_min) / (y_max - y_min)
        fmax = model_copied.get_fmax()
        print('MT recommended point: ', xt)
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
    return X, np.array(report_points), np.array(report_values), np.array(curr_values)

def mt_suggest(model, x_bound, xi):
    # print('mt_suggest')
    xt = mt_suggest_x(model, x_bound, xi)
    # wt = suggest_explorative_w(model, xt)
    wt = highest_ei_singletask(model, xt, xi)
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

def highest_ei_singletask(model, x, xi):
    # print('highest_ei_singletask')
    facqs = []
    fmax = model.model.Y.max()
    for w in model.w_domain:
        xw = np.hstack((x.reshape(1,-1), w.reshape(1,-1)))
        m, v = model.model.predict(xw, full_cov=False, include_likelihood=True)
        facq = get_quantile_ei(m, np.sqrt(np.clip(v, 1e-30, None)), fmax, xi)
        facqs.append(facq)
    ind = np.argmax(facqs)
    # print('done highest_ei_singletask')
    return model.w_domain[ind:ind+1,:].reshape(1,-1)

def mt_suggest_x(model, x_bound, xi):
    # print('mt_suggest_x')
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
        n_restarts_optimizer=50
        )
    xt = xmin.reshape(1,-1)
    return xt

def compute_ei(model, x, xi):
    # print('compute_ei')
    m, s = model.predict(x)
    fmax = model.get_fmax()
    return get_quantile_ei(m,s, fmax, xi) 

def get_quantile_ei(m,s, fmax, xi):
    # print('get_quantile_ei')
    u = (m - fmax - xi)/ np.clip(s, SMALL_POSITIVE_EPS, None)
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