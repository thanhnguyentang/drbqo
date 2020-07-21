import sys
# sys.path.insert(0, '/home/thanhnt/DeakinML/GPyOpt')

import numpy as np 
from copy import deepcopy

from .TS import TS_posterior_func_sample
from optimizer.lbfgs import minimize as lbfgs_minimize  
from optimizer.direct import minimize as direct_minimize  
from utils.compute_dr_weight import min_weighted_sum_with_weights_on_chi_squared_ball


def drts_selection_routine(model, X_init, Y_init, horizon, objective_func, x_bound, **kargv):
    """
    args:
        model: GPyOpt.models.quadrature.GaussianFiniteQuadrature
    """
    print('=== Start DRTS selection routine ===')
    m = kargv['m']
    epsilon = kargv['epsilon']
    kernel_type = kargv['kernel_type']
    rho = kargv['rho']

    y_min = np.min(Y_init)
    y_max = np.max(Y_init)

    X = np.copy(X_init)
    Y = (np.copy(Y_init) - y_min) / (y_max - y_min)
    report_points = []
    report_values = []
    model_copied = deepcopy(model)
    for t in range(horizon):
        print('t=%d'%(t))
        xt, report_value = ts_suggest(model_copied, epsilon , x_bound, m, rho, kernel_type)
        # report_values.append(report_value)
        yt = objective_func.evaluate(xt.reshape(-1)) # Objective evaluated at (d,)
        yt_unscaled = yt 
        yt = (yt - y_min) / (y_max - y_min)
        print('DRTS recommended point: ', xt)
        print('Unscaled value at recommended point: %.5f'%(yt_unscaled))

        # print('Report value: %.5f'%(report_value))
        
        X = np.vstack((X, xt.reshape(1,-1)))
        Y = np.vstack((Y, yt))
        # model_copied.updateModel(X, Y) 
        # BUG: When fitting the TS recommended points to GP, 
        # np.linalg.LinAlgError happens due to non positive definite matrix. I don't know why. 
        # Temporial solution: If it happens, throw it away and try again until it's good to go. 
        model_copied_backup = deepcopy(model_copied)
        try:
            model_copied.updateModel(X, Y)
        except np.linalg.LinAlgError as exc:
            print('np.linalg.LinAlgError happens when fitting GP:')
            print('GP MLE parameters:', model_copied.model.param_array)
            print('X_train: ', model_copied.model.X)
            print('Y_train: ', model_copied.model.Y)
            
            print('Reuse the last GP hyperparameters!')
            model_copied = deepcopy(model_copied_backup)
            model_copied.model.set_XY(X, Y)
        print('GP MLE parameters:', model_copied.model.param_array)
        report_point, report_value = model_copied.get_q_robust_report(rho=rho)
        print('Report_point: ', report_point, 'Report_value: ', report_value)      
        report_points.append(report_point)
        report_values.append(report_value)
    return X, report_points, report_values

def ts_suggest(model, epsilon, x_bound, m = 1000, rho=0.1, kernel_type='RBF'):
    xt, report_value = ts_suggest_x(model, epsilon, x_bound, m = m, rho=rho, kernel_type=kernel_type)
    wt = suggest_explorative_w(model, xt)
    return np.hstack((xt, wt)), report_value

def suggest_explorative_w(model, x):
    # Based on the highest posterior variance. 
    x = x.reshape(1,-1)
    w_domain = model.w_domain
    xw = np.hstack((np.kron(x, np.ones((w_domain.shape[0], 1))), np.kron(np.ones((x.shape[0], 1)), w_domain))) 
    m, v = model.model.predict(xw, full_cov=False, include_likelihood=True)
    v = np.clip(v, 0, None)
    ind = np.argmax(v)
    return w_domain[ind:ind+1,:].reshape(1,-1)

def ts_suggest_x(model, epsilon, x_bound, m = 1000, rho=0.1, kernel_type='RBF'):
    """
    args:
        model: GPyOpt.models.quadrature.GaussianFiniteQuadrature
    """

    data = (model.model.X, model.model.Y)
    posterior_f = TS_posterior_func_sample(model, m, 1, data, kernel_type=kernel_type)[0]
    def obj_func(x):
        x = x.reshape(1,-1)
        w_domain = model.w_domain
        xw = np.hstack((np.kron(x, np.ones((w_domain.shape[0], 1))), np.kron(np.ones((x.shape[0], 1)), w_domain))) 
        ts_vals = posterior_f(xw)
        _, ts_val = min_weighted_sum_with_weights_on_chi_squared_ball(ts_vals, rho=rho)
        return -ts_val

    if np.random.uniform() < epsilon:
        xt = np.random.uniform(x_bound[:,0], x_bound[:,1], (1, x_bound.shape[0]))
        report_value = obj_func(xt)
        return xt, report_value
    
    xmin, yval = lbfgs_minimize(obj_func, 
        x_bound, 
        approx_grad=1, 
        maxiter=15000, 
        n_warmup=0, 
        n_restarts_optimizer=30
        )
    xt = xmin.reshape(1,-1)

    return xt, -yval
    