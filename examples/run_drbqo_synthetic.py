"""
This example code demostrates how to run EI-BQO, TS-BQO, KG-BQO and DRBQO.  

# Contact: Thanh Tang Nguyen <thanhnt@deakin.edu.au>, <nguyent2792@gmail.com>
"""
import sys
import numpy as np 
from scipy.stats import norm
import os 
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from copy import deepcopy 
import random 
import pickle 
import argparse
import json

from utils.my_plots import plot_acquisition_1d
from utils.parallel import run_function_different_arguments_parallel
from models.quadrature import GaussianFiniteQuadrature  

from acqs.ei_quadrature import ei_selection_routine
from acqs.ts_quadrature import ts_selection_routine
from acqs.kg_quadrature import kg_selection_routine
from acqs.ts_drquadrature import drts_selection_routine  

from .synthetic_function import LogisticLoss_nD

import GPy 


is_minimization_problem = False
noise_var = 0.0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--f', help="Objective function", default='Logistic_nD')
    parser.add_argument('--dim', help="Dimension of the logistic", type=int, default=1)
    parser.add_argument('--n_w', help="Number of w samples", type=int, default=10)

    parser.add_argument('--kernel_type', help='[RBF, Matern32, Matern52]', default='RBF')
    parser.add_argument('--horizon', help="Number of BO iterations",type=int, default=-1)
    parser.add_argument('--n_trials', help="Number of experiment trials",type=int, default=30)
    parser.add_argument('--trial_start', help="Start from this trial",type=int, default=0)

    parser.add_argument('--n_adam_steps', help="Number of Adam steps",type=int, default=100)
    parser.add_argument('--n_z_samples_for_approx', help="Number of z samples to compute VOI",type=int, default=300)
    parser.add_argument('--multi_start', help="Number of Adam restarts",type=int, default=10)
    parser.add_argument('--lr', help="Adam learning rate",type=float, default=0.1)
    parser.add_argument('--batch_size', help="Adam batch size",type=int, default=10)
    parser.add_argument('--epsilon', help="Thompson epsilon",type=float, default=0.1)
    parser.add_argument('--n_spectral_points', help="Number of spectral points to estimate spectral density in Thompson sampling",\
                        type=int, default=1000)
    parser.add_argument('--xi', help="xi in EI",type=float, default=0.1)
    parser.add_argument('--DEBUG', help="DEBUG mode",type=bool, default=False)
    parser.add_argument('--method_to_debug', help="['EI', 'TS', 'KG'] Method to be debug (only effective when DEBUG=True", default='EI')


    args = parser.parse_args()

    objective_function = LogisticLoss_nD(is_minimization_problem, noise_var, args.dim)
    decision_dim = objective_function._dim 
    x_bound = objective_function._search_domain
    
    w_dim = objective_function._dim 
    n_w = args.n_w
    w_domain = np.random.normal(loc=0, scale=4.0, size=(n_w, args.dim)).reshape(n_w, -1)
    print(w_domain)
    objective_function._w_domain = w_domain
    objective_function._n_w = n_w 
    
    hybrid_dim = decision_dim + w_dim
    n_init = 5*decision_dim 
    horizon =  50*decision_dim if args.horizon == -1 else args.horizon
    n_trials = 30 if args.n_trials == -1 else args.n_trials
    trial_start = args.trial_start
    # rho_list = [0.1, 0.5, 1.0, 5.0]
    # rho_list = 10**(np.linspace(np.log10(0.01), np.log10(10), 10) )
    if n_w == 10:
        rho_list = [0.1, 0.3, 0.5, 1.0, 3.0, 5.0]
    elif n_w == 15:
        rho_list = [0.1, 0.4, 0.8, 1.0, 4.0, 8.0]
    else:
        raise NotImplementedError
    print(rho_list)
    # array([ 0.01      ,  0.02154435,  0.04641589,  0.1       ,  0.21544347,
    #     0.46415888,  1.        ,  2.15443469,  4.64158883, 10.        ])
    
    if args.DEBUG:
        horizon = 1 
        n_trials = 1 
        trial_start = 0 
        rho_list = [0.1]
    # EI 
    xi = args.xi

    # TS 
    epsilon = args.epsilon
    n_spectral_points = args.n_spectral_points
    kernel_type = args.kernel_type

    # KG 
    # batch_size = args.batch_size
    # lr = args.lr
    # multi_start = args.multi_start
    # n_z_samples_for_approx = args.n_z_samples_for_approx
    # n_adam_steps = args.n_adam_steps

    exp_dir_prefix = os.path.join('/home/thanhnt/DeakinML/GPyOpt/experiment/logistic/results/drquadrate/d=%d_nw=%d'%(decision_dim, n_w))
    tmp = 1
    exp_dir = exp_dir_prefix + '_run=%d'%(tmp)
    while os.path.exists(exp_dir):
        tmp += 1 
        exp_dir = exp_dir_prefix + '_run=%d'%(tmp)

    print('Exp dir = %s'%(exp_dir))
    data_dir = os.path.join(exp_dir, 'data')
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    np.savez(os.path.join(exp_dir, 'w_domain'), w_domain)
    
    exp_config = vars(args)
    print(exp_config)
    with open(os.path.join(exp_dir, 'exp_config.json'), 'w') as fp:
        json.dump(exp_config, fp)

    np.random.seed(2019)
    x_init_list = []
    for trial in range(n_trials):
        xs = np.random.uniform(x_bound[:,0], x_bound[:,1], (n_init, decision_dim)) 
        ws = w_domain[np.random.randint(0, n_w, n_init), :].reshape(n_init, -1)
        xws = np.hstack((xs, ws))
        x_init_list.append(np.copy(xws))
    
    for trial in range(trial_start, n_trials):
        print('trial = %d'%(trial))
        if args.kernel_type == 'RBF':
            kern = GPy.kern.RBF(hybrid_dim, variance=1., ARD=True)
        elif args.kernel_type == 'Matern32':
            kern = GPy.kern.Matern32(hybrid_dim, variance=1., ARD=True)
        elif args.kernel_type == 'Matern52':
            kern = GPy.kern.Matern52(hybrid_dim, variance=1., ARD=True)
        else:
            raise ValueError(args.kernel_type)
        
        gpmle = GaussianFiniteQuadrature(kernel=kern, 
                        optimize_restarts=100, 
                        verbose=False,
                        decision_dim=decision_dim,
                        w_domain=w_domain)


        x_init = x_init_list[trial]
        y_init = []
        for x_i in x_init:
            y_i = objective_function.evaluate(x_i.reshape(-1))
            y_init.append(y_i)
        y_init = np.array(y_init).reshape(-1,1) # Unscaled Y 

        X = np.copy(x_init)
        y_min = np.min(y_init)
        y_max = np.max(y_init)
        Y = (np.copy(y_init) - y_min) / (y_max - y_min) # Scaled Y 
        gpmle.updateModel(X, Y) 
        print('GP MLE parameters:', gpmle.model.param_array)

        print('===RANDOM SELECTION===')
        rand_data = np.hstack(
            (
                np.random.uniform(x_bound[:,0], x_bound[:,1], (horizon, decision_dim)), 
                w_domain[np.random.randint(0, n_w, horizon), :].reshape(horizon, -1)
            )
        )
        

        # Random
        rand_data = np.vstack((x_init, rand_data))

        # EI quadrature
        ei_data, ei_report_points, ei_report_values = ei_selection_routine(gpmle, x_init, y_init, horizon, objective_function, x_bound, xi=xi)

        # TS quadrature 
        ts_data, ts_report_points, ts_report_values = \
            ts_selection_routine(gpmle, x_init, y_init, horizon, objective_function, x_bound, \
            m=n_spectral_points, epsilon=epsilon, kernel_type=kernel_type)
        
        # DRBQO  
        drts_data_list = []
        for rho in rho_list:
            print('=========   rho= %s'%(str(rho)))
            drts_data, drts_report_points, drts_report_values \
                = drts_selection_routine(gpmle, x_init, y_init, horizon, objective_function, x_bound, \
                        m=n_spectral_points, epsilon=epsilon, kernel_type=kernel_type, rho=rho)
            drts_data_list.append(drts_data)

        ## Knowledge Gradient (KG) quadrature
        # kg_data = kg_selection_routine(gpmle, x_init, y_init, horizon, objective_function[args.f], x_bound, \
        #     multi_start=multi_start, lr=lr, batch_size=batch_size, \
        #         n_z_samples_for_approx = n_z_samples_for_approx, n_adam_steps=n_adam_steps)
        
        np.savez(os.path.join(data_dir, 'X_trial.%d'%(trial+1)), rand_data, ei_data, ts_data, *tuple(drts_data_list))
     
 