"""
This is to compute quadratureEI and quadratureTS in 
    tuning hyperaperamters of ElasticNet on unmanipulated MNIST using k-folds. 
"""


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
import GPy 
from .objective_function import CNN
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('./tmp/data/mnist', one_hot=True)
X_train = mnist.train.images
Y_train = mnist.train.labels
X_test = mnist.test.images
Y_test = mnist.test.labels


is_minimization_problem = False
noise_var = 0.0

def compute_onehot(ks,n):
    a = np.array(ks).astype('int')
    x = np.zeros((a.shape[0], n))
    x[range(a.shape[0]), a] = 1. 
    return x

def generate_init_x(init_data_dir, n_w, decision_dim, n_trials, seed):
    if not os.path.exists(init_data_dir):
        os.makedirs(init_data_dir) 

    n_init = 3*decision_dim

    x_bound = np.array([
                    [1e-8, 1], # dropout1 
                    [1e-8, 1], # dropout2
                    [0, 1] # lr in log10 scale, then scaled to [0,1]
                ]) 
    kern_w_domain = np.eye(n_w)

    np.random.seed(seed)
    for trial in range(n_trials):
        x = np.random.uniform(x_bound[:,0], x_bound[:,1], (n_init, decision_dim)) 
        np.savez(os.path.join(init_data_dir, 'X_init_trial=%d'%(trial+1)), x)
        w = kern_w_domain[np.random.randint(0, n_w, n_init), :].reshape(n_init, -1)
        np.savez(os.path.join(init_data_dir, 'W_init_trial=%d'%(trial+1)), w) 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--f', help="Objective function", default='ElasticNet')
    parser.add_argument('--dim', help="Dimension of the logistic", type=int, default=2)
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
    # parser.add_argument('--cv_size',help='Size of train-val MNIST data', type=int, default=1000)
    # parser.add_argument('--cls_prop', help='MNIST class proportion', default=[1, 2, 3, 4, 5, 6, 7, 8, 9 ,10]) 
    parser.add_argument('--data_shuffle', help='Whether to shuffle MNIST data', default=True)
    parser.add_argument('--log_alpha_min', help='Log alpha min in ElasticNet',type=float, default=-6)
    parser.add_argument('--log_alpha_max', help='Log alpha max in ElasticNet',type=float, default=3)


    parser.add_argument('--exact_feval', help='Whether optimize for Gaussain noise variance in GP',type=bool,default=True)

    args = parser.parse_args()
    
    # cls_prop = [10, 1, 1, 1, 1, 1, 1, 1, 1 ,10] # This is very extreme case of uneven data split
    # args.cls_prop = cls_prop 
    # cls_prop = np.array(args.cls_prop)
    
    # X_cv, Y_cv = mnist_data_split(X_train, Y_train, total_samples=args.cv_size, cls_prop=cls_prop, \
    #                               shuffle=args.data_shuffle)

    objective_function = CNN(X_train, Y_train, n_folds=args.n_w, minimize=is_minimization_problem, var_noise=noise_var, seed=1) 
    decision_dim = objective_function._dim 
    x_bound = objective_function._search_domain
    
    n_w = args.n_w
    w_dim = n_w 
    objective_function._n_w = n_w 
    kern_w_domain = np.eye(n_w)
    
    
    hybrid_dim = decision_dim + w_dim
    n_init = 3*decision_dim 
    horizon =  30*decision_dim if args.horizon == -1 else args.horizon
    n_trials = 30 if args.n_trials == -1 else args.n_trials
    trial_start = args.trial_start
    rho_list = [0.1, 0.3, 0.5, 1.0, 3.0, 5.0]
    
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

    exp_dir_prefix = os.path.join('./tmp/cnn/results/d=%d_nw=%d'%(decision_dim, n_w))
    tmp = 1
    exp_dir = exp_dir_prefix + '_run=%d'%(tmp)
    while os.path.exists(exp_dir):
        tmp += 1 
        exp_dir = exp_dir_prefix + '_run=%d'%(tmp)

    print('Exp dir = %s'%(exp_dir))
    data_dir = os.path.join(exp_dir, 'data')
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # np.savez(os.path.join(exp_dir, 'data_cv'), X_cv, Y_cv)
    
    exp_config = vars(args)
    print(exp_config)
    with open(os.path.join(exp_dir, 'exp_config.json'), 'w') as fp:
        json.dump(exp_config, fp)

    
    seed=2019
    init_data_dir = './tmp/cnn/init'
    if not os.path.exists(init_data_dir):
        generate_init_x(init_data_dir, n_w, decision_dim, n_trials, seed)

    np.random.seed(seed)
    x_init_list = []
    for trial in range(n_trials):
        # xs = np.random.uniform(x_bound[:,0], x_bound[:,1], (n_init, decision_dim)) 
        # np.savez(os.path.join(data_dir, 'X_init_trial=%d'%(trial+1)), xs)
        # ws = kern_w_domain[np.random.randint(0, n_w, n_init), :].reshape(n_init, -1)
        xs = np.load(os.path.join(init_data_dir, 'X_init_trial=%d.npz'%(trial+1)))['arr_0']
        ws = np.load(os.path.join(init_data_dir, 'W_init_trial=%d.npz'%(trial+1)))['arr_0']
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
                        exact_feval=args.exact_feval, 
                        optimize_restarts=100, 
                        verbose=False,
                        decision_dim=decision_dim,
                        w_domain=kern_w_domain)


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
        
        # gpmle.updateModel(X, Y) 
        
        gpmle_backup = deepcopy(gpmle)
        try:
            gpmle.updateModel(X, Y)
        except np.linalg.LinAlgError as exc:
            print('np.linalg.LinAlgError happens when fitting GP:')
            print('GP MLE parameters:', gpmle.model.param_array)
            print('X_train: ', gpmle.model.X)
            print('Y_train: ', gpmle.model.Y)
            
            print('Reuse the last GP hyperparameters!')
            gpmle = deepcopy(gpmle_backup)
            
            if gpmle.model is None:
                gpmle._create_model(X, Y)
            else:
                gpmle.model.set_XY(X, Y)
            
        print('GP MLE parameters:', gpmle.model.param_array)
        
        print('===RANDOM SELECTION===')
        rand_data = np.hstack(
            (
                np.random.uniform(x_bound[:,0], x_bound[:,1], (horizon, decision_dim)), 
                kern_w_domain[np.random.randint(0, n_w, horizon), :].reshape(horizon, -1)
            )
        )
        
        # RANDOM SEARCH
        rand_data = np.vstack((x_init, rand_data))
        report_values_list = []
        report_points_list = []
        
        # EI-BQO 
        ei_data, ei_report_points, ei_report_values = ei_selection_routine(gpmle, x_init, y_init, horizon, objective_function, x_bound, xi=xi)
        report_values_list.append(np.array(ei_report_values))
        report_points_list.append(np.array(ei_report_points))
        
        # TS-BQO
        ts_data, ts_report_points, ts_report_values = ts_selection_routine(gpmle, x_init, y_init, horizon, objective_function, x_bound, \
            m=n_spectral_points, epsilon=epsilon, kernel_type=kernel_type)
        report_values_list.append(np.array(ts_report_values))
        report_points_list.append(np.array(ts_report_points))
        

        # DRBQO
        drts_data_list = []
        for rho in rho_list:
            print('=========   rho= %s'%(str(rho)))
            drts_data, drts_report_points, drts_report_values = drts_selection_routine(gpmle, x_init, y_init, horizon, objective_function, x_bound, \
                        m=n_spectral_points, epsilon=epsilon, kernel_type=kernel_type, rho=rho)
            drts_data_list.append(drts_data)
            report_values_list.append(np.array(drts_report_values))
            report_points_list.append(np.array(drts_report_points))
        
        np.savez(os.path.join(data_dir, 'X_trial.%d'%(trial+1)), ei_data, ts_data, *tuple(drts_data_list), rand_data)
        np.savez(os.path.join(data_dir, 'report_pts_trial.%d'%(trial+1)), *tuple(report_points_list))
        np.savez(os.path.join(data_dir, 'report_values_trial.%d'%(trial+1)), *tuple(report_values_list))