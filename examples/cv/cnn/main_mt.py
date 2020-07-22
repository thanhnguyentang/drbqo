"""
The original experiment: results/quadrate/d=2_nw=5_run=2 runs for T = 40 and n_init = 6. 

This script extends further to T = 150 to check if DRBQO takes a more perceivable effect. 
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

from utils.plotting.my_plots import plot_acquisition_1d
from utils.parallel import run_function_different_arguments_parallel
from models.quadrature import GaussianFiniteQuadrature  

from acqs.mt_quadrature import mt_selection_routine
 
import GPy 
from .objective_function import CNN



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
    parser.add_argument('--dim', help="Dimension of the logistic", type=int, default=3)
    parser.add_argument('--n_w', help="Number of w samples", type=int, default=10)
    parser.add_argument('--data', default='mnist')

    parser.add_argument('--kernel_type', help='[RBF, Matern32, Matern52]', default='mt')
    parser.add_argument('--horizon', help="Number of BO iterations",type=int, default=90)
    parser.add_argument('--n_trials', help="Number of experiment trials",type=int, default=30)
    parser.add_argument('--trial_start', help="Start from this trial",type=int, default=0)
    parser.add_argument('--trial_end', help="Start from this trial",type=int, default=30)

    parser.add_argument('--n_init',type=int, default=9)


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


    parser.add_argument('--exact_feval', help='Whether optimize for Gaussain noise variance in GP',type=bool,default=True)

    args = parser.parse_args()

    main_dir = './tmp/cnn'

    if args.data == 'letter':
        X_train = np.load(os.path.join(main_dir, 'data/letter', 'X_train.npz'))['arr_0']
        Y_train = np.load(os.path.join(main_dir, 'data/letter', 'Y_train.npz'))['arr_0']
        X_test = np.load(os.path.join(main_dir, 'data/letter', 'X_test.npz'))['arr_0']
        Y_test = np.load(os.path.join(main_dir, 'data/letter', 'Y_test.npz'))['arr_0']
    
    elif args.data == 'hiv':
        train_obj = np.load(os.path.join(main_dir, 'data/hiv_protease/hiv1_protease_cleave_train.npz'))
        X_train = train_obj['arr_0']
        Y_train = train_obj['arr_1']

        test_obj =  np.load(os.path.join(main_dir, 'data/hiv_protease/hiv1_protease_cleave_test.npz'))
        X_test = test_obj['arr_0']
        Y_test = test_obj['arr_1']
    elif args.data == 'glass':
        X_train = np.load(os.path.join(main_dir, 'data/glass', 'X_train.npz'))['arr_0']
        Y_train = np.load(os.path.join(main_dir, 'data/glass', 'Y_train.npz'))['arr_0']
        X_test = np.load(os.path.join(main_dir, 'data/glass', 'X_test.npz'))['arr_0']
        Y_test = np.load(os.path.join(main_dir, 'data/glass', 'Y_test.npz'))['arr_0']
    elif args.data == 'mnist':
        from tensorflow.examples.tutorials.mnist import input_data
        mnist = input_data.read_data_sets('./mnist', one_hot=True)
        X_train = mnist.train.images
        Y_train = mnist.train.labels
        X_test = mnist.test.images
        Y_test = mnist.test.labels

    objective_function = CNN(X_train, Y_train, args.n_w, is_minimization_problem)
    decision_dim = objective_function._dim 
    x_bound = objective_function._search_domain

    class ObjectiveFunction(object):
        def evaluate_mt(self, x):
            return objective_function.evaluate_mt(x) 
    FUNC = ObjectiveFunction()

    n_w = args.n_w
    w_dim = n_w 
    kern_w_domain = np.arange(n_w).reshape((-1,1))
    
    
    hybrid_dim = decision_dim + w_dim
    n_init = args.n_init #10 #3*decision_dim 
    horizon =  args.horizon
    n_trials = args.n_trials
    trial_start = args.trial_start
    trial_end = args.trial_end
    # rho_list = [0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0]
    # rho_list = np.exp(np.linspace(np.log(0.05), np.log(5.0), 10) )
    if n_w == 5: 
        rho_list = [0.05, 0.1, 0.5, 1, 2, 3]
    elif n_w == 10:
        rho_list = [0.1, 0.3, 0.5, 1.0, 3.0, 5.0]
    elif n_w == 15:
        rho_list = [0.1, 0.4, 0.8, 1.0, 4.0, 8.0]
    else:
        raise NotImplementedError
    print(rho_list)
    
    if args.DEBUG:
        horizon = 5
        n_trials = 1
        trial_start = 0 
        trial_end = 1
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

    exp_dir_prefix = os.path.join(main_dir, 'results/%s/mt/d=%d_nw=%d'%(args.data, decision_dim, n_w))
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
    np.random.seed(seed)
    # x_init_list = []
    # p_to_init = os.path.join(main_dir, 'results/init_d.%d_nw.%d_ninit.%d_ntrials.%d'%(decision_dim, n_w, n_init, args.n_trials))
    
    # if not os.path.exists(p_to_init):
    #     print('%s does not exist, creating one.'%(p_to_init))
    #     os.makedirs(p_to_init)

    #     np.random.seed(2019)
    #     for trial in range(args.n_trials):
    #         xs = np.random.uniform(x_bound[:,0], x_bound[:,1], (args.n_init, args.dim)) 
    #         ws_ind = np.random.randint(0, args.n_w, args.n_init).reshape(args.n_init, -1)
    #         np.savez(os.path.join(p_to_init, 'X_init_trial=%d.npz'%(trial+1)), xs)
    #         np.savez(os.path.join(p_to_init, 'W_init_trial=%d.npz'%(trial+1)), ws_ind)
    # else:
    #     print('Loading init points from %s'%(p_to_init))

    
    # for trial in range(n_trials):
    #     xs = np.load(os.path.join(p_to_init, 'X_init_trial=%d.npz'%(trial+1)))['arr_0'] 
    #     ws_ind = np.load(os.path.join(p_to_init, 'W_init_trial=%d.npz'%(trial+1)))['arr_0'] 
    #     ws = kern_w_domain[ws_ind, :].reshape(n_init, -1)
    #     xws = np.hstack((xs, ws))
    #     x_init_list.append(np.copy(xws))


    for trial in range(trial_start, trial_end):
        print('trial = %d'%(trial))
        if args.kernel_type == 'RBF':
            kern = GPy.kern.RBF(hybrid_dim, variance=1., ARD=True)
        elif args.kernel_type == 'Matern32':
            kern = GPy.kern.Matern32(hybrid_dim, variance=1., ARD=True)
        elif args.kernel_type == 'Matern52':
            kern = GPy.kern.Matern52(hybrid_dim, variance=1., ARD=True)
        elif args.kernel_type == 'mt':
            kern = GPy.util.multioutput.ICM(input_dim=decision_dim,num_outputs=n_w,kernel=GPy.kern.RBF(decision_dim, variance=1., ARD=True))
        else:
            raise ValueError(args.kernel_type)
        
        gpmle = GaussianFiniteQuadrature(kernel=kern, 
                        exact_feval=args.exact_feval, 
                        optimize_restarts=100, 
                        verbose=False,
                        decision_dim=decision_dim,
                        w_domain=kern_w_domain)


        path_to_init = './tmp/cnn/init'
        if not os.path.exists(init_data_dir):
            generate_init_x(init_data_dir, n_w, decision_dim, n_trials, seed)

        x_init_p = os.path.join(path_to_init, 'X_init_trial=%d.npz'%(trial+1))
        print('Load x_init from %s'%(x_init_p))
        xx_init = np.load(x_init_p)['arr_0']

        w_init_p = os.path.join(path_to_init, 'W_init_trial=%d.npz'%(trial+1))
        print('Load w_init from %s'%(w_init_p))
        w_init_ind = np.argmax(np.load(w_init_p)['arr_0'], axis=1)

        ws = kern_w_domain[w_init_ind, :].reshape(n_init, -1)
        x_init = np.hstack((xx_init, ws))

        y_init = []
        for x_i in x_init:
            y_i = FUNC.evaluate_mt(x_i.reshape(-1))
            print(x_i, y_i)
            y_init.append(y_i)
        y_init = np.array(y_init).reshape(-1,1) # Unscaled Y 

        trial_y_init_p = os.path.join(path_to_init, 'Y_init_trial=%d'%(trial+1))
        if not os.path.exists(trial_y_init_p + '.npz'):
            np.savez(trial_y_init_p, y_init)

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


        method = 'mt'
        curr_pts, report_points, report_values, curr_vals = \
            mt_selection_routine(gpmle, x_init, y_init, horizon, FUNC, x_bound, xi=xi)

        np.savez(os.path.join(data_dir, 'method.%s_X_trial.%d'%(method, trial+1)), curr_pts)
        np.savez(os.path.join(data_dir, 'method.%s_report_pts_trial.%d'%(method, trial+1)), report_points)
        np.savez(os.path.join(data_dir, 'method.%s_report_values_trial.%d'%(method, trial+1)), report_values)
        np.savez(os.path.join(data_dir, 'method.%s_curr_values_trial.%d'%(method, trial+1)), curr_vals)
