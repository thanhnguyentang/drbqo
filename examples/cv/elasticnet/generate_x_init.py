import numpy as np 
import argparse 


if __name__ == '__main__':
    n_trials = 30 
    n_init = 6 
    decision_dim = 2 
    x_bound = np.array([[0, 1], [1e-8, 1]])
    x_init_list = []
    seed = 2019 
    for trial in range(n_trials):
        xs = np.random.uniform(x_bound[:,0], x_bound[:,1], (n_init, decision_dim)) 
        x_init_list.append(xs)
    x_init_list = np.array(x_init_list)
    print(x_init_list.shape)
    fname = 'x_init_seed.%d_ntrials.%d_dim.%d_ninit.%d'%(seed,n_trials,decision_dim,n_init)
    print('Saving to %s'%(fname))
    np.savez(fname, x_init_list)
