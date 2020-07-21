import numpy as np 
from copy import deepcopy
from optimizer.sgd import Adam
from utils.parallel import run_function_different_arguments_parallel

DEBUG = True
# N_ADAM_STEPS = 100 
# NUM_Z_SAMPLES_FOR_APPROX = 400 

def kg_selection_routine(model, X_init, Y_init, horizon, objective_func, x_bound, **kargv):
    """
    args:
        model: GPyOpt.models.quadrature.GaussianFiniteQuadrature
    """
    print('=== Start KG selection routine ===')
    multi_start = kargv['multi_start']
    lr = kargv['lr']
    batch_size = kargv['batch_size']
    n_adam_steps = kargv['n_adam_steps']
    n_z_samples_for_approx = kargv['n_z_samples_for_approx']

    y_min = np.min(Y_init)
    y_max = np.max(Y_init)

    X = np.copy(X_init)
    Y = (np.copy(Y_init) - y_min) / (y_max - y_min)

    model_copied = deepcopy(model)
    kgmodel = AcquisitionQuadrateKG(model_copied)

    for t in range(horizon):
        print('t=%d'%(t))
        xt = kgmodel.optimize_parallel(None, multi_start, maxit=n_adam_steps, batch_size=batch_size, lr=lr,\
                                       n_samples=n_z_samples_for_approx, use_thread=False)
        print('quadrateKG recommended point: ', xt)
        yt = objective_func.evaluate(xt.reshape(-1)).reshape(-1,1)
        yt = (yt - y_min) / (y_max - y_min)
                
        X = np.vstack((X, xt.reshape(1,-1)))
        Y = np.vstack((Y, yt))
        # kgmodel.gpquadrate.updateModel(X, Y) 
        kgmodel_backup = deepcopy(kgmodel)
        try:
            kgmodel.gpquadrate.updateModel(X, Y)
        except np.linalg.LinAlgError as exc:
            print('np.linalg.LinAlgError happens when fitting GP:')
            print('GP MLE parameters:', kgmodel.gpquadrate.model.param_array)
            print('X_train: ', kgmodel.gpquadrate.model.X)
            print('Y_train: ', kgmodel.gpquadrate.model.Y)
            
            print('Reuse the last GP hyperparameters!')
            kgmodel = deepcopy(kgmodel_backup)
            kgmodel.gpquadrate.model.set_XY(X, Y)
        print('GP MLE parameters:', kgmodel.gpquadrate.model.param_array)
        
        # Save 
        # if fig_dir is not None:
        #     filename = os.path.join(fig_dir, 'acq_plot_%d.png'%(t+1))
        #     plot_data_dict = get_plotting_data_from_quadrature_1d(kgmodel, x_bound, G, optimum, xi)
        #     plot_data_dict.update({'filename': filename}) 
        #     plot_acquisition_1d(**plot_data_dict)
    return X

class AcquisitionQuadrateKG(object):
    def __init__(self, gpquadrate): 
        """
        Args:
            gpquadrature: GPyOpt.models.quadrature.GaussianFiniteQuadrature
        """
        self.gpquadrate = gpquadrate
    
    def optimize_parallel(self, start, multi_start, maxit=100, batch_size=1, lr=0.1, n_samples=400, use_thread=False):
        """
        Args:
            start: np.array. (Could be None)
            multi_start: int, number of Adam restarts.
            maxit: int, number of Adam steps.
            batch_size: int, batch size to estimate stochastic gradient. 
            lr: float, learning rate of Adam.
            n_samples: int, number of z sampels to estimate VOI. 
            use_thread: bool, whether use thread for parallel or use process for parallel.  
            
        """
        results = self.optimize_across_w_parallel_multistart(start, multi_start, maxit, batch_size, lr, use_thread)
        print('quadrateKG_candidate_points: ', results)
        return self.compare_points_parallel(results, start=start, n_samples=n_samples, use_thread=False, debug=False)

    def optimize_across_w_parallel_multistart(self, start, multi_start, maxit=300, batch_size=1, lr = 0.1, use_thread=False):
        if start is not None:
            n_start = 1 + multi_start
            start_points = [start] + \
                [np.random.uniform(self.gpquadrate.x_bound[:,0], self.gpquadrate.x_bound[:,1], (1, self.gpquadrate.decision_dim)) for _ in range(multi_start)]
        else:
            n_start = multi_start 
            start_points = [np.random.uniform(self.gpquadrate.x_bound[:,0], self.gpquadrate.x_bound[:,1], (1, self.gpquadrate.decision_dim)) for _ in range(multi_start)]
            
        arguments_ = []
        for w in self.gpquadrate.w_domain:
            for i in range(n_start):
                arguments_.append(( w.reshape(1,-1), start_points[i]   ))
        arguments =  dict(zip(range(len(arguments_)), arguments_))
        results = run_function_different_arguments_parallel(self._optimize_given_w_start, 
                                                arguments, 
                                                use_thread = use_thread, 
                                                all_success=True,
                                                **{
                                                   'maxit': maxit,
                                                   'batch_size': batch_size,
                                                   'lr': lr,
                                                   'use_thread_inner_opt': not use_thread})
        return results

    def _optimize_given_w_start(self, w_start, maxit=300, batch_size=1, lr=0.1, use_thread_inner_opt=True):
        w = w_start[0]
        start = w_start[1]
        return self._optimize_given_w(w, start, maxit, batch_size, lr, use_thread_inner_opt)
    
    def _optimize_given_w(self, w, start, maxit=300, batch_size=1, lr=0.1, use_thread_inner_opt=True ):
        optimizer = Adam(
                grad_func=self.sto_grad_func, 
                lr = lr, 
                bound=self.gpquadrate.x_bound,
                **{'w': w.reshape(1,-1), 'batch_size': batch_size, 'use_thread': use_thread_inner_opt})
        point = optimizer.optimize(start, maxit)
        return point 
    
    def optimize_across_w(self, start, maxit=300, batch_size=1, use_thread=True):
        points = []
        for w in self.gpquadrate.w_domain:
            optimizer = Adam(
                grad_func=self.sto_grad_func, 
                lr = 0.1, 
                bound=self.gpquadrate.x_bound,
                **{'w': w.reshape(1,-1), 'batch_size': batch_size, 'use_thread': use_thread})
            point = optimizer.optimize(start, maxit)
            points.append(point)
        return points 
    
    def sto_grad_func(self, cand, w, batch_size=1, use_thread=True):
        """
        Args:
            cand: x.
        """
        cand = cand.reshape(1,-1)
        xw = np.hstack((cand, w))
        if batch_size == 1:
            parallel = False 
        g = self.gpquadrate.grad_approximate_voi(xw, n_samples=batch_size, use_thread=use_thread)
        return -g 
    
    def compare_points_parallel(self, results, start=None, n_samples=100, use_thread=False, debug=False):

        n_w = self.gpquadrate.w_domain.shape[0]
        n_start = int(len(results) / n_w)

        arguments_ = []
        for i in range(n_w):
            for j in range(n_start):
                arguments_.append(
                        ( 
                            np.hstack(
                                    ( 
                                        # BUG: results[i*n_w + j].reshape(1,-1), \
                                        results[i*n_start + j].reshape(1,-1), \
                                        self.gpquadrate.w_domain[i,:].reshape(1,-1) 
                                    )
                                )   
                        )
                    )
        
        arguments = dict(zip(range(len(arguments_)), arguments_ ))
        # for i in range(len(results)):
        #     arguments[i] = np.hstack(( results[i].reshape(1,-1), self.gpquadrate.w_domain[i,:].reshape(1,-1) ))
        
        vois = run_function_different_arguments_parallel( self.gpquadrate.approximate_voi, 
                                                arguments, 
                                                use_thread = use_thread, 
                                                all_success=True,
                                                **{'n_samples': n_samples}
        )
        
        print('candidate_quadrateKG_values: ', vois)
        ind = max(vois, key=vois.get)
        next_point = arguments_[ind]
        # np.hstack( (results[ind].reshape(1,-1), self.gpquadrate.w_domain[ind:ind+1, :]) ) 
        if debug:
            if start is not None:
                print('CHECK KG') # Check if KG(next_point) > KG(start)
                arguments_2 = {}
                for i in range(len(results)):
                    arguments_2[i] = np.hstack(( start.reshape(1,-1), self.gpquadrate.w_domain[i,:].reshape(1,-1) ))
            
                vois_2 = run_function_different_arguments_parallel( self.gpquadrate.approximate_voi, 
                                                    arguments_2, 
                                                    use_thread = use_thread, 
                                                    all_success=True,
                                                    **{'n_samples': n_samples}
                )
                print('start_quadrateKG_values: ', vois_2)
        
        return next_point
    