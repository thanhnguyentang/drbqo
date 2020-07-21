
import numpy as np
import GPy
from copy import deepcopy
import GPyOpt 
from GPyOpt.models.base import BOModel
from scipy.linalg import cholesky, cho_solve, solve_triangular
import sys
from optimizer import minimize 

from utils.parallel import run_function_different_arguments_parallel
from utils.compute_dr_weight import min_weighted_sum_with_weights_on_chi_squared_ball
SMALL_POSITIVE_EPSILON = 1e-30

class GaussianQuadrature(BOModel):
    def __init__(self):
        pass 
    
    
class GaussianFiniteQuadrature(GaussianQuadrature):
    """
    General class for handling a Gaussian Process in GPyOpt.

    :param kernel: GPy kernel to use in the GP model.
    :param noise_var: value of the noise variance if known.
    :param exact_feval: whether noiseless evaluations are available. IMPORTANT to make the optimization work well in noiseless scenarios (default, False).
    :param optimizer: optimizer of the model. Check GPy for details.
    :param max_iters: maximum number of iterations used to optimize the parameters of the model.
    :param optimize_restarts: number of restarts in the optimization.
    :param sparse: whether to use a sparse GP (default, False). This is useful when many observations are available.
    :param num_inducing: number of inducing points if a sparse GP is used.
    :param verbose: print out the model messages (default, False).
    :param ARD: whether ARD is used in the kernel (default, False).

    .. Note:: This model does Maximum likelihood estimation of the hyper-parameters.

    """


    analytical_gradient_prediction = True  # --- Needed in all models to check is the gradients of acquisitions are computable.

    def __init__(self, kernel=None, 
                 decision_dim=None, 
                 w_quadrate=None, 
                 w_domain = None, 
                 noise_var=None, 
                 exact_feval=False, 
                 optimizer='bfgs', 
                 max_iters=1000, 
                 optimize_restarts=5, 
                 sparse = False, 
                 num_inducing = 10,  
                 verbose=True, 
                 x_bound = None,
                 ARD=False):
        self.kernel = kernel
        self.noise_var = noise_var
        self.exact_feval = exact_feval
        self.optimize_restarts = optimize_restarts
        self.optimizer = optimizer
        self.max_iters = max_iters
        self.verbose = verbose
        self.sparse = sparse
        self.num_inducing = num_inducing
        self.model = None
        self.ARD = ARD
        self.decision_dim = decision_dim
        self.w_quadrate = FiniteQuadrate(w_domain) if w_quadrate is None else w_quadrate# an instance of `Quadrate`
        self.w_domain = w_domain 
        if x_bound is None:
            x_bound = np.array([[0,1]]*decision_dim)
        self.x_bound = x_bound

    @staticmethod
    def fromConfig(config):
        return GaussianFiniteQuadrature(**config)

    def _create_model(self, X, Y):
        """
        Creates the model given some input data X and Y.
        """

        # --- define kernel
        self.input_dim = X.shape[1]
        if self.kernel is None:
            kern = GPy.kern.Matern52(self.input_dim, variance=1., ARD=self.ARD) #+ GPy.kern.Bias(self.input_dim)
        else:
            kern = self.kernel
            self.kernel = None

        # --- define model
        noise_var = Y.var()*0.01 if self.noise_var is None else self.noise_var

        if not self.sparse:
            self.model = GPy.models.GPRegression(X, Y, kernel=kern, noise_var=noise_var)
        else:
            self.model = GPy.models.SparseGPRegression(X, Y, kernel=kern, num_inducing=self.num_inducing)

        # --- restrict variance if exact evaluations of the objective
        if self.exact_feval:
            self.model.Gaussian_noise.constrain_fixed(1e-9, warning=False)
        else:
            # --- We make sure we do not get ridiculously small residual noise variance
            self.model.Gaussian_noise.constrain_bounded(1e-9, 1e6, warning=False) #constrain_positive(warning=False)
    
    def _get_woodbury_chol(self):
        return self.model.posterior.woodbury_chol

    def _get_woodbury_inv(self):
        return self.model.posterior.woodbury_inv
    
    def _get_woodbury_vector(self):
        return self.model.posterior.woodbury_vector
    
    def _get_variance(self):
        return self.model.kern.variance 
    
    def _get_lengthscale(self):
        return self.model.kern.lengthscale

    def _get_Gaussian_noise_variance(self):
        return self.model.likelihood.variance
    
    def compute_cross_cov_prior(self, X1, X2=None):
        return self.model.kern.K(X1, X2) 
    
    def compute_grad_cross_cov_prior(self, X1, X2):
        r = self.model.kern._scaled_dist(X1, X2)
        dKdr = self.model.kern.dK_dr(r)
        ls = self.model.kern.lengthscale 
        drdx = ( (X1[:, np.newaxis, :] - X2[np.newaxis,:,:]) / ls**2 ) / np.clip(r[:,:, np.newaxis], SMALL_POSITIVE_EPSILON, None)
        return dKdr[:,:, np.newaxis] * drdx 

    def compute_a_n_b_n(self, x, candidate, return_grad_wrt_x=False, return_grad_wrt_cand=False):
        """Compute Value of Information. 
        
        :param gp: a fitted GP (either use MLE or slice sampling). 
        :param x: (1 x d). 
        :param candidate: (1 x d1) a candidate point, candidate = (x,w). 
        :return B: (1xnxn_w)
        """  
        # Compute a_n(x)
        if x.ndim == 1:
            x = x.reshape(1,-1)
        
        xw = np.hstack( 
                    (
                        np.kron(x, np.ones((self.w_domain.shape[0], 1))), 
                        np.kron(np.ones((x.shape[0], 1)), self.w_domain) 
                        )
            ) 
        woodbury_chol = self._get_woodbury_chol()
        woodbury_vector = self._get_woodbury_vector() # (n x 1)

        B = self.compute_cross_cov_prior(xw, self.model._predictive_variable) # (kxn)
        B = self.w_quadrate.integrate(B, axis=0) # (n,)
        if return_grad_wrt_x:
            dB = self.compute_grad_cross_cov_prior(xw, self.model._predictive_variable)[:,:,:self.decision_dim] # (k x n x d)
            dB = self.w_quadrate.integrate(dB, axis=0).reshape(-1, self.decision_dim) # (n x d)
            grad_a_n_wrt_x = np.dot(woodbury_vector.T, dB) # (1 x d)
        a_n = np.dot(B.reshape(1,-1), woodbury_vector) # (1 x 1)
        
        # b_n 
        
        lamb = self._get_Gaussian_noise_variance()
        if lamb < SMALL_POSITIVE_EPSILON:
            candidate_is_new = True
            for i in range(self.model._predictive_variable.shape[0]):
                if np.array_equal(self.model._predictive_variable[i:i+1,:], candidate):
                    candidate_is_new = False 
                    break 
            if not candidate_is_new:
                b_n = 0 
                grad_b_n_wrt_x = 0 
                grad_b_n_wrt_cand = 0 
                if return_grad_wrt_x and return_grad_wrt_cand:
                    return a_n, b_n, grad_a_n_wrt_x, grad_b_n_wrt_x, grad_b_n_wrt_cand
                elif return_grad_wrt_x:
                    return a_n, b_n, grad_a_n_wrt_x, grad_b_n_wrt_x
                elif return_grad_wrt_cand:
                    return a_n, b_n, grad_b_n_wrt_cand
                else:
                    return a_n, b_n
                    
        b_new = self.compute_cross_cov_prior(xw, candidate) # (n x 1)
        b_new = self.w_quadrate.integrate(b_new, axis=0) # (1,)
        
        gamma = self.compute_cross_cov_prior(candidate, self.model._predictive_variable) # (1 x n) 
        
        A_inv_gamma = cho_solve((woodbury_chol, True), gamma.T) # (n x 1)
        denominator = np.clip(self.compute_cross_cov_prior(candidate) - np.dot(gamma, A_inv_gamma) + lamb, SMALL_POSITIVE_EPSILON, None) #(1,1)
        sqrt_denominator = np.sqrt(denominator)
        numerator = b_new - np.dot(B.reshape(1,-1), A_inv_gamma) #(1,1)
        b_n = numerator / sqrt_denominator
        
        if return_grad_wrt_x:
            grad_b_new_wrt_x = self.compute_grad_cross_cov_prior(xw, candidate)[:,:,:self.decision_dim] # (n x 1 x d)
            grad_b_new_wrt_x = self.w_quadrate.integrate(grad_b_new_wrt_x, axis=0) # (1 x d)
            grad_b_n_wrt_x = (grad_b_new_wrt_x -  np.dot(A_inv_gamma.T, dB)) / sqrt_denominator
            
        # Compute grad wrt candidate 
        if return_grad_wrt_cand:
            grad_gramma_wrt_cand = self.compute_grad_cross_cov_prior(candidate, self.model._predictive_variable)[0,:,:self.decision_dim] # (n x d)
            
            grad_b_new_wrt_cand = self.compute_grad_cross_cov_prior(candidate, xw)[:,:,:self.decision_dim] # (1 x n x d)
            grad_b_new_wrt_cand = self.w_quadrate.integrate(grad_b_new_wrt_cand.reshape(-1, self.decision_dim), axis=0) # (d,) 
            
            A_inv_grad_gramma = cho_solve((woodbury_chol, True), grad_gramma_wrt_cand) #(n x d)
            grad_numerator_wrt_cand = grad_b_new_wrt_cand - np.dot(B, A_inv_grad_gramma) #(1 x d)
            
            # Compute d (gamma^T A_n^{-1} gamma) wrt cand: 
            A_inv = self._get_woodbury_inv()
            # A_inv = cho_solve((_L, True), np.eye(_L.shape[0])) 
            grad_gamma_trans_A_inv_gamma_wrt_cand = np.dot(gamma, A_inv + A_inv.T).dot(grad_gramma_wrt_cand) # (1 x d)
            
            grad_b_n_wrt_cand = (2 * grad_numerator_wrt_cand * denominator + numerator * grad_gamma_trans_A_inv_gamma_wrt_cand) /\
                (denominator * sqrt_denominator)
        
        if return_grad_wrt_x and return_grad_wrt_cand:
            return a_n, b_n, grad_a_n_wrt_x, grad_b_n_wrt_x, grad_b_n_wrt_cand
        elif return_grad_wrt_x:
            return a_n, b_n, grad_a_n_wrt_x, grad_b_n_wrt_x
        elif return_grad_wrt_cand:
            return a_n, b_n, grad_b_n_wrt_cand
        else:
            return a_n, b_n
        
    def compute_grad_b_n_wrt_cand(self, x, candidate):
        # Compute a_n(x)
        if x.ndim == 1:
            x = x.reshape(1,-1)
        
        xw = np.hstack( 
                    (
                        np.kron(x, np.ones((self.w_domain.shape[0], 1))), 
                        np.kron(np.ones((x.shape[0], 1)), self.w_domain) 
                        )
            ) 
        woodbury_chol = self._get_woodbury_chol()
        woodbury_vector = self._get_woodbury_vector() # (n x 1)

        B = self.compute_cross_cov_prior(xw, self.model._predictive_variable) # (kxn)
        B = self.w_quadrate.integrate(B, axis=0) # (n,)

        # b_n 
        
        lamb = self._get_Gaussian_noise_variance()
        if lamb < SMALL_POSITIVE_EPSILON:
            candidate_is_new = True
            for i in range(self.model._predictive_variable.shape[0]):
                if np.array_equal(self.model._predictive_variable[i:i+1,:], candidate):
                    candidate_is_new = False 
                    break 
            if not candidate_is_new:
                grad_b_n_wrt_cand = 0 
                return grad_b_n_wrt_cand
            
        b_new = self.compute_cross_cov_prior(xw, candidate) # (n x 1)
        b_new = self.w_quadrate.integrate(b_new, axis=0) # (1,)
        
        gamma = self.compute_cross_cov_prior(candidate, self.model._predictive_variable) # (1 x n) 
        
        A_inv_gamma = cho_solve((woodbury_chol, True), gamma.T) # (n x 1)
        denominator = np.clip(self.compute_cross_cov_prior(candidate) - np.dot(gamma, A_inv_gamma) + lamb, SMALL_POSITIVE_EPSILON, None) #(1,1)
        sqrt_denominator = np.sqrt(denominator)
        numerator = b_new - np.dot(B.reshape(1,-1), A_inv_gamma) #(1,1)
        
        # Compute grad wrt candidate 
        grad_gramma_wrt_cand = self.compute_grad_cross_cov_prior(candidate, self.model._predictive_variable)[0,:,:self.decision_dim] # (n x d)
        
        grad_b_new_wrt_cand = self.compute_grad_cross_cov_prior(candidate, xw)[:,:,:self.decision_dim] # (1 x n x d)
        grad_b_new_wrt_cand = self.w_quadrate.integrate(grad_b_new_wrt_cand.reshape(-1, self.decision_dim), axis=0) # (d,) 
        
        A_inv_grad_gramma = cho_solve((woodbury_chol, True), grad_gramma_wrt_cand) #(n x d)
        grad_numerator_wrt_cand = grad_b_new_wrt_cand - np.dot(B, A_inv_grad_gramma) #(1 x d)
        
        # Compute d (gamma^T A_n^{-1} gamma) wrt cand: 
        A_inv = self._get_woodbury_inv()
        # A_inv = cho_solve((_L, True), np.eye(_L.shape[0])) 
        grad_gamma_trans_A_inv_gamma_wrt_cand = np.dot(gamma, A_inv + A_inv.T).dot(grad_gramma_wrt_cand) # (1 x d)
        
        grad_b_n_wrt_cand = (2 * grad_numerator_wrt_cand * denominator + numerator * grad_gamma_trans_A_inv_gamma_wrt_cand) /\
            (denominator * sqrt_denominator)
        return grad_b_n_wrt_cand

    def compute_a_n_b_n_many_candidates(self, x, candidates, return_grad_wrt_x=False, return_grad_wrt_cand=False):
        """Compute Value of Information. 
        Args:
            x: (1 x d). 
            candidates: (t x d1) a candidate point, candidate = (x, w). 
            self.model._predictive_variable: (m x d1)
            w_domain: (n x 1)

        Returns:
            a_n: (1,)
            b_n: (t,)
            grad_a_n_wrt_x: (1 x d)
            grad_b_n_wrt_x: (t x d)
            grad_b_n_wrt_cand: (t x d)

        """  
        # Compute a_n(x)
        if x.ndim == 1:
            x = x.reshape(1,-1)
        
        xw = np.hstack( 
                    (
                        np.kron(x, np.ones((self.w_domain.shape[0], 1))), 
                        np.kron(np.ones((x.shape[0], 1)), self.w_domain) 
                        )
            ) 
        woodbury_chol = self._get_woodbury_chol()
        woodbury_vector = self._get_woodbury_vector() # (m x 1)

        B = self.compute_cross_cov_prior(xw, self.model._predictive_variable) # (n x m)
        B = self.w_quadrate.integrate(B, axis=0) # (m,)
        if return_grad_wrt_x:
            dB = self.compute_grad_cross_cov_prior(xw, self.model._predictive_variable)[:,:,:self.decision_dim] # (n x m x d)
            dB = self.w_quadrate.integrate(dB, axis=0).reshape(-1, self.decision_dim) # (m x d)
            grad_a_n_wrt_x = np.dot(woodbury_vector.T, dB) # (1 x d)
        a_n = np.dot(B.reshape(1,-1), woodbury_vector) # (1 x 1)
        
        # b_n 
        mask = np.ones((candidates.shape[0]))
        
        lamb = self._get_Gaussian_noise_variance()
        if lamb < SMALL_POSITIVE_EPSILON:
            mask = self.model.kern._unscaled_dist(candidates, self.model._predictive_variable) 
            mask = 1 - np.array(np.axis(np.array(mask < SMALL_POSITIVE_EPSILON).astype('float'), axis=1) > 0).astype('float')
 
                    
        b_new = self.compute_cross_cov_prior(xw, candidates) # (n x t)
        b_new = self.w_quadrate.integrate(b_new, axis=0) # (t,)
        
        gamma = self.compute_cross_cov_prior(candidates, self.model._predictive_variable) # (t x m) 
        
        A_inv_gamma = cho_solve((woodbury_chol, True), gamma.T) # (m x t)
        denominator = np.clip(self.model.kern.Kdiag(candidates) - np.sum(gamma * A_inv_gamma.T, axis=1) + lamb, SMALL_POSITIVE_EPSILON, None) #(t,1)
        denominator = denominator.reshape(-1,1)
        sqrt_denominator = np.sqrt(denominator)
        # numerator_right_term = np.dot(B.reshape(1,-1), A_inv_gamma) #(1 x t)
        # numerator = (b_new -  np.dot(A_inv_gamma.T, B.reshape(-1,1))).reshape(-1,1) #(t,1)
        numerator = (b_new - np.dot(B.reshape(1,-1), A_inv_gamma)).T
        b_n = numerator / sqrt_denominator # (t,1)
        
        if return_grad_wrt_x:
            grad_b_new_wrt_x = self.compute_grad_cross_cov_prior(xw, candidates)[:,:,:self.decision_dim] # (n x t x d)
            grad_b_new_wrt_x = self.w_quadrate.integrate(grad_b_new_wrt_x, axis=0) # (t x d)
            grad_b_n_wrt_x = (grad_b_new_wrt_x -  np.dot(A_inv_gamma.T, dB)) / sqrt_denominator.reshape(-1,1) # (t x d)

            grad_b_n_wrt_x = mask.reshape(-1,1) * grad_b_n_wrt_x
            
        # Compute grad wrt candidate 
        if return_grad_wrt_cand:
            grad_gramma_wrt_cand = self.compute_grad_cross_cov_prior(candidates, self.model._predictive_variable)[:,:,:self.decision_dim] # (t x m x d)
            
            grad_b_new_wrt_cand = self.compute_grad_cross_cov_prior(candidates, xw)[:,:,:self.decision_dim] # (t x n x d)
            grad_b_new_wrt_cand = self.w_quadrate.integrate(grad_b_new_wrt_cand, axis=1) # (t x d) 
                      
            # Compute d (gamma^T A_n^{-1} gamma) wrt cand: 
            A_inv = self._get_woodbury_inv() # (m x m)
            
            # A_inv_grad_gramma = cho_solve((woodbury_chol, True), grad_gramma_wrt_cand) #(n x d)
            A_inv_grad_gamma = A_inv.dot(grad_gramma_wrt_cand) # (m x t x d) 
            grad_numerator_wrt_cand = grad_b_new_wrt_cand - np.sum(B.reshape(-1,1,1)*A_inv_grad_gamma, axis=0) #(t x d)
            # A_inv = cho_solve((_L, True), np.eye(_L.shape[0])) 
            grad_gamma_trans_A_inv_gamma_wrt_cand = np.sum(np.dot(gamma, A_inv + A_inv.T)[:,:, np.newaxis] * grad_gramma_wrt_cand, axis=1) # (t x d)
            
            grad_b_n_wrt_cand = (2 * grad_numerator_wrt_cand * denominator + numerator * grad_gamma_trans_A_inv_gamma_wrt_cand) /\
                (denominator * sqrt_denominator)

            grad_b_n_wrt_cand = mask.reshape(-1,1) *grad_b_n_wrt_cand

        a_n = a_n 
        b_n = mask.reshape(-1,1) *b_n 

        
        if return_grad_wrt_x and return_grad_wrt_cand:
            return a_n, b_n, grad_a_n_wrt_x, grad_b_n_wrt_x, grad_b_n_wrt_cand
        elif return_grad_wrt_x:
            return a_n, b_n, grad_a_n_wrt_x, grad_b_n_wrt_x
        elif return_grad_wrt_cand:
            return a_n, b_n, grad_b_n_wrt_cand
        else:
            return a_n, b_n    
    def voi_given_z(self, z, candidate):
        def inner_voi(x):
            x = x.reshape(1,-1)
            a_n, b_n, grad_a_n_wrt_x, grad_b_n_wrt_x = self.compute_a_n_b_n(x, candidate, return_grad_wrt_x=True)
            val = a_n + z*b_n  
            grad = grad_a_n_wrt_x + z*grad_b_n_wrt_x
            return -val, -grad 
        x_min, _ = minimize(inner_voi, 
                    self.x_bound, 
                    approx_grad=0, 
                    maxiter=15000, 
                    n_warmup=0, 
                    method='lbfgs', 
                    n_restarts_optimizer=100, 
                    initializer=None, 
                    x_init=None,
                    random_state=None)
        a_n, b_n = self.compute_a_n_b_n(x_min.reshape(1,-1), candidate)
        return a_n + z*b_n
    
    def grad_voi_given_z(self, z, candidate):
        def inner_voi(x):
            x = x.reshape(1,-1)
            a_n, b_n, grad_a_n_wrt_x, grad_b_n_wrt_x = self.compute_a_n_b_n(x, candidate, return_grad_wrt_x=True)
            val = a_n + z*b_n  
            grad = grad_a_n_wrt_x + z*grad_b_n_wrt_x
            return -val, -grad 
        x_min, _ = minimize(inner_voi, 
                    self.x_bound, 
                    approx_grad=0, 
                    maxiter=15000, 
                    n_warmup=0, 
                    method='lbfgs', 
                    n_restarts_optimizer=100, 
                    initializer=None, 
                    x_init=None,
                    random_state=None)
        try:
            grad_b_n_wrt_cand = self.compute_grad_b_n_wrt_cand(x_min.reshape(1,-1), candidate)
        except AttributeError as exc:
            print(x_min)
            x_range = np.linspace(self.x_bound[:,0], self.x_bound[:,1], 100)
            for x in x_range:
                x = x.reshape(1,-1)
                print(self.compute_a_n_b_n(x, candidate, return_grad_wrt_x=True))
            raise exc 
        return z*grad_b_n_wrt_cand
    
    def approximate_voi(self, candidate, n_samples = 30, use_thread=True):
        """
        Args:
            candidate: xw 
        """
        sys.stdout.write('Computing voi...\r')
        sys.stdout.flush()
        
        _, max_mean = self.optimize_posterior_mean()
        zs = np.random.randn(n_samples)
        arguments = dict(zip(range(n_samples), zs)) 
    
        kwargs = {'candidate': candidate} 
        results = run_function_different_arguments_parallel(self.voi_given_z, arguments, use_thread=use_thread, **kwargs)
        v = 0 
        for k, a in results.items():
            v += a
        return v/n_samples - max_mean
    
    def grad_approximate_voi(self, candidate, n_samples = 30, use_thread=True):
        """
        Args:
            candidate: xw 
        """
        zs = np.random.randn(n_samples)
        arguments = dict(zip(range(n_samples), zs)) 
    
        kwargs = {'candidate': candidate} 
        results = run_function_different_arguments_parallel(self.grad_voi_given_z, arguments, use_thread=use_thread, **kwargs)
        g = 0
        for k, a in results.items():
            g += a
        return g/n_samples
    
    def approximate_voi_accross_w(self, candidate, n_samples = 30):
        """
        Args:
            candidate: x 
        
        Return:
            voi for different xw where w in w_domain. 
        """
        xw = np.hstack( 
                    (
                        np.kron(candidate, np.ones((self.w_domain.shape[0], 1))), 
                        np.kron(np.ones((candidate.shape[0], 1)), self.w_domain) 
                        )
            ) 
        vs = []
        gs = []
        for cand in xw:
            v,g = self.approximate_voi(cand.reshape(1,-1), n_samples = n_samples)
            vs.append(v)
            gs.append(g)
        
        return vs, gs
        
        # arguments = dict(zip(range(self.w_domain.shape[0]), xw))  
        # results = run_function_different_arguments_parallel(self.approximate_voi, arguments, all_success=True, **{'n_samples': n_samples})
        # return results 
    
    def updateModel(self, X_all, Y_all, X_new=None, Y_new=None):
        """
        Updates the model with new observations.
        """
        if self.model is None:
            self._create_model(X_all, Y_all)
        else:
            self.model.set_XY(X_all, Y_all)

        # WARNING: Even if self.max_iters=0, the hyperparameters are bit modified...
        if self.max_iters > 0:
            # --- update the model maximizing the marginal likelihood.
            if self.optimize_restarts==1:
                self.model.optimize(optimizer=self.optimizer, max_iters = self.max_iters, messages=False, ipython_notebook=False)
            else:
                self.model.optimize_restarts(num_restarts=self.optimize_restarts, optimizer=self.optimizer, max_iters = self.max_iters, verbose=self.verbose)

    def _predict(self, X, include_likelihood=True):
        if X.ndim==1: X = X[None,:]
        n_x = X.shape[0]
        n_w = self.w_domain.shape[0]
        XW = np.hstack((np.kron(X, np.ones((n_w, 1))), np.kron(np.ones((n_x, 1)), self.w_domain)))  
        m, cov = self.model.predict(XW, full_cov=True, include_likelihood=include_likelihood)
        m = m.reshape(n_x, n_w)
        cov_new = np.zeros((n_x, n_w, n_w))
        for i in range(n_x):
            cov_new[i,:,:] = cov[i*n_w:(i+1)*n_w, i*n_w:(i+1)*n_w]
        m = self.w_quadrate.integrate(m)
        v = self.w_quadrate.integrate(cov_new, double=True)
        
        # In case numerical error 
        v = np.clip(v,0, None)
        return m, v

    def predict(self, X, with_noise=True):
        """
        Predictions with the model. Returns posterior means and standard deviations at X. Note that this is different in GPy where the variances are given.

        Parameters:
            X (np.ndarray) - points to run the prediction for.
            with_noise (bool) - whether to add noise to the prediction. Default is True.
        """
        m, v = self._predict(X, with_noise)
        # We can take the square root because v is just a diagonal matrix of variances
        return m, np.sqrt(v)

    def predict_covariance(self, X, with_noise=True):
        """
        Predicts the covariance matric for points in X.

        Parameters:
            X (np.ndarray) - points to run the prediction for.
            with_noise (bool) - whether to add noise to the prediction. Default is True.
        """
        _, v = self._predict(X, with_noise)
        return v

    def predict_mean(self, X, with_noise=True):
        m, _ = self._predict(X, with_noise)
        return m
    
    def predict_robust_mean(self, X, with_noise=True, rho=0):
        """
        Rho-robust quadrature for predictive mean
        """
        if X.ndim==1: X = X[None,:]
        n_x = X.shape[0]
        n_w = self.w_domain.shape[0]
        XW = np.hstack((np.kron(X, np.ones((n_w, 1))), np.kron(np.ones((n_x, 1)), self.w_domain)))  
        m, cov = self.model.predict(XW, full_cov=True, include_likelihood=with_noise)
        m = m.reshape(n_x, n_w)
        robust_m = np.zeros(n_x) 
        for i in range(n_x):
            m_ = m[i,:] 
            _, rm_ = min_weighted_sum_with_weights_on_chi_squared_ball(m_, rho=rho)
            robust_m[i] = rm_
        return robust_m
    
    def get_q_robust_report(self, rho): 
        vals = self.predict_robust_mean(self.model.X[:,:self.decision_dim], rho=rho)
        ind = np.argmax(vals) 
        return self.model.X[ind,:self.decision_dim], vals[ind] 
    
    def get_q_report(self): 
        vals = self.predict_mean(self.model.X[:, :self.decision_dim])
        ind = np.argmax(vals) 
        return self.model.X[ind,:self.decision_dim], vals[ind] 
            
    def optimize_posterior_mean(self):
        def obj_func(x):
            x = x.reshape(1,-1)
            m = self.predict_mean(x)
            return -m 
        x_min, neg_max_mean = minimize(obj_func, 
                    self.x_bound, 
                    approx_grad=1, 
                    maxiter=15000, 
                    n_warmup=0, 
                    method='lbfgs', 
                    n_restarts_optimizer=100, 
                    initializer=None, 
                    x_init=None,
                    random_state=None)
        return x_min, -neg_max_mean

    def get_fmin(self):
        """
        Returns the location where the posterior mean is takes its minimal value.
        """
        return self.predict_mean(self.model.X[:,:self.decision_dim])[0].min()
    def get_fmax(self):
        """
        Returns the location where the posterior mean is takes its minimal value.
        """
        return self.predict_mean(self.model.X[:, :self.decision_dim])[0].max()
    
    def predict_withGradients(self, X):
        """
        Returns the mean, standard deviation, mean gradient and standard deviation gradient at X.
        """
        raise NotImplementedError('')

    def copy(self):
        """
        Makes a safe copy of the model.
        """
        copied_model = GaussianFiniteQuadrature(kernel = self.model.kern.copy(),
                            noise_var=self.noise_var,
                            exact_feval=self.exact_feval,
                            optimizer=self.optimizer,
                            max_iters=self.max_iters,
                            optimize_restarts=self.optimize_restarts,
                            verbose=self.verbose,
                            ARD=self.ARD)

        copied_model._create_model(self.model.X,self.model.Y)
        copied_model.updateModel(self.model.X,self.model.Y, None, None)
        return copied_model

    def get_model_parameters(self):
        """
        Returns a 2D numpy array with the parameters of the model
        """
        return np.atleast_2d(self.model[:])

    def get_model_parameters_names(self):
        """
        Returns a list with the names of the parameters of the model
        """
        return self.model.parameter_names_flat().tolist()

    def get_covariance_between_points(self, x1, x2):
        """
        Given the current posterior, computes the covariance between two sets of points.
        """
        return self.model.posterior_covariance_between_points(x1, x2)


class GaussianFiniteQuadrature_MCMC(BOModel):
    """
    General class for handling a Gaussian Process in GPyOpt.

    :param kernel: GPy kernel to use in the GP model.
    :param noise_var: value of the noise variance if known.
    :param exact_feval: whether noiseless evaluations are available. IMPORTANT to make the optimization work well in noiseless scenarios (default, False).
    :param n_samples: number of MCMC samples.
    :param n_burnin: number of samples not used.
    :param subsample_interval: sub-sample interval in the MCMC.
    :param step_size: step-size in the MCMC.
    :param leapfrog_steps: ??
    :param verbose: print out the model messages (default, False).

    .. Note:: This model does MCMC over the hyperparameters.

    """

    MCMC_sampler = True
    analytical_gradient_prediction = True # --- Needed in all models to check is the gradients of acquisitions are computable.

    def __init__(self, 
                kernel=None, 
                decision_dim=None, 
                w_quadrate=None, 
                w_domain = None, 
                noise_var=None, 
                exact_feval=False, 
                n_samples = 10, 
                n_burnin = 100, 
                subsample_interval = 10, 
                step_size = 1e-1, 
                leapfrog_steps=20, 
                verbose=False):
        self.kernel = kernel
        self.noise_var = noise_var
        self.exact_feval = exact_feval
        self.verbose = verbose
        self.n_samples = n_samples
        self.subsample_interval = subsample_interval
        self.n_burnin = n_burnin
        self.step_size = step_size
        self.leapfrog_steps = leapfrog_steps
        self.model = None
        self.decision_dim = decision_dim
        self.w_quadrate = FiniteQuadrate(w_domain) if w_quadrate is None else w_quadrate# an instance of `Quadrate`
        self.w_domain = w_domain 

    def _create_model(self, X, Y):
        """
        Creates the model given some input data X and Y.
        """

        # --- define kernel
        self.input_dim = X.shape[1]
        if self.kernel is None:
            kern = GPy.kern.RBF(self.input_dim, variance=1.)
        else:
            kern = self.kernel
            self.kernel = None

        # --- define model
        noise_var = Y.var()*0.01 if self.noise_var is None else self.noise_var
        self.model = GPy.models.GPRegression(X, Y, kernel=kern, noise_var=noise_var)

        # --- Define prior on the hyper-parameters for the kernel (for integrated acquisitions)
        self.model.kern.set_prior(GPy.priors.Gamma.from_EV(2.,4.))
        self.model.likelihood.variance.set_prior(GPy.priors.Gamma.from_EV(2.,4.))

        # --- Restrict variance if exact evaluations of the objective
        if self.exact_feval:
            self.model.Gaussian_noise.constrain_fixed(1e-6, warning=False)
        else:
            self.model.Gaussian_noise.constrain_positive(warning=False)

    def updateModel(self, X_all, Y_all, X_new=None, Y_new=None):
        """
        Updates the model with new observations.
        """

        if self.model is None:
            self._create_model(X_all, Y_all)
        else:
            self.model.set_XY(X_all, Y_all)

        # update the model generating hmc samples
        self.model.optimize(max_iters = 200)
        self.model.param_array[:] = self.model.param_array * (1.+np.random.randn(self.model.param_array.size)*0.01)
        self.hmc = GPy.inference.mcmc.HMC(self.model, stepsize=self.step_size)
        ss = self.hmc.sample(num_samples=self.n_burnin + self.n_samples* self.subsample_interval, hmc_iters=self.leapfrog_steps)
        self.hmc_samples = ss[self.n_burnin::self.subsample_interval]
    
    def _predict(self, X, include_likelihood=True):
        if X.ndim==1: X = X[None,:]
        n_x = X.shape[0]
        n_w = self.w_domain.shape[0]
        XW = np.hstack((np.kron(X, np.ones((n_w, 1))), np.kron(np.ones((n_x, 1)), self.w_domain)))  
        m, cov = self.model.predict(XW, full_cov=True, include_likelihood=include_likelihood)
        m = m.reshape(n_x, n_w)
        cov_new = np.zeros((n_x, n_w, n_w))
        for i in range(n_x):
            cov_new[i,:,:] = cov[i*n_w:(i+1)*n_w, i*n_w:(i+1)*n_w]
        m = self.w_quadrate.integrate(m)
        v = self.w_quadrate.integrate(cov_new, double=True)
        return m, v

    def predict(self, X):
        """
        Predictions with the model for all the MCMC samples. Returns posterior means and standard deviations at X. Note that this is different in GPy where the variances are given.
        """ 
        ps = self.model.param_array.copy()
        means = []
        stds = []
        for s in self.hmc_samples:
            if self.model._fixes_ is None:
                self.model[:] = s
            else:
                self.model[self.model._fixes_] = s
            self.model._trigger_params_changed()
            m,v = self._predict(X)
            means.append(m)
            stds.append(np.sqrt(np.clip(v, SMALL_POSITIVE_EPSILON, np.inf)))
        self.model.param_array[:] = ps
        self.model._trigger_params_changed()
        return means, stds
        
    def predict_mean(self, X, with_noise=True):
        m, _ = self._predict(X, with_noise)
        return m

    def get_fmin(self):
        """
        Returns the location where the posterior mean is takes its minimal value.
        """
        ps = self.model.param_array.copy()
        fmins = []
        for s in self.hmc_samples:
            if self.model._fixes_ is None:
                self.model[:] = s
            else:
                self.model[self.model._fixes_] = s
            self.model._trigger_params_changed()
            fmins.append(self._predict(self.model.X[:, :self.decision_dim])[0].min())
        self.model.param_array[:] = ps
        self.model._trigger_params_changed()

        return fmins

    def get_fmax(self):
        """
        Returns the location where the posterior mean is takes its minimal value.
        """
        ps = self.model.param_array.copy()
        fmaxs = []
        for s in self.hmc_samples:
            if self.model._fixes_ is None:
                self.model[:] = s
            else:
                self.model[self.model._fixes_] = s
            self.model._trigger_params_changed()
            fmaxs.append(self._predict(self.model.X[:, :self.decision_dim])[0].max())
        self.model.param_array[:] = ps
        self.model._trigger_params_changed()

        return fmaxs

    def predict_withGradients(self, X):
        """
        Returns the mean, standard deviation, mean gradient and standard deviation gradient at X for all the MCMC samples.
        """
        # if X.ndim==1: X = X[None,:]
        # ps = self.model.param_array.copy()
        # means = []
        # stds = []
        # dmdxs = []
        # dsdxs = []
        # for s in self.hmc_samples:
        #     if self.model._fixes_ is None:
        #         self.model[:] = s
        #     else:
        #         self.model[self.model._fixes_] = s
        #     self.model._trigger_params_changed()
        #     m, v = self.model.predict(X)
        #     std = np.sqrt(np.clip(v, 1e-10, np.inf))
        #     dmdx, dvdx = self.model.predictive_gradients(X)
        #     dmdx = dmdx[:,:,0]
        #     dsdx = dvdx / (2*std)
        #     means.append(m)
        #     stds.append(std)
        #     dmdxs.append(dmdx)
        #     dsdxs.append(dsdx)
        # self.model.param_array[:] = ps
        # self.model._trigger_params_changed()
        # return means, stds, dmdxs, dsdxs
        raise NotImplementedError('')

    def copy(self):
        """
        Makes a safe copy of the model.
        """

        copied_model = GaussianFiniteQuadrature_MCMC( kernel = self.model.kern.copy(),
                                noise_var= self.noise_var ,
                                exact_feval= self.exact_feval,
                                n_samples = self.n_samples,
                                n_burnin = self.n_burnin,
                                subsample_interval = self.subsample_interval,
                                step_size = self.step_size,
                                leapfrog_steps= self.leapfrog_steps,
                                verbose= self.verbose,
                                decision_dim = self.decision_dim,
                                w_quadrate=self.w_quadrate, 
                                w_domain=self.w_domain 
                                )

        copied_model._create_model(self.model.X,self.model.Y)
        copied_model.updateModel(self.model.X,self.model.Y, None, None)
        return copied_model

    def get_model_parameters(self):
        """
        Returns a 2D numpy array with the parameters of the model
        """
        return np.atleast_2d(self.model[:])

    def get_model_parameters_names(self):
        """
        Returns a list with the names of the parameters of the model
        """
        return self.model.parameter_names()

class Quadrate(object):
    def __init__(self, w_type, w_bounds, distribution):
        """
        :param w_type: 0 if w is continuous, 1 if w is discrete and finite. 
        :param w_bounds:  array(n,d) bounds for w if w is continuous, and domain of w if w is discrete and finite. 
        :param distribution: An instance of 'Distribution'
        """
        self.w_type = w_type 
        self.w_bounds = w_bounds 
        self.distribution = distribution 
    def sample(self, n_samples=-1):
        raise NotImplementedError('')
        
    def integrate(self, f, double=False):
        raise NotImplementedError('')

class EmpiricalQuadrate(Quadrate):
    def __init__(self, w_bounds, distribution=None):
        if distribution is None:
            distribution = np.array([1./ w_bounds.shape[0]]*w_bounds.shape[0]  )
        super(EmpiricalQuadrate, self).__init__(1, w_bounds, distribution)

    def integrate(self, f, axis=None, double=False):
        """
        :param f: (mxn) if double=False
                  (mxnxn) if double=True
        
        Return:
            (m,)
        
        """
        if axis is None:
            axis = 1
        if not double:
            # weights: (n,)
            shape = [1]*f.ndim
            shape[axis] = -1 
            shape = tuple(shape)
            
            return np.sum( f * self.distribution.reshape(*shape), axis=axis)
        else:
            # f: (nxn)
            double_weights = np.dot(self.distribution.reshape(-1,1), self.distribution.reshape(1,-1))
            return np.sum(np.sum(f * double_weights[np.newaxis, :,:], axis=-1), axis=-1)

class DRQuadrate(Quadrate):
    def __init__(self, w_bounds):
        self.w_bounds = w_bounds
        self.w_type = 1 
        self.n_w = w_bounds.reshape[0]
    
    def integrate(self, f, df, axis=None, rho=0.1):
        # return drbo solution. 
        p, pw, dpw = min_weighted_sum_with_weights_on_chi_squared_ball(f, df, rho, return_grad=True)
        return pw, dpw 
        
class FiniteQuadrate(Quadrate):
    def __init__(self, domain, distribution=None, weights=None):
        n_sample = domain.shape[0]
        if distribution is None:
            if weights is None:
                weights = np.array([1./n_sample]*n_sample)
            distribution = FiniteWeighted(weights, domain )
        super(FiniteQuadrate, self).__init__(1, domain, distribution)
    
    def sample(self):
        return self.domain 

    def integrate(self, f, axis=None, double=False):
        """
        :param f: (mxn) if double=False
                  (mxnxn) if double=True
        
        Return:
            (m,)
        
        """
        if axis is None:
            axis = 1
        if not double:
            # weights: (n,)
            shape = [1]*f.ndim
            shape[axis] = -1 
            shape = tuple(shape)
            
            return np.sum( f * self.distribution.weights.reshape(*shape), axis=axis)
        else:
            # f: (nxn)
            double_weights = np.dot(self.distribution.weights.reshape(-1,1), self.distribution.weights.reshape(1,-1))
            return np.sum(np.sum(f * double_weights[np.newaxis, :,:], axis=-1), axis=-1)
        

class Distribution(object):
    def __init__(self, w_type): 
        self.w_type = w_type
    
    def sample(self):
        raise NotImplementedError('')

class FiniteWeighted(Distribution):
    def __init__(self, weights, domain):
        self.weights = weights
        self.domain = domain
        super(FiniteWeighted, self).__init__(1)
    def sample(self):
        return self.domain 
            
        
    