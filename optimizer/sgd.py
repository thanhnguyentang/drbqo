import numpy as np 
import math 
import sys 

class Adam(object):
    """Adam optimizer (Minimization)
    Default parameters follow those provided in the original paper.
    # Arguments
        lr: float >= 0. Learning rate.
        beta_1: float, 0 < beta < 1. Generally close to 1.
        beta_2: float, 0 < beta < 1. Generally close to 1.
        epsilon: float >= 0. Fuzz factor. If `None`, defaults to `K.epsilon()`.
        decay: float >= 0. Learning rate decay over each update.
        amsgrad: boolean. Whether to apply the AMSGrad variant of this
            algorithm from the paper "On the Convergence of Adam and
            Beyond".
    # References
        - [Adam - A Method for Stochastic Optimization](
           https://arxiv.org/abs/1412.6980v8)
        - [On the Convergence of Adam and Beyond](
           https://openreview.net/forum?id=ryQu7f-RZ)
    """

    def __init__(self, lr=0.001, 
                 beta_1=0.9, 
                 beta_2=0.999,
                 epsilon=1e-8, 
                 decay=0., 
                 grad_func = None, 
                 bound = None,
                 dim = None,
                 amsgrad=False,
                 verbose=True, 
                 *args,  **kwargs):
        """Adam for stochastic optimization. 
        Args:
            grad_func: a noisy gradient estimator with a predefined batch size. 
            
        Caveat: Adam requires a good initialization to require less iterations. 
        """
        self.lr = lr 
        self.beta_1 = beta_1 
        self.beta_2 = beta_2 
        self.epsilon = epsilon 
        self.decay = decay
        self.init_decay = decay 
        self.grad_func = grad_func
        self.bound = bound 
        if dim is None and bound is not None:
            dim = bound.shape[0]
        self.dim = dim 
        self._it = 0 
        self.args = args 
        self.kwargs = kwargs
        self.verbose = verbose
        # self._m = np.zeros(*bound.shape[0])
        # self._v = np.zeros(*bound.shape[0])
        
    def get_gradients(self, params):
        """
        Args:
            params: (n,)
        
        Return:
            (n,)
        """
        return self.grad_func(params, *self.args, **self.kwargs)
    
    def constraint(self, p):
        if self.bound is not None:
            return np.clip(p.reshape(1,-1), self.bound[:,0], self.bound[:,1]).reshape(-1)
        else:
            return p 

    def get_updates(self, p):
        if not hasattr(self, '_m'):
            self._m = np.zeros_like(p)
        if not hasattr(self, '_v'):
            self._v = np.zeros_like(p)
            
        g = self.get_gradients(p)
        lr = self.lr
        if self.init_decay > 0:
            lr = lr * (1. / (1. + self.decay * self._it))  

        t = self._it + 1
        lr_t = lr * (math.sqrt(1. - math.pow(self.beta_2, t)) / (1. - math.pow(self.beta_1, t)))
        
        # print(lr_t)

        self.weights = [self._it] + self._m + self._v
        
        m_t = (self.beta_1 * self._m) + (1. - self.beta_1) * g 
        v_t = (self.beta_2 * self._v) + (1. - self.beta_2) * g**2 
        
        p_t = p - lr_t * m_t / (np.sqrt(v_t) + self.epsilon)
         
        self._m = m_t 
        self._v = v_t
        self._it += 1 
        return self.constraint(p_t) 
    
    def optimize(self, start=None, maxit=300):
        if start is None:
            if self.bound is not None:
                start = np.random.uniform(self.bound[:,0], self.bound[:,1], (1, self.dim))
            else:
                start = np.random.randn(1, self.dim)
        self._m = np.zeros_like(start)
        self._v = np.zeros_like(start)
        point = start #.reshape(-1)
        for i in range(maxit):
            if self.verbose:
                sys.stdout.write('Adam: it=%d/%d\r'%(i + 1, maxit))
                sys.stdout.flush()

            previous = np.copy(point)
            point = self.get_updates(point) 
            
            den_norm = (np.sqrt(np.sum(previous ** 2)))

            if den_norm == 0:
                norm = np.sqrt(np.mean((previous - point) ** 2)) / 1e-2
            else:
                norm = np.sqrt(np.mean((previous - point) ** 2)) / den_norm
            if norm < 1e-4:
                # print('break')
                break
            # print(norm)
        return point