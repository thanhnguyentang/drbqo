# Purpose: To compare EI, TS and KG on quadrature optimization. 
#   These functions can also be used to check for rho-regret when using DR variant. 
from __future__ import division
from builtins import object
from past.utils import old_div
import numpy
import numpy as np
from numpy import abs, cos, exp, mean, pi, prod, sin, sqrt, sum
import math

# Input space is scaled to [0,1] 
# Minimization 
class LogisticLoss_nD(object):
    def __init__(self, minimize=True, var_noise=0, dim=1):
        self._dim = dim
        self._w_dim = dim 
        self._n_w = 1
        self._w_domain = np.array([[0.1]]*dim ).reshape(1,-1)
        self._search_domain = numpy.array([[0.0, 1.0]]*dim)
        self._num_init_pts = 3
        self._sample_var = var_noise
        self._min_value = -1.3 # @TODO Not exact, need to check 
        self._observations = []#numpy.arange(self._dim)
        self._num_fidelity = 0
        self._minimize = minimize
        
    def evaluate_true(self, u):
        # u_scaled = 4*u - 2. # [-2, 2]
        x = 4*u[:self._dim] - 2 # scale x only to [-2,2]
        w = u[self._dim:]
        
        y = np.log(1 + np.exp( np.sum(x*w) ))
        
        if self._minimize:
            return y 
        else:
            return -y   
        
    def evaluate(self, x):
        # return self.evaluate_true(x)
        return self.evaluate_true(x) + numpy.sqrt(self._sample_var) * numpy.random.normal(0, 1)
    
class LogisticLoss_mt_nD(object):
    def __init__(self, minimize=True, var_noise=0, dim=1):
        self._dim = dim
        self._w_dim = 1 # w is one-hot vector with n_t elements
        self._n_t = 1
        self._w_domain = np.array([[0.1]]*dim ).reshape(1,-1)
        self._search_domain = numpy.array([[0.0, 1.0]]*dim)
        self._num_init_pts = 3
        self._sample_var = var_noise
        self._min_value = -1.3 # @TODO Not exact, need to check 
        self._observations = []#numpy.arange(self._dim)
        self._num_fidelity = 0
        self._minimize = minimize
        
    def evaluate_true(self, u):
        # u_scaled = 4*u - 2. # [-2, 2]
        x = 4*u[:self._dim] - 2 # scale x only to [-2,2]
        w = self._w_domain[int(np.argmax(u[self._dim:]))]
        y = np.log(1 + np.exp( np.sum(x*w) ))
        
        if self._minimize:
            return y 
        else:
            return -y   
        
    def evaluate(self, x):
        # return self.evaluate_true(x)
        return self.evaluate_true(x) + numpy.sqrt(self._sample_var) * numpy.random.normal(0, 1)


class LogisticLoss(object):
    def __init__(self, minimize=True, var_noise=0):
        self._dim = 1
        self._w_dim = 1 
        self._n_w = 6
        self._w_domain = np.array([[0.1], [0.2], [0.3], [0.4], [0.5], [0.6]]).reshape(6,1)
        self._search_domain = numpy.array([[0., 1.0]])
        self._num_init_pts = 3
        self._sample_var = var_noise
        self._min_value = -1.3 # @TODO Not exact, need to check 
        self._observations = []#numpy.arange(self._dim)
        self._num_fidelity = 0
        self._minimize = minimize
        
    def evaluate_true(self, u):
        u_scaled = 4*u - 2. # [-2, 2]
        x = u_scaled[0]
        w = u_scaled[1]
        
        y = np.log(1 + np.exp( x*w ))
        
        if self._minimize:
            return y 
        else:
            return -y   
        
    def evaluate(self, x):
        # return self.evaluate_true(x)
        return self.evaluate_true(x) + numpy.sqrt(self._sample_var) * numpy.random.normal(0, 1)


class My1DFunc(object):
    def __init__(self, minimize=True, var_noise=0):
        self._dim = 1
        self._w_dim = 1 
        self._n_w = 5
        self._w_domain = np.array([[0.1], [0.3], [0.5], [0.7], [0.9]]).reshape(5,1)
        self._search_domain = numpy.array([[0., 1.0]])
        self._num_init_pts = 3
        self._sample_var = var_noise
        self._min_value = -1.3 # @TODO Not exact, need to check 
        self._observations = []#numpy.arange(self._dim)
        self._num_fidelity = 0
        self._minimize = minimize
        
    def evaluate_single_true(self, u):
        x = 20*u - 10. # [-10, 10] 
        y = numpy.exp(-(x - 2)**2) + numpy.exp(-(x - 6)**2/10) + 1/ (x**2 + 1)
        if self._minimize:
            return -y 
        else:
            return y 
    def evaluate_true(self, u):
        x0 = u[0]
        x1 = u[1]
        y = self.evaluate_single_true(np.array(x0)) * self.evaluate_single_true(np.array(x1)) 
        
        if self._minimize:
            return y 
        else:
            return -y   
        
    def evaluate(self, x):
        # return self.evaluate_true(x)
        return self.evaluate_true(x) + numpy.sqrt(self._sample_var) * numpy.random.normal(0, 1)

class My1DFunc2(object):
    def __init__(self, minimize=True, var_noise=0):
        self._dim = 1
        self._w_dim = 1 
        self._n_w = 5
        self._w_domain = np.array([[0.1], [0.3], [0.5], [0.7], [0.9]]).reshape(5,1)
        self._search_domain = numpy.array([[0., 1.0]])
        self._num_init_pts = 3
        self._sample_var = var_noise
        self._min_value = -1.3 # @TODO Not exact, need to check 
        self._observations = []#numpy.arange(self._dim)
        self._num_fidelity = 0
        self._minimize = minimize
        
    def evaluate_single_true(self, u):
        x = 20*u - 10. # [-10, 10] 
        y = numpy.exp(-(x - 2)**2) + numpy.exp(-(x - 6)**2/10) + 1/ (x**2 + 1)
        if self._minimize:
            return -y 
        else:
            return y 
    def evaluate_true(self, u):
        x0 = u[0]
        x1 = u[1]
        y = -self.evaluate_single_true(np.array(x0)) * self.evaluate_single_true(np.array(x1)) 
        
        if self._minimize:
            return y 
        else:
            return -y   
        
    def evaluate(self, x):
        # return self.evaluate_true(x)
        return self.evaluate_true(x) + numpy.sqrt(self._sample_var) * numpy.random.normal(0, 1)

class Branin(object):
    def __init__(self, minimize=True, var_noise=0.0):
        self._dim = 2
        self._search_domain = numpy.array([[0.0, 1.0], [0.0, 1.0]])
        # self._search_domain = numpy.array([[0.0, 15.], [-5., 15.]])
        self._n_w = 10
        self._w_domain = np.array([[0.1, 0.1], [0.3, 0.3], [0.5, 0.5], [0.7, 0.7], [0.9, 0.9], 
                                   [0.3, 0.1], [0.5, 0.3], [0.7, 0.5], [0.9, 0.7], [0.1, 0.9]
                                   ]).reshape(10,2)
        self._num_init_pts = 3
        self._sample_var = var_noise
        self._min_value = 0.397887
        self._observations = []#numpy.arange(self._dim)
        self._num_fidelity = 0
        self._minimize = minimize

    def evaluate_single_true(self, x):
        """ This function is usually evaluated on the square x_1 \in [0, 15], x_2 \in [-5, 15]. Global minimum
        is at x = [pi, 2.275] and [9.42478, 2.475] with minima f(x*) = 0.397887.
            :param x[2]: 2-dim numpy array
            
        Transformed optimum:
            [0.20943951023931953, 0.36375], 
            [0.6283186666666667, 0.37374999999999997]
        """
        x0 = 15 * x[0]  # [0, 15.]
        x1 = 20 * x[1] - 5.0  # [-5, 15]
        # x0 = x[0]
        # x1 = x[1]
        a = 1
        b = old_div(5.1, (4 * pow(numpy.pi, 2.0)))
        c = old_div(5, numpy.pi)
        r = 6
        s = 10
        t = old_div(1, (8 * numpy.pi))
        y = numpy.array([(a * pow(x1 - b * pow(x0, 2.0) + c * x0 - r, 2.0) + s * (1 - t) * numpy.cos(x0) + s)])
        if self._minimize:
            return y 
        else:
            return -y 
        
    def evaluate_true(self, u):
        """
        Args:
            u: [x1 x2 w1 w2]
        """
        x0 = u[0]
        x3 = u[1]
        x1 = u[2]
        x2 = u[3]
        y = self.evaluate_single_true(np.array([x0, x1])) * self.evaluate_single_true(np.array([x2,x3]))
        
        if self._minimize:
            return y 
        else:
            return -y 
        
    def evaluate(self, x):
        # return self.evaluate_true(x)
        return self.evaluate_true(x) + numpy.sqrt(self._sample_var) * numpy.random.normal(0, 1)

class Rosenbrock(object):
    def __init__(self, minimize=True, var_noise=0.):
        self._dim = 2
        self._search_domain = numpy.repeat([[0., 1.]], self._dim, axis=0)
        self._n_w = 10
        self._w_domain = np.array([[0.1, 0.1], [0.3, 0.3], [0.5, 0.5], [0.7, 0.7], [0.9, 0.9], 
                                   [0.3, 0.1], [0.5, 0.3], [0.7, 0.5], [0.9, 0.7], [0.1, 0.9]
                                   ]).reshape(10,2)
        self._num_init_pts = 3
        self._sample_var = var_noise
        self._min_value = 0.0
        self._observations = []
        self._num_fidelity = 0
        self._minimize = minimize

    def evaluate_single_true(self, u):
        """ Global minimum is 0 at (1, 1, 1, 1)
        Transformed minimum: (0.75, 0.75)
            :param x[4]: 4-dimension numpy array
        """
        x = 4*u - 2. #  [-2, 2]
        # for i in range(self._dim):
        #     x[i] = 4 * x[i] - 2. 
        value = 0.0
        for i in range(self._dim-1):
            value += pow(1. - x[i], 2.0) + 100. * pow(x[i+1] - pow(x[i], 2.0), 2.0)
        results = [value]
        y = numpy.array(results)
        if self._minimize:
            return y 
        else:
            return -y 
        
    def evaluate_true(self, u):
        """
        Args:
            u: [x1 x2 w1 w2]
        """
        x0 = u[0]
        x3 = u[1]
        x1 = u[2]
        x2 = u[3]
        y = self.evaluate_single_true(np.array([x0, x1])) * self.evaluate_single_true(np.array([x2,x3]))
        
        if self._minimize:
            return y 
        else:
            return -y 

    def evaluate(self, x):
        # return self.evaluate_true(x)
        return self.evaluate_true(x) + numpy.sqrt(self._sample_var) * numpy.random.normal(0, 1)

class Hartmann3(object):
    def __init__(self, minimize=True, var_noise=0.0):
        self._dim = 3
        self._search_domain = numpy.repeat([[0., 1.]], self._dim, axis=0)
        self._num_init_pts = 3
        self._n_w = 15
        self._w_domain = np.array([[0.1, 0.1, 0.1], [0.3, 0.3, 0.3], [0.5, 0.5, 0.5], [0.7, 0.7, 0.7], [0.9, 0.9, 0.9], 
                                   [0.9, 0.1, 0.1], [0.1, 0.3, 0.3], [0.3, 0.5, 0.5], [0.5, 0.7, 0.7], [0.7, 0.9, 0.9], 
                                   [0.9, 0.7, 0.1], [0.1, 0.9, 0.3], [0.3, 0.1, 0.5], [0.5, 0.3, 0.7], [0.7, 0.5, 0.9], 
                                   ]).reshape(15,3)
        self._sample_var = var_noise
        self._min_value = -3.86278
        self._observations = []#numpy.arange(self._dim)
        self._num_fidelity = 0
        self._minimize = minimize

    def evaluate_single_true(self, x):
        """ domain is x_i \in (0, 1) for i = 1, ..., 3
            Global minimum is -3.86278 at (0.114614, 0.555649, 0.852547)
            :param x[3]: 3-dimension numpy array with domain stated above
        """
        alpha = numpy.array([1.0, 1.2, 3.0, 3.2])
        A = numpy.array([[3., 10., 30.], [0.1, 10., 35.], [3., 10., 30.], [0.1, 10., 35.]])
        P = 1e-4 * numpy.array([[3689, 1170, 2673], [4699, 4387, 7470], [1091, 8732, 5547], [381, 5743, 8828]])
        results = 0.
        for i in range(4):
            inner_value = 0.0
            for j in range(self._dim):
                inner_value -= A[i, j] * pow(x[j] - P[i, j], 2.0)
            results -= alpha[i] * numpy.exp(inner_value)
        y = numpy.array(results)
        if self._minimize:
            return y 
        else:
            return -y 
    
    def evaluate_true(self, u):
        """
        Args:
            u: [x1 x2 x3 w1 w2 w3]
        """
        x0 = u[0]
        x2 = u[1]
        x4 = u[2]
        x1 = u[3]
        x3 = u[4]
        x5 = u[5]
        
        y = self.evaluate_single_true(np.array([x0, x1, x2])) * self.evaluate_single_true(np.array([x3,x4,x5]))
        
        if self._minimize:
            return y 
        else:
            return -y 

    def evaluate(self, x):
        return self.evaluate_true(x) + numpy.sqrt(self._sample_var) * numpy.random.normal(0, 1)

class Levy4(object):
    def __init__(self, minimize=True, var_noise=0.):
        self._dim = 4
        self._search_domain = numpy.repeat([[0., 1.]], self._dim, axis=0)
        self._n_w = 20
        self._w_domain = np.array([[0.1, 0.1, 0.1, 0.1], [0.3, 0.3, 0.3, 0.3], [0.5, 0.5, 0.5, 0.5], [0.7, 0.7, 0.7, 0.7], [0.9, 0.9, 0.9, 0.9], 
                                   [0.3, 0.1, 0.1, 0.1], [0.5, 0.3, 0.3, 0.3], [0.7, 0.5, 0.5, 0.5], [0.9, 0.7, 0.7, 0.7], [0.1, 0.9, 0.9, 0.9],
                                   [0.3, 0.1, 0.5, 0.1], [0.5, 0.3, 0.7, 0.3], [0.7, 0.5, 0.9, 0.5], [0.9, 0.7, 0.1, 0.7], [0.1, 0.9, 0.3, 0.9],
                                   [0.3, 0.1, 0.5, 0.7], [0.5, 0.3, 0.7, 0.9], [0.7, 0.5, 0.9, 0.1], [0.9, 0.7, 0.1, 0.3], [0.1, 0.9, 0.3, 0.5],
                                   ]).reshape(20,4)
        self._num_init_pts = 3
        self._sample_var = var_noise
        self._min_value = 0.0
        self._observations = []#numpy.arange(self._dim)
        self._num_fidelity = 0
        self._minimize = minimize

    def evaluate_single_true(self, u):
        """ Global minimum is 0 at (1, 1, 1, 1)
            Transformed minimum: (0.6, 0.6, 0.6, 0.6)
            :param x[4]: 4-dimension numpy array
            a difficult test case for KG-type methods.
        """
        x = 10. * u - 5. #[-5., 5.]
        x = numpy.asarray_chkfinite(x)
        n = len(x)
        z = 1 + old_div((x - 1), 4)

        results = (sin( pi * z[0] )**2
                      + sum( (z[:-1] - 1)**2 * (1 + 10 * sin( pi * z[:-1] + 1 )**2 ))
                      +       (z[-1] - 1)**2 * (1 + sin( 2 * pi * z[-1] )**2 ))
        y = numpy.array(results)
        if self._minimize:
            return y 
        else:
            return -y 

    def evaluate_true(self, u):
        """
        Args:
            u: [x1 x2 x3 x4 w1 w2 w3 x4]
        """
        x0 = u[0]
        x2 = u[1]
        x4 = u[2]
        x6 = u[3]
        x1 = u[4]
        x3 = u[5]
        x5 = u[6]
        x7 = u[7]
        
        y = self.evaluate_single_true(np.array([x0, x1, x2, x3])) * self.evaluate_single_true(np.array([x4,x5, x6,x7]))
        
        if self._minimize:
            return y 
        else:
            return -y 

    def evaluate(self, x):
        # return self.evaluate_true(x)
        return self.evaluate_true(x) + numpy.sqrt(self._sample_var) * numpy.random.normal(0, 1)

class Hartmann6(object):
    def __init__(self, minimize=True, var_noise=0.0):
        self._dim = 6
        self._search_domain = numpy.repeat([[0., 1.]], self._dim, axis=0)
        self._n_w = 30
        self._w_domain = np.array(
        [[0.1, 0.1, 0.1, 0.1, 0.1, 0.1], [0.3, 0.3, 0.3, 0.3, 0.3, 0.3], [0.5, 0.5, 0.5, 0.5, 0.5, 0.5], [0.7, 0.7, 0.7, 0.7, 0.7, 0.7], [0.9, 0.9, 0.9, 0.9, 0.9, 0.9], 
        [0.3, 0.1, 0.1, 0.1, 0.1, 0.1], [0.5, 0.3, 0.3, 0.3, 0.3, 0.3], [0.7, 0.5, 0.5, 0.5, 0.5, 0.5], [0.9, 0.7, 0.7, 0.7, 0.7, 0.7], [0.1, 0.9, 0.9, 0.9, 0.9, 0.9],              
        [0.3, 0.5, 0.1, 0.1, 0.1, 0.1], [0.5, 0.7, 0.3, 0.3, 0.3, 0.3], [0.7, 0.9, 0.5, 0.5, 0.5, 0.5], [0.9, 0.1, 0.7, 0.7, 0.7, 0.7], [0.1, 0.3, 0.9, 0.9, 0.9, 0.9],             
        [0.3, 0.5, 0.7, 0.1, 0.1, 0.1], [0.5, 0.7, 0.9, 0.3, 0.3, 0.3], [0.7, 0.9, 0.1, 0.5, 0.5, 0.5], [0.9, 0.1, 0.3, 0.7, 0.7, 0.7], [0.1, 0.3, 0.5, 0.9, 0.9, 0.9],             
        [0.3, 0.5, 0.7, 0.9, 0.1, 0.1], [0.5, 0.7, 0.9, 0.1, 0.3, 0.3], [0.7, 0.9, 0.1, 0.3, 0.5, 0.5], [0.9, 0.1, 0.3, 0.5, 0.7, 0.7], [0.1, 0.3, 0.7, 0.9, 0.9, 0.9],             
        [0.3, 0.5, 0.7, 0.9, 0.3, 0.1], [0.5, 0.7, 0.9, 0.1, 0.5, 0.3], [0.7, 0.9, 0.1, 0.3, 0.7, 0.5], [0.9, 0.1, 0.3, 0.5, 0.9, 0.7], [0.1, 0.3, 0.7, 0.9, 0.1, 0.9]          
        ]).reshape(30,6)
        self._num_init_pts = 3
        self._sample_var = var_noise
        self._min_value = -3.32237
        self._observations = []#numpy.arange(self._dim)
        self._num_fidelity = 0
        self._minimize = minimize

    def evaluate_single_true(self, x):
        """ domain is x_i \in (0, 1) for i = 1, ..., 6
            Global minimum is -3.32237 at (0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573)
            :param x[6]: 6-dimension numpy array with domain stated above
        """
        alpha = numpy.array([1.0, 1.2, 3.0, 3.2])
        A = numpy.array([[10, 3, 17, 3.50, 1.7, 8], [0.05, 10, 17, 0.1, 8, 14], [3, 3.5, 1.7, 10, 17, 8],
                         [17, 8, 0.05, 10, 0.1, 14]])
        P = 1.0e-4 * numpy.array([[1312, 1696, 5569, 124, 8283, 5886], [2329, 4135, 8307, 3736, 1004, 9991],
                                  [2348, 1451, 3522, 2883, 3047, 6650], [4047, 8828, 8732, 5743, 1091, 381]])
        results = 0.
        for i in range(4):
            inner_value = 0.0
            for j in range(self._dim-self._num_fidelity):
                inner_value -= A[i, j] * pow(x[j] - P[i, j], 2.0)
            results -= alpha[i] * numpy.exp(inner_value)
            # for j in range(self._dim-self._num_fidelity):
            #     results[j+1] -= (alpha[i] * numpy.exp(inner_value)) * ((-2) * A[i,j] * (x[j] - P[i, j]))
        y = numpy.array(results)
        if self._minimize:
            return y 
        else:
            return -y 

    def evaluate_true(self, u):
        """
        Args:
            u: [x1 x2 x3 x4 x5 x6 w1 w2 w3 w4 w5 w6]
        """
        x0 = u[0]
        x2 = u[1]
        x4 = u[2]
        x7 = u[3]
        x9 = u[4]
        x11 = u[5]
        x1 = u[6]
        x3 = u[7]
        x5 = u[8]
        x6 = u[9]
        x8 = u[10]
        x10 = u[11]
        
        y = self.evaluate_single_true(np.array([x0,x1,x2,x3,x4,x5])) * self.evaluate_single_true(np.array([x6,x7,x8,x9,x10,x11]))
        
        if self._minimize:
            return y 
        else:
            return -y 

    def evaluate(self, x):
        return self.evaluate_true(x) + numpy.sqrt(self._sample_var) * numpy.random.normal(0, 1)

class Ackley(object):
    def __init__(self, minimize=True, var_noise=0.):
        self._dim = 5
        self._search_domain = numpy.repeat([[0., 1.]], self._dim, axis=0)
        self._n_w = 25
        self._w_domain = np.array(
            [[0.1, 0.1, 0.1, 0.1, 0.1], [0.3, 0.3, 0.3, 0.3, 0.3], [0.5, 0.5, 0.5, 0.5, 0.5], [0.7, 0.7, 0.7, 0.7, 0.7], [0.9, 0.9, 0.9, 0.9, 0.9], 
             [0.3, 0.1, 0.1, 0.1, 0.1], [0.5, 0.3, 0.3, 0.3, 0.3], [0.7, 0.5, 0.5, 0.5, 0.5], [0.9, 0.7, 0.7, 0.7, 0.7], [0.1, 0.9, 0.9, 0.9, 0.9],                                   
             [0.3, 0.5, 0.1, 0.1, 0.1], [0.5, 0.7, 0.3, 0.3, 0.3], [0.7, 0.9, 0.5, 0.5, 0.5], [0.9, 0.1, 0.7, 0.7, 0.7], [0.1, 0.3, 0.9, 0.9, 0.9], 
             [0.3, 0.5, 0.7, 0.1, 0.1], [0.5, 0.7, 0.9, 0.3, 0.3], [0.7, 0.9, 0.1, 0.5, 0.5], [0.9, 0.1, 0.3, 0.7, 0.7], [0.1, 0.3, 0.5, 0.9, 0.9],                                   
             [0.3, 0.5, 0.7, 0.9, 0.1], [0.5, 0.7, 0.9, 0.1, 0.3], [0.7, 0.9, 0.1, 0.3, 0.5], [0.9, 0.1, 0.3, 0.5, 0.7], [0.1, 0.3, 0.5, 0.7, 0.9],                           
                     ]).reshape(25,5)
        self._num_init_pts = 3
        self._sample_var = var_noise
        self._min_value = 0.0
        self._observations = []
        self._num_fidelity = 0
        self._minimize = minimize

    def evaluate_single_true(self, u):
        x = 2*u - 1 #[-1,1]
        x = 20*x
        firstSum = 0.0
        secondSum = 0.0
        for c in x:
            firstSum += c**2.0
            secondSum += math.cos(2.0*math.pi*c)
        n = float(len(x))
        results=[old_div((-20.0*math.exp(-0.2*math.sqrt(old_div(firstSum,n))) - math.exp(old_div(secondSum,n)) + 20 + math.e),6.)]
        # for i in range(int(n)):
        #     results += [-20.0*math.exp(-0.2*math.sqrt(old_div(firstSum,n))) * (-0.2*(old_div(x[i],n))/(math.sqrt(old_div(firstSum,n)))) -
        #                 math.exp(old_div(secondSum,n)) * (2.0*math.pi/n) * (-math.sin(2.0*math.pi*x[i]))]
        y = numpy.array(results)
        
        if self._minimize:
            return y 
        else:
            return -y 
    
    def evaluate_true(self, u):
        """
        Args:
            u: [x1 x2 x3 x4 x5 w1 w2 w3 w4 w5]
        """
        x0 = u[0]
        x2 = u[1]
        x4 = u[2]
        x6 = u[3]
        x8 = u[4]
        x1 = u[5]
        x3 = u[6]
        x5 = u[7]
        x7 = u[8]
        x9 = u[9]
        
        y = self.evaluate_single_true(np.array([x0, x1, x2, x3, x4])) * self.evaluate_single_true(np.array([x5,x6,x7,x8,x9]))
        
        if self._minimize:
            return y 
        else:
            return -y 

    def evaluate(self, x):
        return self.evaluate_true(x) + numpy.sqrt(self._sample_var) * numpy.random.normal(0, 1)