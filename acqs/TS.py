"""Thompson sampling. 

Improved Thompson: 
    Hypothesis: TS does not guarantee that it won't rechoose the previous chosen points.
    Solution: Encourage point-distinction as in the Stein's Variational https://arxiv.org/pdf/1608.04471.pdf
    ts(x) as a log of some unnormalized density function log p(x). Thus we want to maximize \sum_i k(x, x_i) log p(x).
    
Practical Performance of Thompson sampling:
    - It seems there not many use TS in Gaussian Process Optimization. Why is that? 
    - Yes, it more seems that there not many use in Quadrature Optimization. 
        Hypothesis:
            If f1(x,w) is a Posterior sample of f, does (Tf1) is a function sample of (Tf) where T is the expectation wrt some fixed p(x)? 

# Thanh Tang Nguyen <thanhnt@deakin.edu.au>, <nguyent2792@gmail.com>
"""
import numpy as np 
from scipy.linalg import cholesky, cho_solve, solve_triangular

def TS_posterior_func_sample(model, m, n_fs, data, kernel_type='RBF'):
    """
    args:
        model: GPy.models.GPRegression, or GPy.models.SparseGPRegression 
        m: Number of samples 
        n_fs: Number of posterior functions.
        data: train_X, train_Y 
        kernel_type: 'RBF', 'Matern32', 'Matern52'
    """
    d = data[0].shape[1]
    sigma_n = np.sqrt(model._get_Gaussian_noise_variance())
    alpha = model._get_variance()
    ls = model._get_lengthscale()
    if np.size(ls) == 1 and 1 < d:
        S = np.diag(np.array([1./ ls**2]*d)) 
    else:
        S = np.diag(1./ ls**2) 
    if kernel_type == 'RBF':
        sd_sampler = MultivariateNormalSampler(np.zeros(d), S)
    elif kernel_type == 'Matern32':
        sd_sampler = MultivariateStudentSampler(np.zeros(d), S, nu=1.5) 
    elif kernel_type == 'Matern52':
        sd_sampler = MultivariateStudentSampler(np.zeros(d), S, nu=2.5) 
    else:
        raise ValueError(kernel_type)
    posterior_fs = Thomson_sampling(alpha, sigma_n, sd_sampler, data, m, n_fs)
    return posterior_fs
    
def Thomson_sampling(alpha, sigma_n, sd_sampler, data, m, n_fs = 1):
    """
    An shift-invariance kernel k can be represented as:
        k(x,x') = 2 * alpha * E_{p(w,b)} [cos(w^T x + b ) cos(w^T x' + b)]
    where:
        b ~ U[0, 2*pi].
        w ~ p(w) \prop s(w) / alpha.
        s(w) is the Fourier dual of k.  
        alpha = \int s(w) dw.
        
    A posterior sample of f is then described as:
        f(x) = Phi(x)^T theta 
    
    Args:
        alpha:
        sigma_n: 
        sd_sampler: spectral density sampler 
        S:
        X_train:
        m: 
        n_fs: Number of posterior function samples. 
    """
    # Notations from https://arxiv.org/abs/1406.2541 
    X_train, Y_train = data[0], data[1]
    posterior_fs = {}
    for i in range(n_fs):
        W = sd_sampler.sample(m) # (m x d)
        b = np.random.uniform(0, 2*np.pi, size=m) # (m,)
        def phi(x):
            """
            Argument:
                x: (nxd)
                W: (mxd)
                b: (m,)
            
            Return:
                (nxm)
            """
            return np.sqrt(2*alpha / m) * np.cos(np.matmul(x, W.T) + b)

        Phi = phi(X_train) # (nxm)
        A = np.matmul(Phi.T, Phi) # (mxm)
        A[np.diag_indices_from(A)] += sigma_n**2
        try:
            L = cholesky(A, lower=True)
        except np.linalg.LinAlgError as exc:
            A[np.diag_indices_from(A)] += 1e-8 
            L = cholesky(A, lower=True)
        theta_mean = cho_solve((L, True), np.matmul(Phi.T, Y_train)) # A^{-1} Phi^T y_n 
        L_inv_trans = sigma_n * solve_triangular(L.T, np.eye(L.shape[0]))
        # Sample from Gaussian using Cholesky: https://math.stackexchange.com/questions/2079137/generating-multivariate-normal-samples-why-cholesky
        theta = np.matmul(L_inv_trans, np.random.multivariate_normal(mean=np.zeros(m), cov=np.eye(m), size=1).T) + theta_mean 
        posterior_fs[i] = lambda x: np.matmul(phi(x), theta)
    return posterior_fs
    
# Formular for spectral density of RBF and Matern kernel: https://link.springer.com/article/10.1007/s10898-018-0609-2
class MultivariateNormalSampler(object):
    def __init__(self, m, S):
        self.m = m 
        self.S = S
    def sample(self, n):
        return np.random.multivariate_normal(mean=self.m,cov=self.S, size=n)
    
class MultivariateStudentSampler(object):
    def __init__(self, m, S, nu):
        self.m = m 
        self.S = S 
        self.nu = nu # if nu = np.inf, t-student reduces to Gaussian. 
    def sample(self, n):
        d = len(self.m)
        if self.nu == np.inf:
            x = 1.
        else:
            x = np.random.chisquare(nu, n)/self.nu
        z = np.random.multivariate_normal(np.zeros(d), self.S, (n,))
        return self.m + z/np.sqrt(x)[:,None]