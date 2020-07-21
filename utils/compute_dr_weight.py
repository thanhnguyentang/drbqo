"""
=== Optimization class I: Lagrangian multipliers for the inner opt **seems promising**
This approach directly compute the inner optimization and its gradients using the KKT conditions.
Given a fixed x, the inner opt is a constrained convex optimization thus the strong duality can apply. 

# Thanh Tang Nguyen <thanhnt@deakin.edu.au>, <nguyent2792@gmail.com>
"""
  
import warnings
import numpy as np  
    
def min_weighted_sum_with_weights_on_chi_squared_ball(w, dw=None, rho=0.1, tol=1e-10, return_grad=False):
    """
    (i) First, solving
        min_{p} sum_i p_i * w_i 
        s.t. 
            sum(p) = 1
            p_i >= 0 
            (1/n) * (0.5) * sum_i (n*p_i - 1)**2 <= rho (i.e., D_{phi}(p / hat{p}) <= rho) 
    
    (ii) Second, return 
        d ( min_{p} sum_i p_i * w_i ) / dx
        where dw is the gradients of w wrt dx. 
        Note that p, lam, and eta all depend on w. 
        Given a fixed x, the derivatives of p wrt x can be computed explicitly by taking derivatives of the KKT conditions. 
    
    --------------------- solving (i) ----------------------
    Solving (i) is equivalent to solving for its KKT conditions. 
        
    The dual variables:
        lam: for the constraint (1/n) * (0.5) * sum_i (n*p_i - 1)**2 <= rho
        eta: for the constraint sum(p) = 1
    
    Assume that w_1 <= w_2 <= ... <= w_n
        
    The KKT conditions for derivative wrt p and the constraint p_i >= 0 gives:
            p_i = (1/ (lam *n)) *(-w_i-eta) if 1 <= i <= I and 0 if i > I    (1)
            where I = |{i: w_sort_i + eta <= 0}|
            
    Compute eta: 
            eta = (-sum_{i=1}^I w_i - lam *n) / I                            (2)

    Obtaining a finite upper bound on lam^*:
        Using the condition (1/n) * (0.5) * sum_i (n*p_i - 1)**2 <= rho , we have:
        lam <= (-min(w) - eta) / sqrt(1 + 2*rho)                             (3)
            
        If eta <= - max(w), eta + w_i <= 0 for all i, and 
            eta = (-sum_{i=1}^n w_i - lam *n) / n                            (2')
        Replacing (2') into (3) to obtain an upper bound on lam^*.  
        
        Else, eta > -max(w). Plug in this inequality into (3) to obtain another upper bound on lam. 
        
        The final upper bound on lam is: 
    
            lam_max = max{  
                (-min(w) + (1/n)*sum_i w_i) / (sqrt(1 + 2*rho) - 1), # when -eta > max w_i 
                (-min(w) + max(w)) / sqrt(1 + 2*rho) # when eta in [-max w_i, -min w_i]
                }
            
            Note that if -eta < min w_i, then lam = 0 and p_i = 0 for all i (contradiction). 
            Thus this case never happens.  
            
        Given this upper bound on lam, we perform a bisection search on the following KKT condition for lam > 0:
            2*rho + 1 = n * sum_{i} p_i**2                                   (4)
            
    --------------------- Compute derivatives (ii) ----------------------
    
    If w.min = w.max:
        The derivative is simply dw[0,:], not dependent on p. 
    Elif lam = 0:
        w_1 = w_2 = ... = w_I = -eta < w_{I+1} <= ... 
        Thus, d(p^T w) dx = d (-eta) dx = dw_1 / dx = dw[0,:] if I >=1. 
    Else lam > 0:
        Taking derivatives of Eqs (1), (2) and (4) gives:
            p'_i = (-lam' / (n*lam**2))*(-w_i - eta)   
                + (1/ (lam *n)) *(-w'_i-eta')                                (5)
            I * eta' =  -sum_{i=1}^I w'_i - lam' *n                          (6)
            sum_{i=1}^I p_i p'_i = 0                                         (7)
        
        Plug (6) into (5), then plug the result into (7) gives an explicit computation of eta':
        
            eta' * (n*lam + I*(eta + sum_{i=1}^I p_i * w_i)) = 
                -sum_{i=1}^I w'_i *(eta + sum_{j=1}^I p_j * w_j) - n * lam * sum_{i=1}^I p_i * w'_i 
            
            lam' = (-sum_{i=1}^I w'_i - eta' * I) / n 
    """
    # assert rho > 0, "Make sure rho >= 0, if you want to set rho =0, set it to a small positive number, e.g., rho = 1e-6."
    if dw is not None:
        return_grad = True 

    if isinstance(w, np.ndarray):
        w = w.reshape(-1,)
    
    # if isinstance(dw, np.ndarray):
    #     dw = dw.reshape(-1,)

    assert rho >= 0
    if rho == 0:
        rho = 1e-10
    n = np.size(w)
    if w.min() == w.max():
        print('Equiprobale w=%s.'%(str(w)))
        p = np.zeros(n)
        p[0] = 1.
        if return_grad:
            return p, w[0], dw[0,:]
        else:
            return p, w[0]
    else:
        w_sort = np.sort(w) # increasing
        w_sort_cumsum = w_sort.cumsum()
        if return_grad:
            ind_map = np.argsort(w)
            ind_inv_map = np.argsort(ind_map)
            dw_sort = dw[ind_map]
        lam_min = 1e-10
        lam_max = max(
            (w_sort[-1] - w_sort[0]) / np.sqrt(1 + 2*rho), 
            (-w_sort[0] + np.mean(w)) / (np.sqrt(1 + 2*rho) - 1)
            )
#         print('lam_max = %s'%(str(lam_max)))
        lam_init_max = lam_max

        if lam_max < 0 or rho > 0.5*(n-1): #lam = 0 
            if rho > 0.5*(n-1):
                warnings.warn('rho > 0.5*(n-1), thus lam = 0 and all weights associated the the individual losses > their min are zeros.')
            eta, ind = solve_inner_eta_for_weighted_sum(w_sort, w_sort_cumsum, n, 0.)
            if not np.all(-w -eta <= 0):
                print('Make sure w + eta >=0 ')
                print('[====================ATTENTION==============]')
                print('-w-eta = ', -w-eta)
            # n_eta_at_w = (-w - eta == 0).sum() 
            n_eta_at_w = (-w - eta >= 0).sum() # some numerical error, -w -eta might be a very small positive
            eta_ind_at_w = np.argmax(w + eta == 0)
            p = np.ones(n)
            p[-w-eta < 0] = 0
            p = p / n_eta_at_w
            if return_grad:
                return p, -eta, dw[eta_ind_at_w,:]
            else:
                return p, -eta
            # if np.all(-w - eta <= 0):
            #     p = np.ones(n)
            #     p[-w-eta < 0] = 0
            #     n_eta_at_w = (-w - eta == 0).sum()
            #     if n / (2*rho + 1) <= n_eta_at_w: #this implicitly says that  n_eta_at_w > 0
            #         print("Replace the nonnegative entries in p with any positive number in (0,1) s.t. sum_i p_i = 1 and \
            #         sum_i p_i**2 <= (1 + 2*rho)/n = %s. \
            #         Currently, we choose to return p_i = 1 / n_eta_at_w for non-negative entries."%(str((1+2*rho)/n)))
            #         eta_ind_at_w = np.argmax(w + eta == 0)
            #         if return_grad:
            #             return p / n_eta_at_w, -eta , dw[eta_ind_at_w,:]
            #         else:
            #             return p / n_eta_at_w, -eta 
        # This is when lam > 0, thus Eq. (4) is satisfied, implying an upper bound on rho:
        #       2*rho + 1 = n * sum_{i} p_i**2 <= n * (sum_i p_i)**2 = n 
        #       rho <= 0.5*(n-1) 
        # assert rho <= 0.5*(n-1), "The problem has no solution with rho = %s. Specify rho <= 0.5*(n-1)."%(str(rho))
        else: 
        # bisect search 
            def kkt_lam(lam):
                eta, ind = solve_inner_eta_for_weighted_sum(w_sort, w_sort_cumsum, n, lam)
                p = (1 / (lam * n)) * (-w - eta)
                p[p<0] = 0 
                return p, 0.5 + rho - 0.5*n*(p**2).sum(), eta, ind
            while (lam_max - lam_min > tol*lam_init_max):
                lam = 0.5 * (lam_max + lam_min) 
                p, kkt_val,_, _ = kkt_lam(lam)
                if kkt_val < 0:
                    lam_min = lam 
                else:
                    lam_max = lam 
            lam = 0.5*(lam_min + lam_max)
            p, kkt_v, eta, ind = kkt_lam(lam)
            
            pw = p.dot(w.T)
            if return_grad:
                dw_sort_sum = np.sum(dw_sort[:ind+1], axis=0)
                deta = (
                    (eta + pw) * (-1.) * dw_sort_sum - n*lam* np.sum(p.reshape(-1,1)*dw, axis=0)
                ) / (n*lam + (ind+1)*(eta + pw ) + 1e-10)
                dlam = (-dw_sort_sum - (ind+1)*deta) / n 
                dp_sort = ( (-dlam / lam)*(-w_sort.reshape(-1,1)-eta) - dw_sort - deta) / (n*lam)
                dp_sort[ind+1:,:] = 0
                dp = dp_sort[ind_inv_map]
                #CHECK
                    # print(np.sum(dp, axis=0)) # Should be ~ zeros
                return p, pw, np.sum(dp*w.reshape(-1,1) + p.reshape(-1,1)*dw , axis=0)
            else:
                return p, pw
        
def solve_inner_eta_for_weighted_sum(w_sort_inc, w_sort_inc_cumsum, n, lam):
    """
    Given lam >=0, solve for eta, the dual variable for the constraint sum(p) = 1, that satisfies: 
            lam * n = sum_i max(-w_i - eta, 0)
    where lam * n * p_i = max(-w_i - eta, 0) 

    @CHECK: 
    **Edge case 1**: lam = 0 
        When lam = 0, we expect -eta = w_1.
        This solver implicitly considers this edge case because it starts from ind = 1 (not 0) so no problem at all.
        
    **Edge case 2**: When -eta > w_n, ind = n-1 (i.e., I = ind + 1 = n). This already covers it! 
    """
    eta = (-w_sort_inc_cumsum - lam *n) / np.arange(1, n+1)
    ind = (-w_sort_inc - eta >= 0).sum() -1 
    return eta[ind], ind