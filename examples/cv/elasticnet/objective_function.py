import numpy as np 
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
import sys
# sys.path.insert(0, '/home/thanhnt/DeakinML/GPyOpt')

from utils.compute_dr_weight import min_weighted_sum_with_weights_on_chi_squared_ball
from utils.parallel import run_function_different_arguments_parallel

class ElasticNet(object):
    '''
    ElaticNet_function: function

    param sd: standard deviation, to generate noisy evaluations of the function
    '''
    def __init__(self, X_train, Y_train, n_folds=10, minimize=True, var_noise=0, dim=2, \
                 log_alpha_min=-5, log_alpha_max=1):
        self._dim = dim
        self._w_dim = 1 # w is one-hot vector with n_t elements
        self._n_w = n_folds
        self._w_domain = None
        self._search_domain = np.array([[0, 1], [1e-8, 1]]) # scaled alpha, l1_ratio
        self.log_alpha_min, self.log_alpha_max = log_alpha_min, log_alpha_max
        self._sample_var = var_noise
        self._min_value = -1.
        self._minimize = minimize        
        self.X, self.Y = X_train, Y_train
        self.fold_size = int(self.X.shape[0] / n_folds)
        self.name = 'ElasticNet'
        
    def final_evaluate(self, x, X_train, Y_train, X_test, Y_test):
        u = x[0] # x[0] in [0,1] 
        scaled_u = (self.log_alpha_max - self.log_alpha_min) * u + self.log_alpha_min
        alpha = 10**scaled_u
        l1_ratio = x[1]
        
        clf = SGDClassifier(loss="log", penalty="elasticnet",
                            alpha=alpha, l1_ratio=l1_ratio,
                            max_iter=20, shuffle=True, random_state=1)
        clf.fit(X_train, Y_train)
        Y_test_pred = clf.predict(X_test)
        test_acc = accuracy_score(Y_test, Y_test_pred)
        if self._minimize:
            return 1. - test_acc
        else:
            return test_acc
    
    def evaluate_single_fold(self, k, x):
        print('Evaluating fold %d-th out of %d folds'%(k+1, self._n_w), ' at x = ', x)
        u = x[0] # x[0] in [0,1] 
        scaled_u = (self.log_alpha_max - self.log_alpha_min) * u + self.log_alpha_min
        alpha = 10**scaled_u 
        l1_ratio = x[1]
        fold_size = self.fold_size
        val_values = []
        clf = SGDClassifier(loss="log", penalty="elasticnet",
                            alpha=alpha, l1_ratio=l1_ratio,
                            max_iter=20, shuffle=True, random_state=1)
        
        val_ind = range(k*fold_size,(k+1)*fold_size)
        train_ind = np.array([ i for i in range(self.X.shape[0]) if i not in val_ind ]).astype('int')
        X_val = self.X[val_ind,:]
        Y_val = self.Y[val_ind]
        
        X_train = self.X[train_ind, :]
        Y_train = self.Y[train_ind]
                
        clf.fit(X_train, Y_train)

        Y_val_predict = clf.predict(X_val)
        val_acc = accuracy_score(Y_val, Y_val_predict)
        return val_acc
        
    def average_k_folds(self, x, rho=None):        
        arguments = dict(zip(range(self._n_w), range(self._n_w)))
        results = run_function_different_arguments_parallel(self.evaluate_single_fold, arguments, **{'x': x})
        print('K-fold values: ', results)
        val_values = []
        for k,v in results.items():
            val_values.append(v)
        val_values = np.array(val_values)
        if rho is None:
            val_acc = np.mean(val_values) 
        else:
            _, val_acc = min_weighted_sum_with_weights_on_chi_squared_ball(val_values, rho=rho)
        
        if self._minimize:
            return 1. - val_acc
        else:
            return val_acc
            
    def quadrate_evaluate(self, x):
        print('Evaluating at x = ', x) 
        # params: list of hyperparameters need to optimize
        u = x[0] # x[0] in [0,1] 
        scaled_u = (self.log_alpha_max - self.log_alpha_min) * u + self.log_alpha_min
        alpha = 10**scaled_u
        l1_ratio = x[1]
        k = int(np.argmax(x[2:])) # k in form of 1-hot vector 
        
        
        fold_size = self.fold_size

        clf = SGDClassifier(loss="log", penalty="elasticnet",
                            alpha=alpha, l1_ratio=l1_ratio,
                            max_iter=20, shuffle=True, random_state=1)
        
        val_ind = range(k*fold_size,(k+1)*fold_size)
        train_ind = np.array([ i for i in range(self.X.shape[0]) if i not in val_ind ]).astype('int')
        X_val = self.X[val_ind,:]
        Y_val = self.Y[val_ind]
        
        X_train = self.X[train_ind, :]
        Y_train = self.Y[train_ind]
                
        clf.fit(X_train, Y_train)

        Y_val_predict = clf.predict(X_val)
        val_acc = accuracy_score(Y_val, Y_val_predict)
        if self._minimize:
            return 1. - val_acc
        else:
            return val_acc
    
    # def evaluate(self, x):
    #     return self.evaluate_true(x) + np.sqrt(self._sample_var) * np.random.normal(0, 1)
    
    
def mnist_data_split(X_train, Y_train, total_samples=1000, cls_prop=None, shuffle=True):
    X_train_copied = np.copy(X_train)
    Y_train_copied = np.copy(Y_train)
    if shuffle:
        perm_ind = np.random.permutation(X_train_copied.shape[0])
        X_train_copied = X_train_copied[perm_ind,:]
        Y_train_copied = Y_train_copied[perm_ind]
        
    cls_prop = cls_prop / cls_prop.sum()
    cls_num = np.round(cls_prop * total_samples)
    cls_num[-1] = total_samples - cls_num[:-1].sum()
    cls_count = [int(i) for i in cls_num]
    
    X_train_new = np.zeros((total_samples, X_train_copied.shape[1]))
    Y_train_new = np.zeros((total_samples)).astype('uint8')

    j = 0 
    for x,y in zip(X_train_copied, Y_train_copied):
        if cls_count[int(y)] > 0 and j < total_samples:
            X_train_new[j,:] = x 
            Y_train_new[j] = y 
            cls_count[y] -= 1 
            j += 1 
        else:
            continue
    if shuffle:
        perm_ind = np.random.permutation(total_samples)
        X_train_new = X_train_new[perm_ind, :]
        Y_train_new = Y_train_new[perm_ind]
    return X_train_new, Y_train_new