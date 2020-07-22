import numpy as np 
import tensorflow as tf 
import sys
import math
from utils.compute_dr_weight import min_weighted_sum_with_weights_on_chi_squared_ball
from utils.parallel import run_function_different_arguments_parallel

# Same setting as in:
# https://www.dropbox.com/sh/u0vd5cvpf73ztxb/AACgjCDhq7e2URIU805jPLCva?dl=0&preview=functions_ml_model.py 
# We construct a 2-layer MLP with 512 neurons/layer. 
#   We optimize three hypeparameters: the learning rate l and the L2 norm regularization parameters lr1 and lr2 of the
#   two layers. All the hyperparameters are tuned in the exponent space (base 10).
#   The MLP model is implemented using tensorflow. 
#   The model is trained with the Adam optimizer in 20 epochs and the batch size is 128.

class CNN(object):
    '''
    MLP_MNIST_function: function
 
    param sd: standard deviation, to generate noisy evaluations of the function
    '''
    # X_train, Y_train, n_folds=10, minimize=True, var_noise=0, dim=2, \
                #  log_alpha_min=-5, log_alpha_max=1
    def __init__(self, X_train, Y_train, n_folds=10, minimize=False, var_noise=0, seed=1, log_lb=-8, log_ub=3):
    # bounds=None, sd=None, seed=0):
        self._dim = 3
        self.minimize = minimize
        self.log_lb = log_lb 
        self.log_ub = log_ub 
        self._w_dim = 1 # w is one-hot vector with n_t elements
        self._n_w = n_folds
        self._w_domain = None
        self._search_domain = np.array([
                [1e-8, 1], # dropout1 
                [1e-8, 1], # dropout2
                [0, 1] # lr in log10 scale, then scaled to [0,1]
            ]) 
        self.X = X_train 
        self.Y = Y_train 
        # self.log_alpha_min, self.log_alpha_max = log_alpha_min, log_alpha_max
        self._sample_var = var_noise
        self._min_value = -1.
        self._minimize = minimize        
        self.X, self.Y = X_train, Y_train
        self.fold_size = int(self.X.shape[0] / n_folds)
        self.name = 'CNN'
        self.seed = seed
        tf.reset_default_graph()  # Safeguard just in case
        tf.set_random_seed(self.seed)  # Set seed for tensorflow
        np.random.seed(self.seed)      # Set seed for numpy
 
    def run_CNN(self, params, X_train, Y_train, X_test, Y_test, save_path=None, restore_path=None):
       # NOTE: params has len being 1
 
        # Reset or set seed
        tf.reset_default_graph()  # Safeguard just in case
        tf.set_random_seed(0)  # Set seed for tensorflow
        np.random.seed(0)      # Set seed for numpy
 
        # Define some fixed hyperparameters
        num_classes = 10
        num_epochs = 20
 
        # Extract hyperparameters from params:
        dropout_rate1 = params[0]
        dropout_rate2 = params[1]
        learning_rate = 10**((self.log_ub - self.log_lb)*params[2] + self.log_lb)
        batch_size = 128
 
        # Compute sample size of training and testing dataset
        train_sample_size = X_train.shape[0]
        test_sample_size = X_test.shape[0]
 
        # Define placeholder
        # Reshape to (batch, height, width, channel)
        tf_x = tf.placeholder(tf.float32, [None, 28*28])
        image = tf.reshape(tf_x, [-1, 28, 28, 1])
        tf_y = tf.placeholder(tf.int32, [None, 10])
 
        # First convo layer
        conv1 = tf.layers.conv2d(inputs=image,  # shape(28,28,1))
                                 filters=32,
                                 kernel_size=5,
                                 strides=1,
                                 padding='same',
                                 activation=tf.nn.relu,
                                 kernel_initializer=tf.contrib.layers.xavier_initializer(seed=1))  # -> shape(28,28,32))
        pool1 = tf.layers.max_pooling2d(conv1,
                                        pool_size=2,
                                        strides=2)  # -> shape(14, 14, 32)
        pool1 = tf.nn.dropout(pool1, dropout_rate1)
         
        # Second convo layer
        conv2 = tf.layers.conv2d(inputs=pool1,
                                 filters=64,
                                 kernel_size=5,
                                 strides=1,
                                 padding='same',
                                 activation=tf.nn.relu,
                                 kernel_initializer=tf.contrib.layers.xavier_initializer(seed=1))  # -> shape (14, 14, 64)
        pool2 = tf.layers.max_pooling2d(conv2,
                                        pool_size=2,
                                        strides=2)  # -> shape (7, 7, 64)
        pool2 = tf.nn.dropout(pool2, dropout_rate2)
 
        # Output
        flat = tf.reshape(pool2, [-1, 7*7*64])  # -> shape (7*7*64, )
        output = tf.layers.dense(flat, num_classes,
                                 kernel_initializer=tf.contrib.layers.xavier_initializer(seed=1))
 
        # Loss function and training operation
        loss = tf.losses.softmax_cross_entropy(onehot_labels=tf_y,
                                               logits=output)
        train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)
        accuracy = tf.metrics.accuracy(labels=tf.argmax(tf_y, axis=1),
                                       predictions=tf.argmax(output, axis=1),)[1]
 
        # sess = tf.Session()
        # init_op = tf.group(tf.global_variables_initializer(),
        #                    tf.local_variables_initializer())


        config = tf.ConfigProto()
        config.gpu_options.allow_growth=True
        sess = tf.Session(config=config)

        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        sess.run(init_op)

        try:
            saver = tf.train.Saver()
            saver.restore(sess, restore_path)
            print("Model restored from %s."%(restore_path))
            accuracy_test, flat_test = sess.run([accuracy, flat],
                                            {tf_x: X_test, tf_y: Y_test}) 
            return accuracy_test 
        
        except:
            print('No saved model in %s. Training a new one ...'%(restore_path))
            if save_path is not None:
                saver = tf.train.Saver()
            # Training model
            seed = 0
            for epoch in range(num_epochs):
    
                epoch_loss = 0
                num_batches = int(train_sample_size/batch_size)
                seed = seed + 1
                batches = random_mini_batches(X_train,
                                            Y_train,
                                            batch_size=batch_size,
                                            seed=seed)
                for step, batch in enumerate(batches):
    
                    # Select a minibatch
                    b_X, b_Y = batch
    
                    # Run the session to execute the optimizer and the cost
                    _, b_loss = sess.run([train_op, loss], {tf_x: b_X, tf_y: b_Y})
    
                    epoch_loss += b_loss/num_batches
    
    #                if step % 200 == 0:
    #                    accuracy_, flat_rep = sess.run([accuracy, flat],
    #                                                   {tf_x: X_test, tf_y: Y_test})
    #                    print('Step:', step, '| train loss: %.4f' % b_loss,
    #                          '| test accuracy: %.2f' % accuracy_)
    
                if epoch % 5 == 0:
                    accuracy_, flat_rep = sess.run([accuracy, flat],
                                                {tf_x: X_test, tf_y: Y_test})
                    print("Cost after epoch %i: %f | accuracy %f" % (epoch,
                                                                    epoch_loss,
                                                                    accuracy_))
            if save_path is not None:
                save_path_sess = saver.save(sess, save_path)
                print("Model saved in path: %s"%save_path_sess)
            accuracy_test, flat_test = sess.run([accuracy, flat],
                                                {tf_x: X_test, tf_y: Y_test})
    
            return accuracy_test
    def evaluate(self, x):
        params = x[:self._dim]
        k = int(np.argmax(x[self._dim:])) # w is a one-hot vector 

        fold_size = self.fold_size
        val_ind = range(k*fold_size,(k+1)*fold_size)
        train_ind = np.array([ i for i in range(self.X.shape[0]) if i not in val_ind ]).astype('int')
        X_val = self.X[val_ind,:]
        Y_val = self.Y[val_ind]
        
        X_train = self.X[train_ind, :]
        Y_train = self.Y[train_ind]

        val_acc = self.run_CNN(params, X_train, Y_train, X_val, Y_val) 
        if self.minimize:
            return 1 - val_acc 
        else:
            return val_acc 

    def evaluate_mt(self, x):
        params = x[:self._dim]
        k = int(x[self._dim:])

        fold_size = self.fold_size
        val_ind = range(k*fold_size,(k+1)*fold_size)
        train_ind = np.array([ i for i in range(self.X.shape[0]) if i not in val_ind ]).astype('int')
        X_val = self.X[val_ind,:]
        Y_val = self.Y[val_ind]
        
        X_train = self.X[train_ind, :]
        Y_train = self.Y[train_ind]

        val_acc = self.run_CNN(params, X_train, Y_train, X_val, Y_val) 
        if self.minimize:
            return 1 - val_acc 
        else:
            return val_acc 

def random_mini_batches(X, Y, batch_size=64, seed=0):
    """
    Creates a list of random minibatches from (X, Y)
    
    Arguments:
    X -- input data, of shape (number of examples, input size)
    Y -- true "label" vector of shape (number of examples, number of features)
    batch_size -- size of the mini-batches, integer
    seed -- this is only for the purpose of grading, so that you're "random
    minibatches are the same as ours.
    
    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """
    
    m = X.shape[0]                  # number of training examples
    mini_batches = []
    np.random.seed(seed)
    
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation, :]
    shuffled_Y = Y[permutation, :].reshape((m, Y.shape[1]))
    
    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_minibatches = math.floor(m/batch_size)
    for k in range(0, num_minibatches):
        mini_batch_X = shuffled_X[k*batch_size:k*batch_size + batch_size, :]
        mini_batch_Y = shuffled_Y[k*batch_size:k*batch_size + batch_size, :]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < batch_size)
    if m % batch_size != 0:
        mini_batch_X = shuffled_X[num_minibatches*batch_size:m, :]
        mini_batch_Y = shuffled_Y[num_minibatches*batch_size:m, :]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches

if __name__ == '__main__':
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets('./mnist', one_hot=True)
    X_train = mnist.train.images
    Y_train = mnist.train.labels
    X_test = mnist.test.images
    Y_test = mnist.test.labels

    cnn = CNN(X_train, Y_train, n_folds=10, minimize=False, var_noise=0, seed=1) 
    params = [0.1, 0.1, 0.1]
    acc = cnn.run_CNN(params, X_train, Y_train, X_test, Y_test)
    print(acc)

    p2 = [0.01, 0.01, 0.01, 1,0,0,0,0,0,0,0,0,0]
    val_acc = cnn.evaluate(p2)
    print(val_acc)