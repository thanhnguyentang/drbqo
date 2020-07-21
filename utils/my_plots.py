
import numpy as np
from pylab import grid
import matplotlib.pyplot as plt
from pylab import savefig
import pylab

def plot_acquisition_1d(x, 
                        f_objs = None, # Objective function values evaluated at x. 
                        obsv = None, # A tuple of (x_obsv, y_obsv) 
                        pred = None, # A tuple of (mu, sigma) evaluated at x.  
                        f_acqs = None,  # Acquisition evaluated at x. 
                        next_point = None, # Next point to be acquired. 
                        x_bound =[0,1], 
                        margin = 0.01, 
                        optimum = None, 
                        legend_show=True, 
                        figsize=None, 
                        filename=None,
                        close_fig=True):
    # Plot approximation
    if figsize is None and f_acqs is None:
        figsize = (4,3)
    if figsize is None and f_acqs is not None:
        figsize = (12,3)
    fig=plt.figure(figsize=figsize)
    if f_acqs is not None:
        fig.subplots_adjust(wspace = 0.4)
        plt.subplot(1,2,1)
    if f_objs is not None:
        plt.plot(x.ravel(), f_objs.ravel(), label='Obj.')
    
    if pred is not None:
        m,s = pred 
        plt.fill_between(x.ravel(), m.ravel() + 1.96 * s.ravel(), m.ravel() - 1.96 * s.ravel(), alpha=0.3, label='95% Conf.')
        plt.plot(x.ravel(), m , label=r'$\mu(x)$')
    if next_point is not None:
        plt.axvline(x=next_point, ls='--', c= 'k', lw=1, label='Next')
    
    if obsv is not None:
        x_obsv, y_obsv = obsv
        plt.plot(x_obsv, y_obsv, 'kx', mew=2, label='Obsv.')
    
    if optimum is not None:
        plt.plot(optimum[0], optimum[1], 'bo', mew=2, label='Optimum')
        
    # plt.ylabel('f(x)', fontdict={'size': 12})
    plt.xlabel('x', fontdict={'size': 12})
    plt.xlim([x_bound[0] - margin, x_bound[1] + margin])
    if legend_show:
        plt.legend()
    
    # Plot acquisition function
    if f_acqs is not None:
        plt.subplot(1,2,2)
        plt.plot(x, f_acqs)
        if next_point is not None:
            plt.axvline(x=next_point, ls='--', c= 'k', lw=1, label='Max')
            # plt.plot(x_next, y_next, 'b+', mew=3, label='Maximum Ultility')
        plt.ylabel(r'$\alpha(x)$', fontdict={'size':15})
        plt.xlabel('x', fontdict={'size':15})
    
    plt.xlim(x_bound)
    # Save figure
    if filename is not None:
        fig.savefig(filename, dpi=300, bbox_inches='tight')
    if close_fig:
        fig.clf()


def plot_acquisition(bounds,input_dim,model,Xdata,Ydata,acquisition_function,suggested_sample, filename = None):
    '''
    Plots of the model and the acquisition function in 1D and 2D examples.
    '''

    # Plots in dimension 1
    if input_dim ==1:
        # X = np.arange(bounds[0][0], bounds[0][1], 0.001)
        # X = X.reshape(len(X),1)
        # acqu = acquisition_function(X)
        # acqu_normalized = (-acqu - min(-acqu))/(max(-acqu - min(-acqu))) # normalize acquisition
        # m, v = model.predict(X.reshape(len(X),1))
        # plt.ioff()
        # plt.figure(figsize=(10,5))
        # plt.subplot(2, 1, 1)
        # plt.plot(X, m, 'b-', label=u'Posterior mean',lw=2)
        # plt.fill(np.concatenate([X, X[::-1]]), \
        #         np.concatenate([m - 1.9600 * np.sqrt(v),
        #                     (m + 1.9600 * np.sqrt(v))[::-1]]), \
        #         alpha=.5, fc='b', ec='None', label='95% C. I.')
        # plt.plot(X, m-1.96*np.sqrt(v), 'b-', alpha = 0.5)
        # plt.plot(X, m+1.96*np.sqrt(v), 'b-', alpha=0.5)
        # plt.plot(Xdata, Ydata, 'r.', markersize=10, label=u'Observations')
        # plt.axvline(x=suggested_sample[len(suggested_sample)-1],color='r')
        # plt.title('Model and observations')
        # plt.ylabel('Y')
        # plt.xlabel('X')
        # plt.legend(loc='upper left')
        # plt.xlim(*bounds)
        # grid(True)
        # plt.subplot(2, 1, 2)
        # plt.axvline(x=suggested_sample[len(suggested_sample)-1],color='r')
        # plt.plot(X,acqu_normalized, 'r-',lw=2)
        # plt.xlabel('X')
        # plt.ylabel('Acquisition value')
        # plt.title('Acquisition function')
        # grid(True)
        # plt.xlim(*bounds)

        x_grid = np.arange(bounds[0][0], bounds[0][1], 0.001)
        x_grid = x_grid.reshape(len(x_grid),1)
        acqu = acquisition_function(x_grid)
        acqu_normalized = (-acqu - min(-acqu))/(max(-acqu - min(-acqu)))
        m, v = model.predict(x_grid)


        model.plot_density(bounds[0], alpha=.5)

        plt.plot(x_grid, m, 'k-',lw=1,alpha = 0.6)
        plt.plot(x_grid, m-1.96*np.sqrt(v), 'k-', alpha = 0.2)
        plt.plot(x_grid, m+1.96*np.sqrt(v), 'k-', alpha=0.2)

        plt.plot(Xdata, Ydata, 'r.', markersize=10)
        plt.axvline(x=suggested_sample[len(suggested_sample)-1],color='r')
        factor = max(m+1.96*np.sqrt(v))-min(m-1.96*np.sqrt(v))

        plt.plot(x_grid,0.2*factor*acqu_normalized-abs(min(m-1.96*np.sqrt(v)))-0.25*factor, 'r-',lw=2,label ='Acquisition (arbitrary units)')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.ylim(min(m-1.96*np.sqrt(v))-0.25*factor,  max(m+1.96*np.sqrt(v))+0.05*factor)
        plt.axvline(x=suggested_sample[len(suggested_sample)-1],color='r')
        plt.legend(loc='upper left')


        if filename!=None:
            savefig(filename)
        else:
            plt.show()

    if input_dim ==2:
        X1 = np.linspace(bounds[0][0], bounds[0][1], 200)
        X2 = np.linspace(bounds[1][0], bounds[1][1], 200)
        x1, x2 = np.meshgrid(X1, X2)
        X = np.hstack((x1.reshape(200*200,1),x2.reshape(200*200,1)))
        acqu = acquisition_function(X)
        acqu_normalized = (-acqu - min(-acqu))/(max(-acqu - min(-acqu)))
        acqu_normalized = acqu_normalized.reshape((200,200))
        m, v = model.predict(X)
        plt.figure(figsize=(15,5))
        plt.subplot(1, 3, 1)
        plt.contourf(X1, X2, m.reshape(200,200),100)
        plt.plot(Xdata[:,0], Xdata[:,1], 'r.', markersize=10, label=u'Observations')
        plt.colorbar()
        plt.xlabel('X1')
        plt.ylabel('X2')
        plt.title('Posterior mean')
        plt.axis((bounds[0][0],bounds[0][1],bounds[1][0],bounds[1][1]))
        ##
        plt.subplot(1, 3, 2)
        plt.plot(Xdata[:,0], Xdata[:,1], 'r.', markersize=10, label=u'Observations')
        plt.contourf(X1, X2, np.sqrt(v.reshape(200,200)),100)
        plt.colorbar()
        plt.xlabel('X1')
        plt.ylabel('X2')
        plt.title('Posterior sd.')
        plt.axis((bounds[0][0],bounds[0][1],bounds[1][0],bounds[1][1]))
        ##
        plt.subplot(1, 3, 3)
        plt.contourf(X1, X2, acqu_normalized,100)
        plt.colorbar()
        plt.plot(suggested_sample[:,0],suggested_sample[:,1],'k.', markersize=10)
        plt.xlabel('X1')
        plt.ylabel('X2')
        plt.title('Acquisition function')
        plt.axis((bounds[0][0],bounds[0][1],bounds[1][0],bounds[1][1]))
        if filename!=None:
            savefig(filename)
        else:
            plt.show()


def plot_convergence(Xdata,best_Y, filename = None):
    '''
    Plots to evaluate the convergence of standard Bayesian optimization algorithms
    '''
    n = Xdata.shape[0]
    aux = (Xdata[1:n,:]-Xdata[0:n-1,:])**2
    distances = np.sqrt(aux.sum(axis=1))

    ## Distances between consecutive x's
    plt.figure(figsize=(10,5))
    plt.subplot(1, 2, 1)
    plt.plot(list(range(n-1)), distances, '-ro')
    plt.xlabel('Iteration')
    plt.ylabel('d(x[n], x[n-1])')
    plt.title('Distance between consecutive x\'s')
    grid(True)

    # Estimated m(x) at the proposed sampling points
    plt.subplot(1, 2, 2)
    plt.plot(list(range(n)),best_Y,'-o')
    plt.title('Value of the best selected sample')
    plt.xlabel('Iteration')
    plt.ylabel('Best y')
    grid(True)

    if filename!=None:
        savefig(filename)
    else:
        plt.show()
