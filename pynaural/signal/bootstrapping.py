import numpy as np
from pynaural.signal.fitting import circular_linear_regression

################################ BOOTSTRAP #############################

from matplotlib.pyplot import *    
def sample_with_replacement(data, size = None, axis = -1):
    '''
    Takes a sample of the data of the same size by default.
    
    ``size'' : the size of the sample, defaults to the same size as the data
    ``axis'' : if is is specified then lines/columns are returned. 
    If it is 1 then columns are returned, if 0 lines are returned.
    '''
    if axis == -1:
        datasize =  data.shape[0] * data.shape[1]
        if size == None:
            size =  datasize
        indices = np.random.randint(datasize, size = size)
        return data.flatten()[indices]
    else:
        datasize = data.shape[axis]
        if size == None:
            size =  datasize
        indices = np.random.randint(datasize, size = size)
        if axis == 0:
            return data[indices, :]
        elif axis == 1:
            return data[:, indices]

def bootstrap(data, Niter = 1000, statistic = lambda x: np.mean(x), size = None, axis = -1):
    res = np.zeros(Niter, dtype = object)
    for i in xrange(int(Niter)):
        curdata = sample_with_replacement(data, size = size, axis = axis)
        res[i] = statistic(curdata)
    return res
        

def bootstrap_regression(xdata, ydata, Nbootstrap = 1000, Nresargs = 2):
    '''
    Performs multiple linear regressions on the data given in datax and datay
    Nbootstrap : number of distributions to be re drawn
    Nresargs :  number of result arguments from the linregress function to be returned, defaults to 2 (slope, intercept)
    
    returns a 2d array, with each row 
    '''
    
    statfun = lambda x: linregress(x[0,:], x[1,:])[:Nresargs]
    res = bootstrap(np.vstack((xdata, ydata)), statistic = statfun, axis = 1, 
                        Niter = Nbootstrap)
    out = np.zeros((Nbootstrap, Nresargs))
    for i in xrange(int(Nbootstrap)):
        out[i,:] = np.array([x for x in res[i]], dtype = float)
    return out
    

def bootstrap_cl_regression(xdata, ydata, Nbootstrap = 1000.,
                            slope_extent = None):
    '''
    Performs multiple circular linear regressions on the data given in datax and datay
    xdata is the linear data
    ydata is the circular data
    '''
    statfun = lambda x: circular_linear_regression(x[0,:], x[1,:], slope_extent = slope_extent)

    res = bootstrap(np.vstack((xdata, ydata)), statistic = statfun, axis = 1, 
                        Niter = Nbootstrap)

    out = np.zeros((Nbootstrap, 2))
    for i in xrange(int(Nbootstrap)):
        out[i,:] =np.array([x for x in res[i]], dtype = float).flatten()
        
    return out

