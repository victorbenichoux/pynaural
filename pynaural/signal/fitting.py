'''
Some fitting routines that can be useful
'''
from scipy.optimize import leastsq, fmin
from scipy.stats import linregress
import numpy as np


################################ FITTING #############################
# contains routines for circular linear regression

# Unwrapping based method        
def unwrap_and_fit(frequencies, 
                   phases, 
                   plow = -np.pi, phigh = np.pi):
    """
    Unwraps a phase value (in radians, between -pi and pi) and perform a linear regression
    """
    scale = lambda x, pl, ph: (ph - plow)/(phigh-plow)
    phases_rads = scale(phases, plow, phigh)
    phases_rads *= 2*np.pi

    phases_rads = scale(phases, plow, phigh)
    unwrapped = np.unwrap(phases)
    p = linregress(frequencies, unwrapped)
    slope = p[0]
    intercept = p[1]
    return slope, intercept
    

# Phase space method
def circular_slope_bruteforce(frequencies, phases, 
                              extent = None,
                              weights = None):
    
    vals = np.zeros(len(extent))

    if weights == None:
        def vs(A, xs, phis):
            return np.abs(np.sum(np.exp(1j*(phis - A *xs))) )**2
    else:
        def vs(A, xs, phis):
            return np.abs(np.sum(np.exp(1j*(phis - A *xs))*weights) )**2

    for i, x in enumerate(extent):
        vals[i] = vs(x, frequencies, phases)
        
    idmax = np.argmax(vals)
    if False:
        plot(extent, vals)
        plot(extent[idmax], vals[idmax])
        show()
    return extent[idmax]

def guess_init(freqs, x, method = 'simple'):
    '''
    provides a reasonable guess of the slope I think
    '''
    if method == 'linreg':
        s, i, _,_,_, = linregress(freqs, x)
        return s
    else:
        allslopes = np.diff(x)/np.diff(freqs)
        return np.median(allslopes)


def circular_slope_maximization(frequencies, phases, init = 0., weights = None):
    '''
    given a good guess of the max, checks for a best max with scipy's fmin function
    '''
    if weights == None:
       def vsprime(A, xs, phis):
           return -np.abs(np.sum(np.exp(1j*(phis - A *xs))) )**2
           
    else:
        def vsprime(A, xs, phis):
            return -np.abs(np.sum(np.exp(1j*(phis - A *xs))*weights ) )**2

    m = fmin(vsprime, init, args = (frequencies, phases), disp = False)

    return m

def intercept_maximization(frequencies, phases, slope, init = 0., weights = None):
    '''
    First performs a combination of the two above functions to find the slope, and then finds the intercept by maximizing a function
    '''
    if weights == None:
        q = lambda phi0, x, y: -np.sum(np.cos(y - slope * x - phi0))
    else:
        q = lambda phi0, x, y: -np.sum(np.cos(y - slope * x - phi0)*weights)

    m = fmin(q, init, args = (frequencies, phases), disp = False)
    return m
    
def puredelay_fit(frequencies, phases, init = 0., weights = None):
    '''
    finds the best pure delay fit, it's a *linear* (as in not affine) regression
    '''
    if weights == None:
        q = lambda phi0, x, y: -np.sum(np.cos(y - phi0*x))
    else:
        q = lambda phi0, x, y: -np.sum(np.cos(y - phi0*x)*weights)

    m = fmin(q, init, args = (frequencies, phases), disp = False)
    return m

def circular_linear_regression(frequencies, phases,
                               slope_extent = None, Npoints_guess = 100,
                               slopeguess = None,
                               verbose = False, weights = None):
    '''
    Uses the method described in the paper Frequency invariant repr of ITDs (plos comp)
     
    first finds the slope and then the intercept
    '''
    # first guess the vicinity of the best CD
    if slope_extent == None:
        slope_extent = np.arange(-.002, .002, Npoints_guess)
        
    if slopeguess is None:
        slopeguess = circular_slope_bruteforce(frequencies, phases, extent = slope_extent, 
                                               weights = weights)

    # refine the slope estimation
    slope = circular_slope_maximization(frequencies, phases, init = slopeguess,
                                        weights = weights)
     
    # finally find the best phase
    intercept = intercept_maximization(frequencies, phases, slope, init = 0.,
                                       weights = weights)
    
    if verbose:
        print 'First slope guess: ', slopeguess
        print 'Final slope value: ', slope
        print 'Intercept value: ', slope
     
    return slope, intercept

def weighed_linear_regression(x, y, weights = None, init = None):
    '''
    Performs a linear regression of y vs x, possibly weighed by *weights*.
    Returns (slope, intercept)
    '''
    if weights == None:
        weights = np.ones(len(x))
    if init == None:
        init = np.zeros(2)
    
    fitfunc = lambda p, x: p[0] + x * p[1]
    errorfunc = lambda p, x, y, z: np.abs(fitfunc(p, x) - y) * z
    p1, success = leastsq(errorfunc, init, args = (x, y, weights))
    return p1[1], p1[0]


