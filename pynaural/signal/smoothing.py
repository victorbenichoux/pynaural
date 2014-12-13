import numpy as np
from scipy.signal import *
from numpy.fft import *
from misc import *


######################## Spectral smoothing ########################
# both uniform & non uniform (in frequency) are implemented

def uniform_smoothing(tfs, method = 'rectangular', ntap = 5, end = 'zeros'):
    ''' 
    Performs uniform smoothing on a power spectrum.

    Inspired from:
    http://terpconnect.umd.edu/~toh/spectrum/Smoothing.html

    
    Keywords:
    - ``ends`` if it is 'zeros' then the end and beginning are tapped with zeros
    if it is 'progressive' then the tap width is progressively reduced so that the edges are not padded
    - ``ntap`` the number of taps
    - ``method`` support the following windowing: rectangular, triangular, pseudogaussian. Rectangular is staightforward, triangular is equivalent to two runs of rectangular, pseudo gaussian 3 runs of rectangular.
    
    Notes:
    - Works with nD arrays, arranged in (time/fqcy index, channel). independantly smoothes all channels
    
    '''
#    if not isinstance(tfs, TransferFunction):
#        tfs = TransferFunction(tfs)
#    tfs = asarray(tfs)
#    print tfs.shape
#    tfs = tfs.reshape((tfs.shape[0], tfs.shape[1]))
    
    if method == 'triangular':
        # it's equivalent to two rectangular smoothings
        return uniform_smoothing(uniform_smoothing(tfs, ntap = ntap, end = end), 
                                 ntap = ntap, end = end)
    elif method == 'pseudogaussian':
        # it's equivalent to 3 rectangular smoothings
        return uniform_smoothing(uniform_smoothing(tfs, ntap = ntap, end = end, 
                                                   method = 'triangular'),
                                 ntap = ntap, end = end)
    elif method == 'rectangular':
        if tfs.ndim == 1:
            tfs.shape = (len(tfs), 1)
        res = np.zeros(tfs.shape, dtype = complex)
        sumpoints = np.sum(tfs[:ntap, :], axis = 0)
        halfw = np.round(ntap/2)
        L = tfs.shape[0]
        for k in range(0, L-ntap):
            res[k+halfw, :] = sumpoints
            sumpoints = sumpoints - tfs[k]
            sumpoints = sumpoints + tfs[k+ntap]
        res[k+halfw,:] = sum(tfs[L-ntap:L,:], axis = 0)
        res /= ntap
        if end == 'progressive':
            # in this case we progressively reduce the number of taps at the edges.
            # hence we overwrite over the zeros at the beginning and end
            startpoint = (ntap+1)/2
            res[0, :] = tfs[0, :] + tfs[1, :]
            res[0, :] /= 2
            for k in range(1, startpoint):
                res[k] = mean(tfs[:2*k])
                res[-(k+1)] = mean(tfs[-(2*k+1):])
            res[-1, :] = tfs[-1, :] + tfs[-2, :]
            res[-1, :] /=2
        elif end == 'original':
            res[0,:] = tfs[0,:]
            res[-1,:] = tfs[-1,:]
        
        return res
    else:
        raise ValueError('I Don\' know the '+str(method)+' method')

### tools for window width evaluation (nonuniform smoothing)
def octaveband(f, fraction = 1./3):
    # pd in the paper
    fU = 2**((fraction/2.))*f
    fL = .5**((fraction/2.))*f
    return fU-fL

def windowwidth(k, N, fbin, octavefraction = 1./3):
    res = np.floor(octaveband(fbin*k, fraction = octavefraction)/(2*fbin))
    return res

### filtering windows    
def twocoefs_window(m, k , b = .54):
    '''
    A two-coeficient window of width m at point k. A kwdarg (b) can be set to set the value of the coeficients. 
    The formula is the following:
    W(k) = (b-(b-1)*cos(pi/m*k))/(2*b*(m+1)-1)
    Output will always be normalized.
    '''
    res = (b-(b-1)*cos(pi/m*k))/(2*b*(m+1)-1)
    norm = np.sum(res)
    if not norm == 1:
        # sometimes we need to re-normalize, why?
        res /= norm
    return res

def hannwindow(m, k):
    '''
    A Hann window of width m at point k
    '''
    return twocoefs_window(m, k, b = .5)

def hammingwindow(m, k):
    '''
    A Hamming window of width m at point k
    '''
    return twocoefs_window(m, k, b = .54)

def rectangularwindow(m, k):
    '''
    A rectangular window of width m at point k
    '''
    return twocoefs_window(m, k, b = 1)

### nonuniform spectral smoothing algorithm

def octaveband_smoothing(data, width = 1./3):
    '''
    Octave band average with a moving window.
    Data is assumed to be in columns (i.e. data.shape[0] is the number of samples)
    '''
    if len(data.shape) == 1:
        data.shape = (data.shape[0], 1)
    res = np.zeros_like(data)
    
    for k in xrange(1, data.shape[0]):
        low = int(round(k/2.0**(width)))
        high= int(min(round(k*2.0**(width)),data.shape[0]))
        res[k] = np.mean(data[low:high, :], axis = 0)

    return res

    
def nonuniform_spectralsmoothing(data, samplerate, b = 1.,
                                 width = 1./3, datascale = None):
    '''
    Implements (power) spectrum smooothing. The considered window is 
    W(k) = b - (1-b) cos (2*pi*k/N)
    The b parameter is a keyword argument. B = 1 corresponds to a rectangular window. b = .5 is the Hamming window, b = .54 the Hann window.
    
    NB: Smoothing can be either uniform in frequency or frequency dependant. If ``width`` is an integer then it represents the number of taps. If it is a float (<1) then the smoothing is frequency dependant and it represents the fraction of octave that one wants to use. So beware of the type of the number you pass to ``width``.
    
    Notes:
    - This is inspired by HATZIANTONIOU and MOURJOUPOULOS, and thus should work with complex TFs. But it doesn't currently.
    - It only works on 1D arrays
    - Doesn't support the cool progressive ends feature of the direct uniform_spectralsmoothing function
    '''
    N = data.shape[0]
    fbin = float(samplerate) / N
    res = zeros(data.shape, dtype = complex)

    if not (datascale is None):
        with_datascale = True
    else: 
        with_datascale = False
    
    # parsing the input arguments
    if type(width) == int:
        uniform = True
    elif type(width) == float:
        uniform = False

    for k in range(1, N/2):
        # evalutation of the window width
        if uniform:
            mk = width
        else:
            mk = windowwidth(k, N, fbin, octavefraction = width)+1
        # respective indices
        up = min(k+mk+1, N/2)
        down = max(k-mk, 0)

        w = twocoefs_window(mk, arange(down-k, up-k), b = b)            
        
        res[k] += np.sum(data[down:up]*w)
        if np.isnan(res[k]):
            print 'smoothing yields nans?'
            print 'mk', mk
            print 'k',k
            print 'up, down', (up, down)
            print 'w', w

    # then we complete the negative freq values with the complex conjugates
    res[0] = data[0]
    rest = np.conj(res[:N/2])
    res[N/2:] = rest[::-1]
    return res
    
def nonuniform_smoothing(data, samplerate = 44100., width = 1./3, datascale = None, b = .54):
    '''
    Performs non uniform smoothing on value points given in data at frequency positions given in the datascale kwdargs.
    For now only support rectangular window.
    '''
    if hasattr(data, 'samplerate'):
        samplerate = data.samplerate
        
    if not datascale is None and (not data.shape[0] == len(datascale)):
        raise ValueError
    
    if datascale is None:
        datascale = fftfreq(data.shape[0])*samplerate

    N = len(datascale)
        
    res = zeros(data.shape, dtype = complex)
    
    # parsing the input arguments
    if type(width) == int:
        uniform = True

    elif type(width) == float:
        uniform = False

    for k,cf in enumerate(datascale):
        # evalutation of the window width
        if uniform:
            mk = width
        else:
            band = octaveband(cf, fraction = width)
            fup = cf+band/2
            fdown = cf-band/2
            # indice up and down
            indices = np.nonzero((datascale<=fup) * (datascale >=fdown))[0]
            if len(indices) == 0:
                w = 0
            else:
                up = indices[-1]
                down = indices[0]
                #        semilogx(k*fbin, up, ' +')
                #        semilogx(k*fbin, down, 'o')
                mk = up-down
                w = twocoefs_window(mk, arange(-mk/2, mk/2), b = b)            
        res[k] = np.sum(data[down:up]*w)
    return res


def complexsmoothing(x, samplerate, width = 1./3, b = .54, onset_safeguard = 0):
    '''
    Implements complex smoothing on impulse responses.
    Takes an impulse response as an input and outputs a smoothed transfer function and the onset delay in samples
    I could probably implement yielding a new ir with the good length
    '''
    N = x.shape[0]
    # first step: remove linear part of the phase
    # estimation of the impulse response lag
    onset = onset_detection(x) - onset_safeguard
    # we remove it from the ir

    xtrim = x[onset:]
    Nxtrim = len(xtrim)
    # hack, its better if the output is even
    if Nxtrim % 2 == 1:
        onset -= 1
        xtrim = x[onset:]
        Nxtrim = len(xtrim)
    
    # second step: we compute the smoothed amplitudes
    xtrim_ft = fft(xtrim.flatten())
    amps = nonuniform_spectralsmoothing(np.abs(xtrim_ft), samplerate, width = width, b = b)
    
#    semilogx(fftfreq(Nxtrim)*samplerate, amps, label = 'amps')
    # third step: we compute the phases
    # new phases computations
    phases = nonuniform_spectralsmoothing(xtrim_ft/np.abs(xtrim_ft), samplerate, width = width, b = b)
    # and set all the amplitudes to 
    phasepart = np.exp(1j*np.angle(phases))

    # fourth step: the result
    res_ft = amps*phasepart
#    print res_ft
#    res = np.hstack((np.zeros(onset), ifft(res_ft)))
#    
    # hack: replace nans by original values
    nan_idx = np.isnan(res_ft)
#    print nan_idx
    res_ft[nan_idx] = xtrim_ft[nan_idx]

#    freqs = fftfreq(len(res_ft))*samplerate
#    semilogx(freqs, np.abs(res_ft), label = 'smoothed')
#    semilogx(freqs[nan_idx], np.zeros(len(np.nonzero(nan_idx)[0])), '*', label = 'nans')
    # fifth step: back in the time domain
#    res = np.hstack((np.zeros(onset), ifft(res_ft).real))
    res_ft = complete_toneg(res_ft)
    res = ifft(res_ft).real
    return res, onset


def newcomplexsmoothing(x, samplerate, width = 1./3, b = .54, onset_safeguard = 0):
    '''
    slightly different version where the linear delay is removed from the phase. 
    onset_safeguard kwdargs is IGNORED
    still going back to the time domain doesn't work
    '''
    N = x.shape[0]
    # first step: remove linear part of the phase
    # estimation of the impulse response lag
    onset = onset_detection(x)

    x_ft = fft(x.flatten())
    x_phases = np.angle(x_ft)
    linearpart = fftfreq(len(x_ft))*onset
    x_newphases = exp(1j*(x_phases-linearpart))

    # second step: we compute the smoothed amplitudes
    amps = nonuniform_spectralsmoothing(np.abs(x_ft), samplerate, width = width, b = b)
    # third step: we compute the phases
    # new phases computations
    phases = nonuniform_spectralsmoothing(x_newphases, samplerate, width = width, b = b)
    # and set all the amplitudes to 
    phasepart = np.exp(1j*phases)
    # fourth step: the result
    res_ft = amps*phasepart

    res_ft = complete_toneg(res_ft)
    res = ifft(res_ft).real
    return res, onset

################### Onset Detection ##################

def cummax(x):
    '''
    Calculates the maximas of a sequence up to all indexes. Basically like
    for i in range(len(x)): res[i] = max(x[:i])
    but way faster.
    '''
    tmp = np.tile(asarray(x).reshape(1, len(x)), (len(x), 1))
    tmp *= np.tril(ones(tmp.shape))
    return np.max(tmp, axis = 1)

def onset_detection(x, threshold = .01):
    '''
    Detects the onset of an impulse response.
    '''
    data = x.copy()
    data /= np.max(abs(data))
    data_crit = np.diff(cummax(abs(data)))
    return np.nonzero(data_crit >= threshold)[0][0]


######################## IR windowing #################
def apply_windowing(x, t = None, bandwidth = None):
    '''
    Windows the ImpulseResponse given in input. The first samples are set to 1 and the trailing ones to 0. In the middle (transition band), they are windowed with a decreasing cosine window of width bandwidth and centered around t.
    '''
    if bandwidth is None:
        bandwidth = len(x)

    if t is None:
        t = len(x)/2

    trans_window = (np.cos(arange(bandwidth, dtype = float)/bandwidth * np.pi)+1)/2
    
    n = len(x)
    
    nbefore = t-bandwidth/2
    nafter = n-(t+bandwidth/2)

    window = np.hstack((ones(nbefore), trans_window, zeros(nafter)))
    window.shape = (len(window),1)
    window_tot = np.tile(window, (1,x.shape[1]))
    return x*window


