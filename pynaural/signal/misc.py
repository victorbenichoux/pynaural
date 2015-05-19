from numpy.fft import *
from scipy.signal import *
import scipy as sp
import numpy as np

from pynaural.signal.sounds import Sound

__all__ = ['my_logspace', 'rms',
           'fftconvolve', 'fftxcorr',
           'ola_filter', 'zeropad',
           'nextpow2'
           ]

def my_logspace(fdown, fup, nfreqs):
    '''
    Returns nfreqs logarithmically distributed frequencies between fdown and fup
    '''
    upvalue = np.log(fup)/np.log(fdown)
    return np.logspace(1, upvalue, num = nfreqs, endpoint = True, base = fdown)

## FFT-based convolution and cross correlation

def fftconvolve(x, h):
    '''
    Uses FFT to convolve two 1D signals together
    '''
    x = x.flatten()
    h = h.flatten()
    Nx = len(x)
    Nh = len(h)
    Ntot = 2**np.ceil(np.log2(Nx+Nh-1))
    x = zeropad(x, Ntot)
    h = zeropad(h, Ntot)
    xft = fft(x)
    hft = fft(h)
    return ifft(xft*hft)[:Nx+Nh-1].real

def fftxcorr(x, h, axis = 0, normalized = True):
    '''
    Uses FFT to do a cross correlation.
    It is equivalent to the function correlate from sp except it uses FFTs (so it's faster).
    '''
    Nx = x.shape[axis]
    Nh = h.shape[axis]
    Ntot = 2**np.ceil(np.log2(Nx+Nh-1))
    x = zeropad(x, Ntot, axis = axis)
    h = zeropad(h, Ntot, axis = axis)
    xft = fft(x, axis = axis)
    hft = fft(h, axis = axis)
    res = fftshift(ifft(xft*np.conj(hft), axis = axis), axes = axis)
    mid = len(res)/2

    norm_factor = rms(x, axis = axis) * rms(h, axis = axis)
    res /= norm_factor * res.shape[axis]

    if axis == 0:
        return res[mid - max(Nx, Nh) + 1:mid + max(Nx, Nh),:].real
    else:
        return res[:,mid - max(Nx, Nh) + 1:mid + max(Nx, Nh)].real

## Overlap-and-add filtering

def ola_filter(x, h):
    '''
    Overlap add method for linear convolution.
    Usually x must be longer than h (or the same length).
    '''
    if not x.shape[0] >= h.shape[0]:
        raise ValueError('x should be bigger of equal than ir')
    if x.ndim > 1 and not (x.shape[1] == h.shape[1]):
        raise ValueError('should have the same shape')
    if x.ndim > 1:
        raise ValueError('Not supported for now')

    L = nextpow2(h.shape[0])
    res = np.zeros(x.shape[0] + h.shape[0] - 1)

    for i in range(int(np.ceil(x.shape[0]/L))):
        down = int(i * L)
        up = int(min((i+1) * L, x.shape[0]))
        chunk = fftconvolve(x[down:up], h)
        res[down:up+h.shape[0]-1] += chunk
    return res

## Decibels

def dBconv(x):
    '''
    Returns the dB value of the array (i.e. 20*log10(x))
    '''
    return 20*np.log10(x)

def dB_SPL(x):
    '''
    Returns the dB SPL value of the array, assuming it is in Pascals
    '''
    return 20.0*np.log10(rms(x)/2e-5)


def rms(x, axis = 0):
    '''
    Returns the RMS value of the array given as argument
    '''
    return np.sqrt(np.mean(np.asarray(x)**2, axis = axis))
# old but bad
#    return np.sqrt(np.mean((np.asarray(x)-np.mean(np.asarray(x), axis = axis))**2, axis = axis))



## padding utils functions for FFT

def zeropad(x, n, axis = 0):
    '''
    Zero pads the given array so that it ends up with the given length
    '''
    if len(x.shape) == 1:
        return np.hstack((x,np.zeros(n-len(x))))
    else:
        if axis == 0:
            return np.vstack((x, np.zeros((n-x.shape[0], x.shape[1]))))
        else:
            return np.hstack((x, np.zeros((x.shape[0], n-x.shape[1]))))

def nextpow2(n):
    '''
    Returns the next pwer of 2 after n
    '''
    return 2**np.ceil(np.log2(n))

## Band pas noise generation

def bandpass_noise(duration, fc,
                   fraction = 1./2, stopband_fraction = 1.,
                   samplerate = 44100.,
                   gstop = 10., gpass = 0.01):
    '''
    Returns a Sound which is a bp filtered version of white noise with fc and fraction filter, using butterworth filters.
    '''
    # TODO: move to Sound.bandpass_noise
    noise = whitenoise(duration, samplerate = samplerate)

    fhigh, flow = 2**((fraction/2.))*fc, .5**((fraction/2.))*fc
    fhigh_stop, flow_stop = 2**(((fraction+stopband_fraction)/2.))*fc, .5**(((fraction+stopband_fraction)/2.))*fc

    fnyquist = noise.samplerate/2.

    N, wn = sp.signal.buttord(ws = [flow/fnyquist,fhigh/fnyquist],
                              wp = [flow_stop/fnyquist,fhigh_stop/fnyquist],
                              gpass=gpass,
                              gstop=gstop)

    b, a = sp.signal.butter(N, wn, btype = 'bandpass')
    sf = Sound(sp.signal.lfilter(b, a, noise.flatten()),
               samplerate = noise.samplerate)
    return sf

# gammatone filtering + cross correlation


