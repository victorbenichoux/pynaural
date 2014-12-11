from numpy.fft import *

from brian.hears import Gammatone, Repeat
from scipy.signal import *
import scipy as sp

from pynaural.signal.sounds import Sound

__all__ = ['my_logspace',
           'octaveband_filterbank',
           'octaveband_smoothing', 'rms',
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

## smoothing of spectral data in bands of constant octave width.

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

def fftxcorr(x, h):
    '''
    Uses FFT to do a cross correlation.
    It is equivalent to the function correlate from sp except it uses FFTs (so it's faster).
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
    res = fftshift(ifft(xft*np.conj(hft)))
    mid = len(res)/2


    return res[mid - max(Nx, Nh) + 1:mid + max(Nx, Nh)].real

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

def zeropad(x, n):
    '''
    Zero pads the given array so that it ends up with the given length
    '''
    return np.hstack((x,np.zeros(n-len(x))))

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

def gammatone_correlate(hrir, samplerate, cf, return_times = False):
    '''
    returns the correlograms of hrir per band
    '''
    hrir = hrir.squeeze()

    if not isinstance(hrir, Sound):
        hrir = Sound(hrir, samplerate = samplerate)

    fb = Gammatone(Repeat(hrir, len(cf)), np.hstack((cf, cf)))
    filtered_hrirset = fb.process()
    res = np.zeros((hrir.shape[0]*2-1, len(cf)))
    for i in range(len(cf)):
        left = filtered_hrirset[:, i]
        right = filtered_hrirset[:, i+len(cf)]
        res[:,i] = fftxcorr(left, right)
    if return_times:
        times = (np.arange(len(left)+len(right))+1-len(left))/hrir.samplerate
        return times, res
    else:
        return res


def octaveband_filterbank(sound, cfs, samplerate, fraction = 1./3, butter_order = 3):
    '''
    passes the input sound through a bank of butterworth filters with fractional octave bandwidths
    :param sound:
    :param cfs:
    :param samplerate:
    :param fraction:
    :param butter_order:
    :return:
    '''
    res = np.zeros((len(sound), len(cfs)))

    for kcf, cf in enumerate(cfs):
        fU = 2**((fraction/2.))*cf
        fL = .5**((fraction/2.))*cf
        fU_norm = fU / (samplerate/2)
        fL_norm = fL / (samplerate/2)
        b, a = sp.signal.butter(butter_order, (fL_norm, fU_norm), 'band')
        res[:,kcf] = sp.signal.lfilter(b,a,sound.flatten())

    return res
from matplotlib.pyplot import *
