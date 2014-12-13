from brian.hears import Gammatone, Repeat
import numpy as np
import scipy as sp
from pynaural.signal.misc import fftxcorr
from pynaural.signal.sounds import Sound

__author__ = 'victorbenichoux'

__all__ = ['octaveband_filterbank',
            'gammatone_filterbank',
            'gammatone_correlate']


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

def gammatone_filterbank(hrir, samplerate, cf, return_times = False):
    '''
    returns the correlograms of hrir per band
    '''
    hrir = hrir.squeeze()

    if not isinstance(hrir, Sound):
        hrir = Sound(hrir, samplerate = samplerate)

    fb = Gammatone(Repeat(hrir, len(cf)), np.hstack((cf, cf)))
    filtered_hrirset = fb.process()
    return filtered_hrirset

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