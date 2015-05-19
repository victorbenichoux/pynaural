from brian.hears import Gammatone, Repeat
import brian.hears.sounds as bhsounds
from brian.stdunits import Hz
import numpy as np
import scipy as sp
from pynaural.signal.misc import fftxcorr, rms
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
    try:
        nchannels = sound.shape[1]
    except:
        nchannels = 1
    res = np.zeros((sound.shape[0], len(cfs)*nchannels))

    for kcf, cf in enumerate(cfs):
        fU = 2**((fraction/2.))*cf
        fL = .5**((fraction/2.))*cf
        fU_norm = fU / (samplerate/2)
        fL_norm = fL / (samplerate/2)
        b, a = sp.signal.butter(butter_order, (fL_norm, fU_norm), 'band')
        tmp = sp.signal.lfilter(b,a,sound, axis = 0)
        for k in range(nchannels):
            res[:,kcf + k * len(cfs)] = tmp[:,k]

    return res

def gammatone_filterbank(sound_in, cf, samplerate, return_times = False):
    '''
    returns the correlograms of hrir per band
    '''
    #sound_in = sound_in.squeeze()

    if not isinstance(sound_in, bhsounds.Sound):
        sound_in = bhsounds.Sound(sound_in, samplerate = samplerate*Hz)

    fb = Gammatone(Repeat(sound_in, len(cf)), np.hstack((cf, cf)))
    filtered_hrirset = fb.process()
    return filtered_hrirset

def gammatone_correlate(sound_in, samplerate, cf, return_times = False, normalized = True):
    '''
    returns the correlograms of sound_in per band
    '''
    #sound_in = sound_in.squeeze()

    if not isinstance(sound_in, bhsounds.Sound):
        sound_in = bhsounds.Sound(sound_in, samplerate = samplerate*Hz)

    fb = Gammatone(Repeat(sound_in, len(cf)), np.hstack((cf, cf)))
    filtered_sound_inset = fb.process()

    res = np.zeros((sound_in.shape[0]*2-1, len(cf)))

    for i in range(len(cf)):
        left = filtered_sound_inset[:, i]
        right = filtered_sound_inset[:, i+len(cf)]
        res[:,i] = fftxcorr(left, right)/(rms(left)*rms(right)*len(right))

    if return_times:
        times = (np.arange(res.shape[0])+1-len(left))/sound_in.samplerate
        return times, res
    else:
        return res