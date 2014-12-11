import numpy as np
import scipy as sp
from pynaural.signal.misc import fftxcorr, rms, octaveband_filterbank
from pynaural.signal.sounds import Sound

from brian.hears import Gammatone, Repeat
import brian.hears.sounds as bh_sound
from brian.stdunits import Hz

__author__ = 'victor'

__all__ = ['broadband_coherence',
            'octaveband_coherence',
            'gammatone_coherence']


def broadband_coherence(hrir, samplerate, tcut = 1e-3):
    '''
    returns the coherence of hrir per band in gammatone filters
    '''
    hrir = hrir.squeeze()

    if not isinstance(hrir, Sound):
        hrir = Sound(hrir, samplerate = samplerate)

    left = hrir.left
    right = hrir.right
    xcorr = np.abs(fftxcorr(left, right))
    times = (np.arange(len(left)+len(right)-1)+1-len(left))/hrir.samplerate

    return np.max(xcorr[np.abs(times) < tcut])/(rms(left)*rms(right)*len(right))


def octaveband_coherence(hrir, samplerate, cfs, tcut = 1e-3, butter_order = 3, fraction = 1./3, return_envelope = False):
    '''
    Coherence computed in fraction-of-octave bands of frequency.

    :param hrir:
    :param samplerate:
    :param cf:
    :param tcut:
    :param butter_order:
    :param fraction:
    :return:
    '''
    if not isinstance(hrir, Sound):
        hrir = Sound(hrir, samplerate = samplerate)

    filtered_left = octaveband_filterbank(hrir[:,0], cfs, hrir.samplerate, fraction = fraction, butter_order = butter_order)
    filtered_right = octaveband_filterbank(hrir[:,1], cfs, hrir.samplerate, fraction = fraction, butter_order = butter_order)

    res = np.zeros(len(cfs))

    if return_envelope:
        res_env = np.zeros(len(cfs))

    for i in range(len(cfs)):
        left = filtered_left[:, i]
        right = filtered_right[:, i]
        times = (np.arange(len(left)+len(right)-1)+1-len(left))/hrir.samplerate
        xcorr = fftxcorr(left, right)
        res[i] = np.max(xcorr[np.abs(times) < tcut])/(rms(left)*rms(right)*len(right))

        if return_envelope:
            left_env = np.abs(sp.signal.hilbert(left))
            right_env = np.abs(sp.signal.hilbert(right))
            xcorr_env = fftxcorr(left_env, right_env)
            res_env[i] = np.max(xcorr_env[np.abs(times) < tcut])/(rms(left_env)*rms(right_env)*len(right_env))

    if return_envelope:
        return res, res_env
    else:
        return res


def gammatone_coherence(hrir, samplerate, cf, tcut = 1e-3):
    '''
    returns the coherence of hrir per band in gammatone filters
    '''
    if not isinstance(hrir, bh_sound.Sound):
        hrir = bh_sound.Sound(hrir, samplerate = samplerate*Hz)

    fb = Gammatone(Repeat(hrir, len(cf)), np.hstack((cf, cf)))
    filtered_hrirset = fb.process()
    res = np.zeros(len(cf))
    for i in range(len(cf)):
        left = filtered_hrirset[:, i]
        right = filtered_hrirset[:, i+len(cf)]
        times = (np.arange(len(left)+len(right)-1)+1-len(left))/hrir.samplerate
        xcorr = fftxcorr(left, right)
        res[i] = np.max(xcorr[np.abs(times) < tcut])/(rms(left)*rms(right)*len(right))
    return res
