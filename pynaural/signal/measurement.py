import numpy as np
from pynaural.signal.misc import ola_filter
from pynaural.signal.impulseresponse import ImpulseResponse, dur2sample
from pynaural.signal.sounds import Sound

def deconvolve(signal, sweep):
    '''
    Deconvolves the second argument out of the first one.
    
    Usage:
    ir = deconvolve(signal, sweep)

                --------
    sweep ---> | system | ---> signal
                -------
                
    '''
    res = ola_filter(signal, sweep[::-1])

    m = (len(res)-1)/2
    res_trimmed = res[m:]

    return res_trimmed

def swept_sine(fmin, fmax, samplerate, N):
    '''
    Returns a swept sine between fmin and fmax, at samplerate samplerate.
    N gives the length of the swept sine.
    '''
    w1 = np.pi*2*fmin
    w2 = np.pi*2*fmax
    duration = 2**N/samplerate;
    K = duration*w1*np.log(w2/w1)

    L = duration/np.log(w2/w1)
    t = np.linspace(0, float(duration), 2**N)
    return Sound(np.sin(K*(np.exp(t/L)-1))*.999, samplerate = samplerate)

def excitation_signal(T, f1, f2, samplerate):
    '''
    Done according to marc's paper, 
    '''
    N = dur2sample(T, samplerate)
    times = np.linspace(0, float(T), N)
    
    return Sound(np.sin(2*np.pi * (f1 * T / (np.log(f2/f1))) * (np.exp(times/T*np.log(f2/f1))-1) - np.pi/2), samplerate = samplerate)

