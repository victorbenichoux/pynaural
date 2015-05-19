import numpy as np
from pynaural.signal.misc import ola_filter
from pynaural.signal.impulseresponse import ImpulseResponse, dur2sample
from pynaural.signal.sounds import Sound

__all__ = ['deconvolve', 'swept_sine', 'excitation_signal', 'truncate_irs']

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

## To truncate hrirs

def truncate_HRIR(hrir_study, N, Threshold=-50.0):
    ir_resp_dB = 20*np.log10(abs(hrir_study))
    ind_max = np.argmax(ir_resp_dB)

    ir_resp_trunc = hrir_study
    ind_min = np.nonzero(ir_resp_dB[ind_max+N:] < np.max(ir_resp_dB)+Threshold)

    ir_resp_trunc[ind_max+N+ind_min[0][0]:] = 10**(Threshold/20.0)

    return ir_resp_trunc.flatten()

def truncate_irs(hrir, T, Threshold=-50.0):
    Nb_used=int(T*hrir.samplerate)
    Nsamples = max(np.shape(hrir)[0],Nb_used)
    Ncoords = int(0.5*np.shape(hrir)[1])
    samplerate = hrir.samplerate
    coords = hrir.coordinates
    data = np.zeros((Nsamples, Ncoords*2))

    for k in range(Ncoords):
        hrir_look = hrir.forcoordinates(coords[k][0], coords[k][1])

        data[:,k] = truncate_HRIR(hrir_look.left[:,0],Nb_used,Threshold)
        data[:,k+ Ncoords] = truncate_HRIR(hrir_look.right[:,0],Nb_used,Threshold)

    hrir_trunc = ImpulseResponse(data, samplerate = samplerate,
                      binaural = True,
                      coordinates = coords)

    return hrir_trunc
