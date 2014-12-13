from scipy import weave
from scipy.weave import converters
from pynaural.utils.debugtools import log_debug
from pynaural.signal.smoothing import *
from pynaural.signal.impulseresponse import *
from pynaural.signal.filterbanks import *
import pynaural.signal.fitting as fit
from brian.hears import Gammatone, Repeat


######################################################
################## ITD computation  ##################
######################################################

## Individual ITD functions
# XCORR
# Phase
# fit: itdp, itdg, idi

# The following section deals with ITD estimation, the main wrapper function and the individual 
def itdp_itdg_idi(ipds, cfs, samplerate, 
                  only_posfreqs = False,  Q = 1., slopeguess = False, 
                  itdg_extent = np.linspace(-.001, 0.001, 100), 
                  fcut_lf = 400., cut_lf = False):
    '''
    Returns the phase ITD, ITDp, the group ITD, ITDg and the IDI around the given center frequencies, and from the given ipd pattern.

    Units:
    ITDg: is in rad by Hz, i.e must be divided by 2 pi to get a value in second
    ITDp: is in rad by Hz.
    IDI: is in rad

    * Arguments
    
    ITDg and IDI are computed by a circular linear regression around the cf points. 
    cfs is the array of center frequencies.
    The bandwidth are controlled via the 'Q' kwdarg.

    the cut_lf and fcut_lf kwdargs linearize the ipd in below the fcut lf frequency. As a result the ipd is smoothed in lf and estimation is more consistent

    ipds is 1D array of ipds, at the usual fftfreq-like positions.
    Note: if only_posfreqs is True then it is assumed that ipds contains the ipd spectrum for positive frequencies only
    '''
    itdp, itdg, idi = np.zeros(len(cfs)), np.zeros(len(cfs)), np.zeros(len(cfs))

    if not only_posfreqs:
        freqs = fftfreq(len(ipds)) * samplerate
        posfreqs = freqs[freqs>0]
        posipds =  ipds[freqs>0]
    else:
        freqs = np.linspace(0, samplerate/2., len(ipds))
        posfreqs = freqs
        posipds = ipds

    if cut_lf:
        lfcut_freqs = (posfreqs < fcut_lf)
        lfcut_ids = np.nonzero(lfcut_freqs)[0]
        new_posipds = posipds.copy()
        new_posipds[lfcut_ids] = np.linspace(0., posipds[lfcut_ids[-1]+1], len(lfcut_ids))
        posipds = new_posipds
        
    for k, cf in enumerate(cfs):
        fhigh, flow = 2**(1./(2*Q))*cf, .5**(1/(2*Q))*cf
        curfreq_ids = (posfreqs> flow)*(posfreqs < fhigh)
        curfreqs = posfreqs[curfreq_ids]
        curipds = posipds[curfreq_ids]
        if not (slopeguess is False):
            itdg[k], idi[k] = fit.circular_linear_regression(curfreqs, 
                                                             curipds, 
                                                             slope_extent = itdg_extent)
        else:
            if slopeguess is True:
                guess = fit.guess_init(curfreqs, curipds)
            elif slopeguess == 'puredelay':
                guess = fit.puredelay_fit(curfreqs, curipds, init = 0)
            else:
                guess = False
            itdg[k], idi[k] = fit.circular_linear_regression(curfreqs, 
                                                             curipds, 
                                                             slopeguess = guess)
    itdp = itdg + idi/cfs
    return itdp, itdg, idi

######## XCORR ITD

def itd_xcorr(hrir,cfs = None):
    '''
    Computes ITDs by looking at the cross correlation between channels
    '''
    itds = np.zeros((len(cfs), hrir.shape[1]/2))
    for k in range(hrir.ncoordinates):
        curir = hrir.forcoordinates(k)
        C, times = gammatone_correlate(curir, hrir.samplerate, cfs)
        for i in range(len(cfs)):
            itds[i, k] = times[np.argmax(C[:, i])]
    return itds

######## Phase ITD

def itd_phase(hrir):
    '''
    Returns the phase ITD computed as the unwrapped IPD divided by the pulsation.
    '''
    freqs =  fftfreq(hrir.nsamples)*hrir.samplerate
    return ipd(hrir, unwrap = True)/(2*np.pi*freqs)

######## Onset ITD

def itd_onset(hrir, cf, threshold = .15):
    '''
    Computing ITD in bands by just looking at when the IR first reaches ``threshold`` fraction of its maximum. It uses the onset_time method of the ImpulseResponse object.
    
    In Duda & Mertens
    '''
    if hrir.ncoordinates > 1:
        for k in range(hrir.ncoordinates):
            if k==0:
                res = itd_simple_bare(hrir.forcoordinate(k), cf, threshold = threshold)
            else:
                res = vstack((res,itd_simple_bare(hrir.forcoordinate(k), cf, threshold = threshold)))
        return res.T
    else:
        return itd_simple_bare(hrir, cf, threshold = threshold)    

def itd_onset_bare(hrir, cf, threshold = .15):
    if not isinstance(hrir, Sound):
        hrir = Sound(hrir, samplerate = hrir.samplerate)

    fb = Gammatone(Repeat(hrir, len(cf)), hstack((cf, cf)))
    filtered_hrirset = fb.process()

    itds = zeros_like(cf)
    
    for i in range(len(cf)):
        left = ImpulseResponse(filtered_hrirset[:, i], hrir.samplerate)
        right = ImpulseResponse(filtered_hrirset[:, i+len(cf)], hrir.samplerate)
        delay_left = left.onset_time(threshold)
        delay_right = right.onset_time(threshold)
        itds[i] = (delay_left-delay_right)
    return itds



################# IPD and phase unwrapping
def ipd(hrir, unwrap = False, threshold = np.pi, positivefreqs = False):
    '''
    Computes the phase difference of a binaural ImpulseResponse object.
    '''
    hrir_ft = fft(asarray(hrir), axis = 0)
    middle = hrir.shape[1]/2
    ipds =  np.angle(hrir_ft[:, :middle]) - np.angle(hrir_ft[:, middle:])
    if unwrap:
        ipds = np.unwrap(ipds, axis = 0, discont = threshold)
    return ipds
    

######################## ILD ########################
def ild(hrir, cf, cpu = 1):
    '''
    Computes the ITDs and ILDs for a given binaural impulse response.
    
    Returns the ILD in the same format as an IR object, i.e. an nd array [cf, position]
    
    '''
    
    if hrir.ncoordinates > 1:
        res = np.zeros((len(cf), hrir.ncoordinates))
        for k in range(hrir.ncoordinates):
            res[:,k] = ild_bare(hrir.forcoordinates(k), cf)
        return res
    else:
        return ild_bare(hrir, cf)
        
def ild_bare(hrir, cf, **kwdargs):
    '''
    ILD computation routine. called by ild that handles multiprocessing,...
    '''
    samplerate = hrir.samplerate

    # perform some checks and special cases
    if (hrir[:,0] == hrir[:,1]).all():
        return np.zeros(len(cf))

    if (abs(hrir[:,0])<= 10e-6).all() or  (abs(hrir[:,1])<=10e-6).all():
        db.log_debug('Blank hrirs detected, output will be weird')

    if not isinstance(hrir, Sound):
        hrir = Sound(hrir, samplerate = samplerate)

    fb = Gammatone(Repeat(hrir, len(cf)), hstack((cf, cf)))
    filtered_hrirset = fb.process()
    
    ilds = []
    for i in range(len(cf)):
        left = filtered_hrirset[:, i]
        right = filtered_hrirset[:, i+len(cf)]
        # This FFT stuff does a correlate(left, right, 'full')
        Lf = fft(hstack((left, zeros(len(left)))))
        Rf = fft(hstack((right[::-1], zeros(len(right)))))
        C = ifft(Lf*Rf).real
        ilds.append(sqrt(amax(C)/sum(right**2)))
    ilds = array(ilds)
    return ilds

# computes both ITD and ILD using xcorr
def bb_itd_ild_xcorr(left, right):
    '''
    returns ILD in dB, ITD in samples
    '''
    Lf = np.fft.fft(np.hstack((left, np.zeros(len(left)))))
    Rf = np.fft.fft(np.hstack((right[::-1], np.zeros(len(right)))))
    C = np.fft.ifft(Lf*Rf).real

    itd_idx = len(left) - np.argmax(C)
    ild = 20*np.log10(C.max())
    return itd_idx, ild

def itd_ild_xcorr(hrir, cf, cpu = 1):
    '''
    Computes the ITDs and ILDs for a given binaural impulse response
    if hrir is nsamplesx2 then simple
    else does some multiprocessing to compute the rest, using conventions noted in ImpulseResponse
    '''
    if isinstance(hrir, ImpulseResponse):
        db.log_debug('Starting ITD computations for '+str(hrir.ncoordinates)+' HRIR pairs')
        if hrir.ncoordinates > 1 and cpu > 1:
            Nruns = int(ceil(hrir.ncoordinates/float(cpu)))
            res = []
            for i in range(Nruns):
                startid = i * cpu
                endid = min(startid + cpu, hrir.ncoordinates)
                
                res += ph.map(itd_ild_bare, range(startid, endid),
                                  shared_data = {'samplerate' : 44100., 
                                                 'hrir': asarray(hrir), 
                                                 'cf': cf})
            return res
        else:
            if hrir.ncoordinates > 1:
                res = []
                for k in range(hrir.ncoordinates):
                    res.append(itd_ild_bare(hrir.forcoordinates
(k), cf))
                return res
            else:
                return itd_ild_bare(hrir, cf)
        
def itd_ild_bare(*args, **kwdargs):
    '''
    ITD/ILD computation routine. called by itd_ild that handles multiprocessing,...
    '''
    if len(args) == 2:
        # hrir, cf
        hrir, cf = args[0], args[1]
    elif len(args) == 1 and type(args[0]) == int:
        # k, shared_data
        shared_data = kwdargs['shared_data']
        k = args[0]
        hrir = shared_data['hrir']
        samplerate = shared_data['samplerate']
        cf = shared_data['cf']

        hrir = ImpulseResponse(hrir[:, [k, k + hrir.shape[1]/2]], 
                               samplerate = samplerate * Hz)
        
    samplerate = hrir.samplerate
    if (hrir[:,0] == hrir[:,1]).all():
        return (zeros(len(cf)),zeros(len(cf)))
    if (abs(hrir[:,0])<= 10e-6).all() or  (abs(hrir[:,1])<=10e-6).all():
        db.log_debug('Blank hrirs detected, output will be weird')
        
    if not isinstance(hrir, Sound):
        hrir = Sound(hrir, samplerate = samplerate)
        
    
    fb = Gammatone(Repeat(hrir, len(cf)), hstack((cf, cf)))
    filtered_hrirset = fb.process()
    itds = []
    ilds = []
    for i in range(len(cf)):
        left = filtered_hrirset[:, i]
        right = filtered_hrirset[:, i+len(cf)]
        
        # This FFT stuff does a correlate(left, right, 'full')
        Lf = fft(hstack((left, zeros(len(left)))))
        Rf = fft(hstack((right[::-1], zeros(len(right)))))
        C = ifft(Lf*Rf).real
        
        i = argmax(C)+1-len(left)
        
        itds.append(i/samplerate)
        ilds.append(sqrt(amax(C)/sum(right**2)))
        
    itds = array(itds)
    ilds = array(ilds)
    
    return itds, ilds



                               

################# IPD and phase unwrapping
def ipd(hrir, unwrap = False, threshold = np.pi, positivefreqs = False):
    '''
    Computes the phase difference of a binaural ImpulseResponse object.
    '''
    hrir_ft = fft(asarray(hrir), axis = 0)
    middle = hrir.shape[1]/2
    ipds =  np.angle(hrir_ft[:, :middle]) - np.angle(hrir_ft[:, middle:])
    if unwrap:
        ipds = np.unwrap(ipds, axis = 0, discont = threshold)
    return ipds
