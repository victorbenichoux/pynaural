'''
This is a new class designed to replace the old Model paradigm that is getting messy and outdated
'''
from .acoustics import c
from .receivers import HRTFReceiver
from .dsp.impulseresponse import ImpulseResponse, dur2sample, delayIR
from .dsp.filters import firwin2
from .dsp.misc import fftconvolve
from .utils.spatprefs import get_pref
from scipy.signal import *
from brian.stdunits import Hz
from matplotlib.pyplot import *
import numpy as np

class Model(object):
    '''
    A model is a class that goes through a Beam object and does various computations on it.
    
    '''
    pass

class DistanceModel(object):
    def __init__(self):
        pass
    
    def render(self, beam):
        return beam.get_totaldists()

################## Delay + Global Attenuation Model ############################

class DelayAttenuationModel(object):
    '''
    Accumulates delays and attenuations for each rays, then if relevant also uses HRTFs
    '''
    def __init__(self, surfaces, values):
        self.surfaces = surfaces
        self.ids = None
        
        self.values = values
    
    def render(self, beam, receiver = None, 
               method = 'closest', 
               nfft = None, samplerate = None):
        '''
        nfft is only used without hrtf
        method only with hrtfs
        '''
        self.check_for_surf_table()

        delays, attenuations = self.compute(beam)
        
        if isinstance(receiver, HRTFReceiver):
            HRIRs = receiver.computeHRIRs(beam, method = method)

            Npairs = HRIRs.shape[1]/2

            attenuations = np.tile(attenuations.reshape((1, Npairs)), (HRIRs.shape[0], 1))

            HRIRs[:,:Npairs] *= attenuations
            HRIRs[:,Npairs:] *= attenuations
            data = np.zeros((HRIRs.shape[0], 2))
            weights = np.ones(Npairs)/Npairs
            for i in range(Npairs):
                delay_samp = dur2sample(delays[i], HRIRs.samplerate)
                data[:,0] += np.roll(HRIRs[:,i], delay_samp)*weights[i]
                data[:delay_samp,0] = 0
                data[:,1] += np.roll(HRIRs[:,i+Npairs], delay_samp)*weights[i]
                data[:delay_samp,1] = 0

            data[:,0] = np.sum(HRIRs[:,:Npairs],axis = 1)/Npairs
            data[:,1] = np.sum(HRIRs[:,Npairs:],axis = 1)/Npairs
            
            return ImpulseResponse(data, samplerate = HRIRs.samplerate, binaural = True)
        else:
            if samplerate is None:
                samplerate = get_pref('DEFAULT_SAMPLERATE', default = 44100*Hz)
                
            if nfft is None:
                nfft = beam.detect_nfft(samplerate)

            data = np.zeros((nfft, 1))
            for i in range(len(delays)):
                data += attenuations[i]*delayIR(dur2sample(delays[i], samplerate = samplerate), (nfft, 1))/beam.nrays
            return ImpulseResponse(data, samplerate = samplerate)

    def compute(self, beam):
        self.check_for_surf_table()
        curbeam = beam[beam.get_reachedsource()]
        delays = np.zeros(curbeam.nrays)
        attenuations = np.ones((curbeam.nrays, 1))
        
        while True:
            currays = -np.isnan(curbeam.distances)
            delays[currays] += curbeam.distances[currays]/c
            
            for i in self.ids:
                idx = np.nonzero(curbeam.surfacehit == i)
                attenuations[idx] *= self.values[i]
            
            if curbeam.hasNext():
                curbeam = curbeam.next
            else:
                break
        
        return delays, attenuations
                 
           
    def check_for_surf_table(self):
        if self.ids == None:
            self.ids = [surface.get_id() for surface in self.surfaces]


################## Delay + Multiple Attenuation Model ############################

class DelayBandAttenuationModel(object):
    '''
    Delay + Band attenuation model
    
    initializer with (surfaces, gains, freqs = freqs)
    
    gains specifies the gains for each surface wanted at each freq point in freqs.
    '''
    def __init__(self, surfaces, 
                 gains, freqs = None, numtaps = None):
        self.surfaces = surfaces
        self.ids = None
        if gains.ndim == 1:
            self.gains = gains.reshape((1, len(gains)))
        else:
            self.gains = gains
        self.freqs = freqs
        if numtaps == None:
            numtaps = 1024
        self.numtaps = numtaps

    def render(self, beam, receiver = None, 
               method = 'closest', 
               nfft = None, samplerate = None):
        '''
        nfft is only used without hrtf
        method only with hrtfs
        '''
        self.check_for_surf_table()

        delays, attenuations = self.compute(beam)
        beam = beam[beam.get_reachedsource()]
        if self.freqs == None:
            freqs = np.linspace(0, 1, gains.shape[0]) * float(samplerate)/2.

        if samplerate is None:
            samplerate = get_pref('DEFAULT_SAMPLERATE', default = 44100*Hz)
            samplerate = 44100*Hz
            

        if nfft is None:
            nfft = 2*beam.detect_nfft(samplerate) + self.numtaps
        
        if isinstance(receiver, HRTFReceiver):
            HRIRs = receiver.computeHRIRs(beam, method = method)

            if HRIRs.samplerate != samplerate:
                raise ValueError('Samplerates not matching')

            if nfft < HRIRs.nsamples:
                nfft = 2*HRIRs.nsamples + self.numtaps

            Npairs = HRIRs.shape[1]/2

            data = np.zeros((self.numtaps+HRIRs.nsamples-1, 2))

            weights = np.ones(Npairs)/Npairs
            for i in range(Npairs):
                # build a filter
                curfilter = firwin2(self.numtaps, self.freqs, attenuations[:,i], nyq = float(samplerate)/2)
                # filter the HRTFs
                left = fftconvolve(HRIRs[:,i], curfilter)
                right = fftconvolve(HRIRs[:,i+Npairs], curfilter)
                # compute delay
                delay_samp = dur2sample(delays[i], HRIRs.samplerate)
                # apply delay and go
                data[:,0] += np.roll(left, delay_samp)*weights[i]
                data[:delay_samp,0] = 0
                data[:,1] += np.roll(right, delay_samp)*weights[i]
                data[:delay_samp,1] = 0

            return ImpulseResponse(data, samplerate = HRIRs.samplerate, binaural = True)
        else:
            data = np.zeros(nfft)
            for i in range(len(delays)):
                curfilter = firwin2(self.numtaps, self.freqs, attenuations[:,i], nyq = float(samplerate)/2)
                
                delays_samp = dur2sample(delays[i], samplerate = samplerate)
                data[delays_samp:delays_samp+self.numtaps] += curfilter
                
#                data += attenuations[i]*delayIR(dur2sample(delays[i], samplerate = samplerate), (nfft, 1))/beam.nrays
            return ImpulseResponse(data, samplerate = samplerate)

    def compute(self, beam):
        self.check_for_surf_table()
        curbeam = beam[beam.get_reachedsource()]
        delays = np.zeros(curbeam.nrays)
        attenuations = np.ones((self.gains.shape[0], curbeam.nrays))
        while True:
            currays = -np.isnan(curbeam.distances)
            delays[currays] += curbeam.distances[currays]/c
            
            for i in self.ids:
                idx = np.nonzero(curbeam.surfacehit == i)[0]
                attenuations[:, idx] *= self.gains[:,i].reshape((self.gains.shape[0], 1))
            if curbeam.hasNext():
                curbeam = curbeam.next
            else: 
                break
        
        return delays, attenuations
            
    def check_for_surf_table(self):
        if self.ids == None:
            self.ids = [surface.get_id() for surface in self.surfaces]




        

        
    
