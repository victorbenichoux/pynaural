from pynaural.raytracer.geometry.rays import *
from pynaural.utils.debugtools import log_debug

from pynaural.raytracer.faddeeva import faddeeva
import numpy as np
from numpy.fft import fft, fftfreq
from matplotlib.pyplot import *

__all__ = ['c']

## Constants
c=342#*meter/second #sound speed
T0=293.15#*kelvin #ref temperature (20C)
hr0=0.5 # ref hygrometry
To1=273.16#*kelvin # triple point temperature
Pr=101.325e3#*pascal #reference ambient atm pressure

class Model(object):
    '''
    Models should be associated to surfaces. (possibly to sources as well)
    Called when the reflection is computed, or collapsed
    They should return complete transfer functions
    (that is nfreqs x nrays arrays)
    
    They should also be initialized with a frequency region. but maybe after specifying the model (e.g when calling render() from the scene)
    '''
    def __init__(self, model):
        if callable(model):
            ## see connections/construction
            self.modelf = model
            self.nargs = model.func_code.co_argcount
            if self.nargs > 2:
                warnings.warn('A model with more than 2 arguments has been passed. discarding the rest')
        elif isinstace(model,np.ndarray):
            ValueError('Model func must be callable') #ok maybe not
            
    def prepare(self, samplerate, nfft):
        self.samplerate = float(samplerate)
        self.nfft = nfft

        if self.nargs == 0:
            return self.nfreqs
        if self.nargs == 1:
            try:
                failed = (np.array(self.modelf(np.zeros(self.nfft,10))).size != freqs.size)
            except:
                failed = True
        elif self.nargs == 2:
            try:
                # checks for vectorization in both directions
                failed = (np.array(self.modelf(self.freqs,np.zeros(self.nfft,10))).shape != (self.nfft, 10)  )
            except:
                failed = True
            if failed:
                log_debug('No vectorization detected')
            # no vectorization in freqs and incidences, maybe
            # in one of those, but not both, so what do we do?
    
    def compute(self, *args):
        nargs = len(args)
        if nargs != self.nargs:
            raise ValueError('modele appele avec mauvais nb dargs')
        else:
            newargs = []
            if self.nargs != 1:
                nrays = len(args[1])
                newargs.append(np.tile(args[0].reshape((nfft,1)),(1,nrays)))
                for i in range(1,self.nargs):
                    newargs.append(np.tile(args[i],(nfreqs,1)))
                newargs = tuple(newargs)
            else:
                newargs = args
#        print self.modelf(1j,1j)
        return self.modelf(*newargs)
    
    def plot(self, *args, **kwargs):
        '''
        Plots the response of the model for all specified argument values.
        '''
        if self.nargs!= len(args):
            raise ValueError('Wrong number of arguments')
        if 'samplerate' in kwargs:
            samplerate = kwargs['samplerate']
        else:
            samplerate = 44100.
        if 'nfft' in kwargs:
            nfft = kwargs['nfft']
        else:
            nfft = 44100
        if 'figure' in kwargs:
            fig = kwargs['figure']
        else:
            fig = None
        if 'display' in kwargs:
            display = kwargs['display']
        else:
            display = None
        
        self.prepare(samplerate, nfft)

        try:
            nd = len(args[0])
            d = np.array(args[0])
        except TypeError:
            nd = 1
            d = np.array([args[0]])
        if self.nargs > 1:
            try:
                ni = len(args[1])
                i = np.array(args[1])
            except TypeError:
                ni = 1
                i = np.array([args[1]])
        else:
            i = np.array([])
            ni = 1
        
        newdist = np.zeros(ni*nd)
        if self.nargs == 2:
            newinc = np.tile(i.reshape((1,len(i))), (1, nd))
        for k,dk in enumerate(d):
            newdist[k*ni:(k+1)*ni] = dk*np.ones((1, ni))
        
        if self.nargs == 1:
            newargs = tuple([newdist.flatten()])
        else:
            newargs = (newdist.flatten(), newinc.flatten())

        res = self.compute(*newargs)
        freqs = fftfreq(nfft)*samplerate

        f = figure(fig)
        ax1 = subplot(211)
        ax1.set_title(self.__class__.__name__)
        xlabel('Frequency (Hz)')        
        ylabel('Attenutation (dB)')
        ax2 = subplot(212)
        xlabel('Frequency (Hz)')        
        ylabel('Phase (rad)')   
        
        
        def strlabel(newargs, k):
            s = 'd='+str(newargs[0][k])[:4]
            if len(newargs) == 2:
                s += ' i='+str(newargs[1][k])[:4]
            return s

        for k in range(res.shape[1]):
            ax1.semilogx(freqs, 20*np.log10(np.abs(res[:,k])), label = strlabel(newargs,k))
            ax2.semilogx(freqs, np.angle(res[:,k]), label = strlabel(newargs,k))
        
        ax1.legend()
        ax2.legend()
        ylim((-np.pi, np.pi))
        
        if display == True:
            show()

class FrequencyModel(Model):
    def __init__(self):
        super(FrequencyModel,self).__init__(lambda f: f)

    def prepare(self):
        return super(FrequencyModel,self).prepare(np.array([1]))

class LogDistanceModel(Model):
    def __init__(self):
        super(LogDistanceModel,self).__init__(lambda f,d: np.log(d))
    
    def prepare(self, *args, **kwargs):
        return super(LogDistanceModel,self).prepare(np.array([1]))
        
class IncidenceModel(Model):
    def __init__(self):
        super(IncidenceModel,self).__init__(lambda f,d,i: i)
    
    def prepare(self):
        return super(IncidenceModel,self).prepare(np.array([1]))

class RigidReflectionModel(Model):
    def __init__(self, gain, shift_delay = 0.):
        self.nargs = 1
        if isinstance(gain, dB_type):
            self.gain = 10**(-float(gain)/20.)
        else:
            self.gain = gain
        self.shift_delay = shift_delay    

    def prepare(self, samplerate, nfft):
        self.samplerate = float(samplerate)
        self.nfft = nfft
        f = fftfreq(int(self.nfft))
        samplerate = float(self.samplerate)
        duration = np.array([self.nfft/self.samplerate])
        
        def fun(d):
            delay = d/c
            if (delay > duration).any():
                raise ValueError('The delay ('+str(np.max(delay)*1000)+' ms)is longer than the duration of the IR ('+str(duration*1000)+' ms)')
            delay_samples = np.round(delay/duration * self.nfft)
            freqs = np.tile(f.reshape((self.nfft, 1)), (1,len(d)))
            gains = np.tile((self.gain * (d**-1)).reshape((1, len(d))), (self.nfft, 1))
            tf = gains * np.exp(-1j*2*np.pi*delay_samples*freqs)
            return tf
        super(RigidReflectionModel,self).__init__(fun)

    def compute(self, d):
        return self.modelf(d)
    
    @property
    def alpha(self):
        return 1-self.gain
    
class RigidReflectionModelNoRound(Model):
    def __init__(self, gain, shift_delay = 0.):
        self.nargs = 1
        if isinstance(gain, dB_type):
            self.gain = 10**(-float(gain)/20.)
        else:
            self.gain = gain
        self.shift_delay = shift_delay    

    def prepare(self, samplerate, nfft):
        self.samplerate = float(samplerate)
        self.nfft = nfft
        f = fftfreq(int(self.nfft))
        samplerate = float(self.samplerate)
        duration = np.array([self.nfft/self.samplerate])
        
        def fun(d):
            delay = d/c
            if (delay > duration).any():
                raise ValueError('The delay ('+str(np.max(delay)*1000)+' ms)is longer than the duration of the IR ('+str(duration*1000)+' ms)')
            delay_samples = delay/duration * self.nfft
            freqs = np.tile(f.reshape((self.nfft, 1)), (1,len(d)))
            gains = np.tile((self.gain * (d**-1)).reshape((1, len(d))), (self.nfft, 1))
            tf = gains * np.exp(-1j*2*np.pi*delay_samples*freqs)
            return tf
        super(RigidReflectionModelNoRound,self).__init__(fun)

    def compute(self, d):
        return self.modelf(d)

    @property
    def alpha(self):
        return 1-self.gain

class RigidAbsorptionModel(Model):
    def __init__(self, absorption, shift_delay = 0.):
        self.nargs = 1
        self.absorption = absorption
        self.gain = 1 - absorption
        self.shift_delay = shift_delay    

    def prepare(self, samplerate, nfft):
        self.samplerate = float(samplerate)
        self.nfft = nfft
        f = fftfreq(int(self.nfft))
        samplerate = float(self.samplerate)
        duration = np.array([self.nfft/self.samplerate])
        
        def fun(d):
            delay = d/c
            if (delay > duration).any():
                raise ValueError('The delay ('+str(np.max(delay)*1000)+' ms)is longer than the duration of the IR ('+str(duration*1000)+' ms)')
            delay_samples = delay/duration * self.nfft
            freqs = np.tile(f.reshape((self.nfft, 1)), (1,len(d)))
            gains = np.tile((self.gain * (d**-1)).reshape((1, len(d))), (self.nfft, 1))
            tf = gains * np.exp(-1j*2*np.pi*delay_samples*freqs)
            return tf
        super(RigidReflectionModelNoRound,self).__init__(fun)

    def compute(self, d):
        return self.modelf(d)

    @property
    def alpha(self):
        return self.absorption
        
    
class DummyModel(Model):
    def __init__(self):
        self.nargs = 1
        pass
    
    def prepare(self, samplerate, nfft):
        self.samplerate = samplerate
        self.nfft = nfft
        pass
    
    def compute(self,d):
        return np.ones((self.nfft, len(d)), dtype = complex)
    
class PropagationModel(RigidReflectionModel):
    def __init__(self, gain = 0, shift_delay = 0.):
        super(PropagationModel, self).__init__(1., shift_delay = shift_delay)

################### Atmospheric air absorption ############################
# todo: rethink that class, maybe a model kind of thing would be nice.
# one could thinkg of other was of probing a model class as well
# this is definitely a good idea.
# i need to program stuff for beams.

def atm_abs(d, f, T=T0, hr=hr0, p=Pr, ref=1.0, broadbandcorrection=False):
    # Computes the desired amplitude modulation for air absorption
    # returns values in 
    # ISO 9613-1:1993
    # http://www.sengpielaudio.com/AirdampingFormula.htm
    # (...) = range of validity
    #d: distance
    #f: frequency (50HZ ... 10 kHz)
    #T: Temperature (-20C ... 50 C)
    #h: humidity level (.1 ... -1)
    #p: atmospheric pressure
    
    Psat = Pr * 10 ** (-6.8346 * (To1/T)**1.261 + 4.6151)
    h = hr * (Psat/p)
    
    frN = (p/Pr) * (T/T0)**(-0.5) *( 9+280*h*exp(-4.170*( (T/T0)**(-1./3) -1)))
    frO = (p/Pr) * (24 + 4.04*10**4 * h *( (0.02+h)/(0.391+h) ) )
    
    z = 0.1068 * exp(-3352*kelvin/T) * (frN + f**2/frN)**-1
    y = (T/T0)**(-5./2) * ( 0.01275*exp(-2239.1*kelvin/T)*(frO+f**2/frO)**-1 + z )
    x = 1/(10 * log10((exp(1))**2))
    
    a = 8.686*f**2*( 1.84*10**(-11) * (p/Pr)**-1 * (T/T0)**.5 + y ) #absorption coef
    
    Aa = -20*log10(exp(-x*a*d))
    
    if broadbandcorrection:
        # correction for broad band sound (crocker chap 28)
        Aa = Aa*(1+0.00533*(1-0.2303*Aa))**1.6
    
    Aa.shape = (len(Aa), 1)
    
    return Aa

def spreading_abs(d, condition='spherical', ref=1.0):
    g={ 'spherical':1,
        'plane':0,
        'cylindrical':0.5}[condition]
    As=g*20*log10(d/ref+1)
    return As

def delay_abs(d, f, ref=1.0):
    delay = float(d/c/second)
    p = exp(- 1j*2*pi*delay*f )
    p.shape=(len(p),1)
    return p

def freefield_abs(d,f,ref=1):
    At=exp( - ( flipnegfreq(atm_abs(d,fullrangefreq(f),ref=ref))  + spreading_abs(d,ref=ref) ) / 20)
    H=At*delay_abs(d,f,ref=ref)
    return H


################### Ground reflection model ############################

def k21_z21(f, sigma, model=''):
    # computes the ratio of wavenumbers, and ratio of impedances
    # Miki, and then Komatsu
    z21=(1+0.0699*((f/sigma)**(-0.632)))+1j*0.1071*((f/sigma)**(-0.632))
    k21=(1+0.1093*((f/sigma)**(-0.618)))+1j*0.1597*((f/sigma)**(-0.618))
    # tmp = 2-log(f/(1000*sigma))
    # tmp62 = tmp ** 6.2
    # tmp41 = tmp ** 4.1
    # z21 = (1 + 0.00027 * tmp62) - 0.0047 * tmp41
    # k21 = (0.0069 * tmp41) + 1j * (1 + 0.0004 * tmp62)
    return k21,z21

def spherical_ref_factor(d, phi, f, sigma=20000, allres=False, version='mine'):
    # details of the implementation are in boris' paper
    # arguments:
    # d is the total distance from source to head, with reflection
    # phi is the elevation of the new (virtual) source
    # f is the considered frequency
    # sigma is the resistivity of the material in MKS raylgh
    if version=='boris':
        c=342
        k=2*np.pi*f/c
        k21,z21=k21_z21(f,sigma)
        temp=(1-(k21**-2)*(np.cos(phi)**2))
    #    print Theta,z21*sin((Theta)),(temp**0.5)
        Rp=(z21*np.abs(np.sin(phi))-(temp**0.5))/(z21*np.abs(np.sin(phi))+(temp**0.5))
        w=1j*2*k*d/((1-Rp)**2)*(z21**-2)*temp
    #    numd2=1j*k*d/2*((abs(sin(Theta))+z21**-1)**2)/(1+abs(sin(Theta))/z21)
        Fw=1+1j*np.sqrt(np.pi*w)*faddeeva(np.sqrt(w))#exp(-numd)*(1-erf(-1j*sqrt(numd)))
        return Rp + (1 - Rp) * Fw
    
    elif version=='mine':
        f=f+0j
        k21,z21 = k21_z21(f,sigma)
        S = (z21 ** -2) * (1 - (k21 ** -2) * (np.cos(phi) ) ** 2)
        sqS = S ** 0.5
        R = (np.sin(phi) - sqS) / (np.sin(phi) + sqS)
        c=343
        la = 2 * np.pi * f / c
        w = 2j * la * d * S * ((1-R) ** -2) 
        F = lambda x : 1 + 1j * (np.pi * x) ** 0.5 * faddeeva(np.sqrt(x))
        Fw = F(w)
        Q = R + (1 - R) * Fw
    if not allres:
        return np.nan_to_num(Q)#Q
    else:
        return R,w,Fw

class NaturalGroundModel(Model):
    '''
    implements Miki s model, see below
    '''
    def __init__(self, sigma, gain = .5):
#        self.complement = SimpleReflectionModel(gain)
        self.sigma = sigma
        self.nargs = 2
        
    def prepare(self, samplerate, nfft):
        self.samplerate = float(samplerate)
        self.nfft = nfft
#        self.complement.prepare(samplerate, nfft)

        def fun(distances, incidences):
            res = np.zeros((self.nfft, len(distances)), dtype = complex)
            print nfft
            freqs = fftfreq(nfft, d = 1/samplerate)
            f = freqs
            f[f<0] = -f[f<0]
            for i,d in enumerate(distances):
                res[:,i] = spherical_ref_factor(d, incidences[i], f, sigma = self.sigma)
                res[freqs>0, :] = np.conj(res[freqs>0, :])
            #att = self.gain * np.tile(distances.reshape((1, len(distances))), (nfft, 1))**-1
            return res#*self.complement.compute(distances)
        
        super(NaturalGroundModel, self).__init__(fun)

    def compute(self, d, i):
        return self.modelf(d, i)

        
