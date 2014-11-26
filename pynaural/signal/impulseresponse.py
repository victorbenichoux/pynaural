import os, re
import warnings, pickle
import matplotlib.cm as cmap
from scipy.signal import *
from scipy.io import loadmat
import numpy as np

try:
    from brian import *
    from brian.hears import *
    has_brian = True
except:
    has_brian = False

from pynaural.raytracer.geometry.base import *
from pynaural.signal.misc import *
from pynaural.signal.smoothing import *
from pynaural.utils.spatprefs import get_pref



__all__ = ['ImpulseResponse',
           'onesIR', 'zerosIR', 'delayIR', 'binauralIR',
           'dur2sample', 'sample2dur', 'TransferFunction',
           'onesTF', 'zerosTF', 'delayTF', 'binauralTF',
           'zeropad']

#####################################################################################################
########################################## Impulse Response #########################################
#####################################################################################################
        
class ImpulseResponse(np.ndarray):
    '''
    :class:`ImpulseResponse` for working with impulses responses. 
    This class basically behaves like an numpy.ndarray with each channel arranged as a column.
    
    Useful features:
    
    ... listening
    ir.listen()
    ... Spat related stuff
    ir.forsource(i)
    ir.nsources
    ...coordinates related stuff
    ir.forcoordinate
    ir.ncoordinates
    ir.coordinates
    .... dsp stuff
    ir.apply
    ... plotting
    ir.plot

    ir.reorder_coordinates

    Is used in the binaural cues module to compute itds/ilds
    
    '''
#    @check_units(samplerate=Hz, duration=second)
    def __new__(cls, data, 
                samplerate = None,  # major attributes
                binaural = False, # binaural flag
                coordinates = None, # coordinates attributes
                target_source = None, # spatializer related
                is_delay = False
                ):
        # Possibly initialize with TransferFunction
        if isinstance(data, TransferFunction):
            # that is for sure
            data_ift = ifft(data, axis = 0).real
            # then add all relevant attributes
            kwdargs = data.get_kwdargs()
            return ImpulseResponse(data_ift, **kwdargs)
        if isinstance(data, np.ndarray):
            if samplerate is None:
                samplerate = 44100.*Hz
                #samplerate = prefs.get_pref('DEFAULT_SAMPLERATE', default = 44100.)*Hz
            x = array(data, dtype = float)
        else:
            print cls, data
            raise ValueError('Wrong Impulse Response initialization')

        if x.ndim == 1:
            x.shape = (len(x), 1)

        if isinstance(target_source, int):
            target_source = np.array([target_source])
        x = x.view(cls)
        # minimum attributes
        x.samplerate = ensure_Hz(samplerate)
        # flags for different types of IR
        x.binaural = binaural
        x.is_delay = is_delay

        x.coordinates = _makecoordinates(x.shape, coordinates, binaural = binaural)
        x.target_source = _make_target_source(x.shape, target_source, binaural = binaural)
         
        # buffering
        x.buffer_init()#not sure about that
        return x
    
    ################## internals
    # util for new object/view creation
    # pass whenever creating a new ImpulseResponse in this class
    def get_kwdargs(self, _slice = None):
        res = {
            # major attribute
            'samplerate' : self.samplerate,
            # binaural flag
            'binaural' : self.binaural,
            # coordinates attributes
            'coordinates' : self.coordinates,
            # spatializer related            
            'target_source' : self.target_source,
            # to simplify apply
            'is_delay' : self.is_delay,
            }

        if slice is None:
            return res
        else:
            # kwdargs slicing
            # binaural: the ir stays binaural iff all the channels are taken
            if _slice  != None:
                test = np.zeros(self.shape[1])
                if not (test[_slice].shape == test.shape):
                    res['binaural'] = False
            # coordinates
            if self._has_coordinates:
                res['coordinates'] = self.coordinates[_slice].flatten()
            # spatializer internals
            if self._has_sources:
                res['target_source'] = self.target_source[_slice].flatten()
            return res

    ################### ndarray internals
    def __array_wrap__(self, obj, context=None):
        handled = False
        x = np.ndarray.__array_wrap__(self, obj, context)
        if not hasattr(x, 'samplerate') and hasattr(self, 'samplerate'):
            x.samplerate = self.samplerate
        if context is not None:
            ufunc = context[0]
            args = context[1]
        return x

    def __array_finalize__(self, obj):
        if obj is None: return
        # major attribute
        self.samplerate = getattr(obj, 'samplerate', None)
        # binaural flag
        self.binaural = getattr(obj, 'binaural', False)
        # coordinates
        self.coordinates = getattr(obj, 'coordinates', None)
        # spatializer internals
        self.target_source = getattr(obj, 'target_source', None)
        
    ################## Buffering
    # TODO: - CHECK BUFFERABLE (like Sounds)
    # - what do with channels?
    def buffer_init(self):
        pass
        
    def buffer_fetch(self, start, end):
        if start<0:
            raise IndexError('Can only use positive indices in buffer.')
        samples = end-start
        X = asarray(self)[start:end, :]
        if X.shape[0]<samples:
            X = vstack((X, zeros((samples-X.shape[0], X.shape[1]))))
        return X
    
    ################## SLICING #######################
    def __getitem__(self, key):
        return asarray(self).__getitem__(key)
    
    def __setitem__(self, key, val):
        asarray(self).__setitem__(key, val)
        
    def trim(self, *args):
        '''
        Returns the same HRIR but trimmed between start and stop.
        '''
        if len(args) == 2:
            start = args[0]
            end = args[1]
        elif len(args) == 1:
            start = 0
            end = args[0]
            
        if units.have_same_dimensions(start, second):
            start = dur2sample(start, self.samplerate)
        if units.have_same_dimensions(end, second):
            end = dur2sample(end, self.samplerate)

        data = self[start:end,:].copy()                
        kwdargs = self.get_kwdargs()
        return ImpulseResponse(data, **kwdargs)
        
    def zeropad(self, nzeros):
        '''
        Adds the specified number of zeros at the end of the IR
        '''
        zero = np.zeros((nzeros, self.shape[1]))
        data = np.vstack((self, zero))
        kwdargs = self.get_kwdargs()
        return ImpulseResponse(data, **kwdargs)

    def subsample(self, factor):
        data = np.zeros((self.shape[0]/factor, self.shape[1]))
        for ksig in xrange(self.shape[1]):
            data[:,ksig] = decimate(np.asarray(self)[:,ksig], factor)

        kwdargs = self.get_kwdargs()
        kwdargs['samplerate'] = kwdargs['samplerate']/factor
        return ImpulseResponse(data, **kwdargs)
        
    ################## binaural stuff
    @property
    def left(self):
        if not self.binaural:
            return self
        data = asarray(self[:, :self.shape[1]/2])
        kwdargs = self.get_kwdargs(_slice = slice(None, self.shape[1]/2))
        kwdargs['binaural'] = False
        return ImpulseResponse(data, **kwdargs)

    @property
    def right(self):
        if not self.binaural:
            return self
        data = asarray(self[:, self.shape[1]/2:])
        kwdargs = self.get_kwdargs(_slice = slice(self.shape[1]/2, None))
        kwdargs['binaural'] = False
        return ImpulseResponse(data, **kwdargs)

################## Coordinates
    @property
    def _has_coordinates(self):
        return not (self.coordinates == None)

    @property
    def ncoordinates(self):
        if self._has_coordinates:
            if self.binaural:
                return self.shape[1]/2
            else:
                return self.shape[1]
        else:
            raise AttributeError('Impulse Response doesnt have coordinates')

    def forcoordinates(self, *args):
        '''
        Should return an impulseResponse containing all the rays headed towards source args
        
        Usage:
        hrir.forcoordinates(20, 0) # returns az = 20 and elev =0
        hrir.forcoordinates([20, 0], [0, 5]) interpreted as az in [20,0] and elev in [0,5] so 4 pairs
        hrir.forcoordinates(lambda azim,elev: azim == 0 and elev in [0,7.5]
        
        hrir.forcoordinates(i) to iterate over positions
        '''
        # argument parsing
        if len(args) == 1 and isinstance(args[0], int):
            # indexing is just the i'th IR
            i = args[0]
            # kwdargs handling
            kwdargs = self.get_kwdargs(_slice = i)
        
            # data collection
            if self.binaural:
                data = np.zeros((self.shape[0], 2))
                data[:, 0] = self.left[:, i].flatten()
                data[:, 1] = self.right[:, i].flatten()
                kwdargs['binaural'] = True
            else:
                data = self[:,i]
            out = ImpulseResponse(data, **kwdargs)
            return out

        if len(args) == 1 and callable(args[0]):
            fun = args[0]
            
            idx = []
            for i, (az, el) in enumerate(self.coordinates[:self.shape[1]/2]):
                if fun(az, el):
                    idx.append(i)
            if len(idx) > 0:
                idx = np.array(idx)
                data = np.hstack((self[:, idx], self[:, idx + self.shape[1] /2]))

            
                kwdargs = self.get_kwdargs(_slice = idx)
                kwdargs['binaural'] = True
                return ImpulseResponse(data, **kwdargs)
            else:
                db.log_debug('No HRTF found matching condition')
                return None
            
        if len(args) == 2:
            azs = args[0]
            els = args[1]
            if not iterable(azs):
                azs = [azs]
            if not iterable(els):
                els = [els]
            f = lambda azim, elev: azim in azs and elev in els
            return self.forcoordinates(f)
                    
    #################### Spatializer features
    # this includes:
    # attrs:
    # - incoming_direction,source
    # methods:
    # - forsource()
    # flags for different types of ImpulseResponse
    
    @property
    def _has_sources(self):
        return not (self.target_source == None)
    
    @property
    def nsources(self):
        if self._has_sources:
            return len(np.unique(self.target_source))
        else:
#            db.log_debug('ImpulseResponse has no target_source, falling back, you should use .shape! or .ncoordinates')
            if self.binaural:
                return self.shape[1]/2
            else:
                return self.shape[1]


    def forsource(self, k):
        '''
        Returns the part of the impulse response that is related to source k (as precised in the target_source attribute)
        Only implemented for impulse response that are the result of computations in the spatializer
        '''
        if not self._has_sources:
            raise AttributeError('This method isnt implemented for this type of IR')
        else:
            if (self.target_source != k).all():
                raise ValueError('No data for this source')
            if not self.binaural:
                return self[:, self.target_source == k]
            else:
                data = np.zeros((self.nsamples, 2))
                data[:, 0] = self[:, k].flatten()
                data[:, 1] = self[:, k + self.shape[1]/2].flatten()
                kwdargs = self.get_kwdargs(_slice = k)
#                 kwdargs['binaural'] = self.binaural
                return ImpulseResponse(data, **kwdargs)
            
    @property
    def sources(self):
        return np.unique(self.target_source)

    ################### DSP Features
    # -listening of IRs
    # -applying to sounds
    def rms(self):
        return np.sqrt(1/self.duration*(np.sum(self**2, axis = 0)))

    def downsample(self, q):
        '''
        Downsamples the ImpulseResponse by a factor q (integer). Makes use of scipy.signal.decimate.
        
        '''
        data = scipy.signal.decimate(self, q)[0,:,:]
        kwdargs = self.get_kwdargs()
        kwdargs['samplerate'] = kwdargs['samplerate']/q
        return ImpulseResponse(data, **kwdargs)
    
    def normalize(self, level = 1.):
#        id_max = np.argmax(np.abs(self))
#        value = asarray(self).reshape((self.shape[0]*self.shape[1],))[id_max]
        value = np.max(self)
        return self/value*level

    def atlevel(self, level):
        '''
        Normalizes the IR relative to its most intensive source (in terms of max)
        '''
        raise NotImplementedError
        
        
    def listen(self, sound = None, sleep = True, reference = True):
        '''
        If impulse response is more than one binaural or monaural,
        then all the sounds are played in a sequence
        
        TDO: WRITE DOC
        '''
        if sound is None:
            soundfile = get_pref('DEFAULT_SOUND', default = -1)
            if not soundfile == -1:
                import os
                if os.path.isdir(soundfile):
                    files = [f for f in os.listdir(soundfile) if f[-3:] == 'wav']
                    f = files[np.random.randint(len(files))]
                    soundfile += f
                sound = Sound.load(soundfile).left
            if soundfile == -1 or sound.samplerate != self.samplerate:
                sound = pinknoise(500*ms, samplerate = self.samplerate)
        else:
            if type(sound) == str:
                sound = Sound.load(sound)

        if reference:
            db.log_debug('Listening to original sound')
            sound.atlevel(60*dB).play(sleep = sleep)

        if self._has_coordinates:
            if self.binaural:
                db.log_debug('Listening to IR for coordinates '+
                             str(self.coordinates[:len(self.coordinates)/2]))
            else:
                db.log_debug('Listening to IR for coordinates '+
                             str(self.coordinates))
        else:
            db.log_debug('Listening to IR')
            
        out = self.apply(sound)
        out = Sound.sequence(out)
        out.atlevel(60*dB).play(sleep = sleep)
        return out
        

    def apply(self, other, outlevel = None):
        '''
        Applies the current Impulse responses to a sound.
        Either yields a single sound in the case where the Impulse response is only relative to a single source. Otherwise outputs a list of sounds for each source.
        
        Gain normalisation (outlevel kwdarg): 
        By default no gain, otherwise the gain is adjusted to that the loudest channel is at the level specified by outlevel. Hence the difference in level between the two channels is preserved. This is a different behavior from the one in brian.hears.sounds        
        '''
        if isinstance(other, Sound) or isinstance(other, ImpulseResponse):
            if not other.samplerate == self.samplerate:
                if True:
                    db.log_debug('Warning, samplerates not matching!')
                else:
                    raise ValueError('Samplerates not matching')
            if self._has_coordinates and self.ncoordinates > 1:
                res = []
                for i in range(self.nsources):
                    res.append(self.forcoordinates(i)._apply(other, outlevel = outlevel))
                return res
            else:
                return self._apply(other, outlevel = outlevel)
        else:
            raise AttributeError('Can only apply to a sound or IR')
            
    def _apply(self, other, outlevel = None):

        if not self.binaural:
            res = zeros((other.shape[0]+self.shape[0]-1, other.nchannels))
            for chan in range(other.nchannels):
                if self.is_delay:
                    delay_samples = np.nonzero(self.flatten())[0]
                    res[delay_samples:delay_samples+other.shape[0], chan] = other[:,chan].flatten()
                else:
                    res[:, chan] = fftconvolve(other[:,chan].flatten(), self)
            return Sound(res, samplerate = other.samplerate)
        else:
            res = zeros((other.shape[0]+self.shape[0]-1, 2))
            if other.nchannels !=1:
#                raise ValueError('Don\'t know what to do with 2 channel sound and binaural IR')
                db.log_debug('Applying IR separately to each channel')
                if self.is_delay:
                    print "here0"
                    delay_samples = np.nonzero(self.flatten())[0]
                    res[delay_samples:delay_samples+other.shape[0], chan] = other[:,chan].flatten()
                else:
                    res[:,0] = fftconvolve(other.left.flatten(), self[:,0].flatten())
                    res[:,1] = fftconvolve(other.right.flatten(), self[:,1].flatten())
            else:
                if self.is_delay:
                    print "here"
                    delay_samples_0 = np.nonzero(self[:,0].flatten())[0]
                    delay_samples_1 = np.nonzero(self[:,1].flatten())[0]
                    res[delay_samples_0:delay_samples_0+other.shape[0], 0] = other[:,0].flatten()
                    res[delay_samples_1:delay_samples_1+other.shape[0], 1] = other[:,1].flatten()
                else:
                    res[:,0] = fftconvolve(other.flatten(), self[:,0].flatten())
                    res[:,1] = fftconvolve(other.flatten(), self[:,1].flatten())
                if isinstance(outlevel, dB_type):
                    rms_dB_left = dB_SPL(res[:,0])
                    rms_dB_right = dB_SPL(res[:,1])
                    gain_dB = min(outlevel-rms_dB_left, outlevel-rms_dB_right)
                    gain = 10**(float(gain_dB)/20.)
                    res *= gain
            return Sound(res, samplerate = other.samplerate)

    def window(self, t = None, bandwidth = None):
        '''
        Uses smoothing.apply_windowing to window the impulseresponse
        '''
        return apply_windowing(self, t = t, bandwidth = bandwidth)

######################### Plotting
    def plot_binaural_cues(self, **kwdargs):
        selftf = TransferFunction(self)
        return selftf.plot_binaural_cues(**kwdargs)

    def plot(self, cutIR = None, display = False, dB = False, label = ''):
        '''
        easy plotting of IRs
        '''
        # dB conversion?
        conv = lambda x: x
        suff = ''
        if dB:
            conv = lambda x: 20*np.log10(np.abs(x))
            suff += ' (dB)'
        
        # IR truncation?
        cutIR = cutIR or self.shape[0]
        if isinstance(cutIR, Quantity):
            if not units.have_same_dimensions(cutIR, second):
                raise DimensionMismatchError('cutIR must be specified in samples or in seconds')
            else:
                cutIR = round(float(cutIR) * self.samplerate)

        cutIR = min(float(cutIR), self.shape[0])

        # plotting per se
        n = self.shape[1]
        if self.binaural:
            n /= 2
        for i in range(n):
            if self.binaural:
                dataleft = conv(self[:cutIR, i])
                dataright = conv(self[:cutIR, i + n])
                subplot(211)
                plot(arange(cutIR)*1000.0/float(self.samplerate), dataleft, label = label+' L')
                subplot(212)
                plot(arange(cutIR)*1000.0/float(self.samplerate), dataright, label = label+' R')
            else:
                plot(arange(cutIR)*1.0/float(self.samplerate), self[:cutIR,i], label = label)

        # axes labeling, legends
        if self.binaural:
            axleft = subplot(211)
            ylabel('Left IR'+suff)
            xlabel('Time (ms)')
            if not label == '':
                legend()
            axright = subplot(212)
            if not label == '':
                legend()
            ylabel('Right IR'+suff)
            xlabel('Time (ms)')
        else:
            if not label == '':
                legend()
            ylabel('IR'+suff)
            xlabel('Time (ms)')
        # display?
        if display:
            show()

    def plot_spectrum(self, *args, **kwdargs):
        '''
        Calls plot on the transfer function from this ir
        '''
        tf = TransferFunction(self)
        return plot(*args, **kwdargs)

    def convolve(self, other):
        '''
        convolving two IRs should take them to the Frequency domain, linear convolve them
        '''
        raise NotImplementedError('Not yet')
        if not (isinstance(other, Sound) or isinstance(other, ImpulseResponse)):
            raise TypeError('Can only convole sounds and impulse responses')
        if self.shape != other.shape:
            raise ValueError('Shape mismatch in convolution')
        if self.samplerate != other.samplerate:
            raise ValueError('Samplerate mismatch')
        if not other.nchannels == self.nchannels:
            raise ValueError('Number of channels mismatch')

        self_offset0 = np.min(np.argmin(1.0*(np.abs(self) < 1e-10), axis = 0))
        self_offset1 = self.shape[0]-np.min(np.argmin(1.0*(np.abs(self[::-1,:]) < 1e-10), axis = 0)) + 1
        other_offset0 = np.min(np.argmin(1.0*(np.abs(other) < 1e-10), axis = 0))
        other_offset1 = other.shape[0]-np.min(np.argmin(1.0*(np.abs(other[::-1,:]) < 1e-10), axis = 0)) + 1
        # we do a real linear convolution, mostly because lengths never match
        # it is costly so we try and trim the scene IRs that are quite often sparse
        nself = self_offset1 - self_offset0
        nother = other_offset1 - other_offset0
        N =  2**(np.ceil(log2(nself + nother - 1)))
        self_padded = np.vstack((self[self_offset0:self_offset1, :], np.zeros((N - nself, self.nchannels))))
        other_padded = np.vstack((other[other_offset0:other_offset1,:], np.zeros((N - nother, other.nchannels))))
        convolution = ifft(fft(self_padded, axis = 0)*fft(other_padded, axis = 0), axis = 0).real
        res = vstack((zeros((max(self_offset0 + other_offset0 - 1, 0), self.nchannels)), 
                      convolution))

        res = vstack((res, zeros((self.shape[0] + other.shape[0] - 1 - res.shape[0], self.nchannels))))
        
        return ImpulseResponse(res, samplerate = self.samplerate, binaural = self.binaural, coordinates = self.coordinates)
            

    # ndarray stuff
    def copy(self):
        kwdargs = self.get_kwdargs()
        data = np.ndarray.copy(self)
        return ImpulseResponse(data, **kwdargs)
    
    def reorder_coordinates(self, order):
        '''
        Returns a version of the ImpulseResponse with coordinates reordered according to the (ncoordinates,) integer permuation.
        '''
        data = np.ndarray.copy(self)
        kwdargs = self.get_kwdargs()
        newcoordinates = kwdargs['coordinates'][:self.ncoordinates][order]
        data[:,:self.ncoordinates] = self[:,order]
        if self.binaural:
            data[:,self.ncoordinates:] = self[:,order+self.ncoordinates]
        kwdargs['coordinates'] = newcoordinates
        return ImpulseResponse(data, **kwdargs)

    ##### TODO : check why repr and str don't work without this
    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        res = "ImpulseResponse object\n"
        res += "Samplerate: %.1f Hz\n" % self.samplerate
        res += "Binaural: %s\n" % ({True: "yes", False: "no"}[self.binaural])
        res += "Shape (Nsamples, Nchannels): (%d, %d)\n" % (self.shape[0], self.shape[1])
        if self._has_coordinates:
            res += "coordinates:" + str(self.coordinates) + "\n"
        return res+(np.asarray(self).__repr__())

    # global properties
    @property
    def nchannels(self):
        return self.shape[1]
    
    @property
    def duration(self):
        return (self.shape[0] / self.samplerate)
    
    @property
    def nsamples(self):
        return self.shape[0]

    @property
    def times(self):
        return np.arange(self.nsamples)/self.samplerate

    # acoustics related measures
    def responsetime(self, criterion = 60*dB):
        '''
        Returns the response time, that is the time when the initial signal loses 60 dB.
        '''
        if isinstance(criterion, dB_type):
            criterion = 10**(-float(criterion)/20)
        cumenergy = np.array(np.cumsum(self.normalize(), axis = 0)/self.samplerate)
        totalenergy =  cumenergy[-1, :]
        n = np.nonzero(cumenergy/totalenergy > criterion)[0][0]
        res = n/self.samplerate
        return res

    def reverberationtime(self, **kwdargs):
        return self.responsetime(**kwdargs)
        
    def itd_onset(self, threshold = 0.15):
        return self.left.onset_time(threshold = threshold) - self.right.onset_time(threshold = threshold)

    def onset_time(self, threshold = .15, unit = 'time'):
        '''
        Returns the time it takes for the IR to reach threshold of its max value
        `` unit'' if 'samples', then in samples, if 'time' (default) then is second.
        '''
        thresholds = threshold * np.amax(np.asarray(self), axis = 0)
        ids = np.zeros_like(thresholds)
        for i in range(thresholds.shape[0]):
            data = np.asarray(self)[:,i]
            tmp = np.nonzero(data > thresholds[i])
            ids[i] = tmp[0][0]
        if unit == 'samples':
            return ids
        else:
            return ids/self.samplerate
    

    # load/save
    def save(self, fileobj):
        '''
        Saves the impulse response as a npz file on the file specified by fileobj.
        '''
        additional_info = np.array([self.samplerate, 
                                    self.binaural])
        
        np.savez(fileobj,
                 data = self,
                 coordinates = self.coordinates,
                 target_source = self.target_source,
                 additional_info = additional_info)

    @staticmethod
    def load(fileobj, coordsfilter = None):
        '''
        Loads an impulse response as saved using the .save method.
        '''
        npzfile = np.load(fileobj)

        coordinates = npzfile['coordinates']
        if coordinates.ndim == 0:
            coordinates = None

        if (not (coordinates is None)) and coordsfilter:
            azs = coordinates['azim']
            els = coordinates['elev']
            indices = np.ones(len(azs), dtype = bool)
            for k in range(len(azs)):
                az = azs[k]
                el = els[k]
                if not coordsfilter(az, el):
                    indices[k] = False
            coordinates = coordinates[indices]
        else:
            indices = slice(None)#np.ones(npzfile['data'].shape[1], dtype = bool)

        target_source = npzfile['target_source']
        if target_source.ndim == 0:
            target_source = None

        if not (target_source is None) and coordsfilter:
            target_source = target_source[indices]

        data = npzfile['data'][:, indices]        
        additional_info = npzfile['additional_info']
        samplerate = float(additional_info[0])*Hz
        binaural = bool(additional_info[1])
        
        res = ImpulseResponse(data, 
                              coordinates = coordinates, 
                              target_source = target_source,
                              binaural = binaural,
                              samplerate = samplerate)
        return res        

    # constructors
    @staticmethod
    def zerosIR(shape, **kwdargs):
        shape = _shape_from_kwdargs(shape, kwdargs)
        return ImpulseResponse(np.zeros(shape), **kwdargs)

    @staticmethod
    def delayIR(delay, shape, **kwdargs):
        defaultsamplerate = prefs.get_pref('DEFAULT_SAMPLERATE', 
                                           default = 44.1*kHz)
        samplerate = kwdargs.get('samplerate', defaultsamplerate)
        kwdargs['is_delay'] = True
        res = zerosIR(shape, **kwdargs)
        if isduration(delay):
            delay = dur2sample(delay, samplerate)
        if delay > res.shape[0]:
            raise ValueError('Delay too long for the duration of the IR')
        res[delay,:] = 1
        return res

    @staticmethod
    def onesIR(shape, **kwdargs):
        shape = _shape_from_kwdargs(shape, kwdargs)
        return ImpulseResponse(np.ones(shape), **kwdargs)

    @staticmethod
    def binauralIR(*args, **kwdargs):
        kwdargs['binaural'] = True
        return ImpulseResponse(*args, **kwdargs)
    
zerosIR = ImpulseResponse.zerosIR        
onesIR = ImpulseResponse.onesIR
delayIR = ImpulseResponse.delayIR        
binauralIR = ImpulseResponse.binauralIR        
        
class TransferFunction(np.ndarray):
    '''
    :class:`TransferFunction` for working with Transfer Function. It is the companion class of :class:`ImpulseResponse` in the frequency domain.
    
    Useful features:
    
    ... listening
    tf.listen()
    ... Spat related stuff
    tf.forsource(i)
    tf.nsources
    ...coordinates related stuff
    tf.forcoordinate
    tf.ncoordinates
    .... dsp stuff
    tf.apply
    ... plotting
    tf.plot_spectrum
    tf.plot

    Is used in the binaural cues module to compute itds/ilds
    
    '''
#    @check_units(samplerate=Hz, duration=second)
    def __new__(cls, data, 
                nfft = None,
                samplerate = None,  # major attributes
                binaural = False, # binaural flag
                coordinates = None, # coordinates attributes
                target_source = None, # spatializer related
                ):
        # Possibly initialize with TransferFunction
        if isinstance(data, ImpulseResponse):
            # that is for sure
            if nfft:
                data_ft = fft(data, nfft = nfft, axis = 0)
            else:
                data_ft = fft(data, axis = 0)
            # then add all relevant attributes
            kwdargs = data.get_kwdargs()
            return TransferFunction(data_ft, **kwdargs)
        if isinstance(data, np.ndarray):
            if samplerate is None:
                samplerate = get_pref('DEFAULT_SAMPLERATE', default = 44100.)*Hz
            x = array(data, dtype = complex)
        else:
            print cls, data
            raise ValueError('Wrong Transfer Function initialization')
        if x.ndim == 1:
            x.shape = (len(x), 1)
        if isinstance(target_source, int):
            target_source = np.array([target_source])
        x = x.view(cls)
        # minimum attributes
        x.samplerate = samplerate
        # flags for different types of TF
        x.binaural = binaural

        x.coordinates = _makecoordinates(data.shape,
                                                coordinates, binaural = binaural)
        x.target_source = _make_target_source(data.shape, 
                                                     target_source, binaural = binaural)
        
        return x
    
    ################## internals
    # util for new object/view creation
    # pass whenever creating a new TransferFunction in this class
    def get_kwdargs(self, _slice = None):
        res = {
            # major attribute
            'samplerate' : self.samplerate,
            # binaural flag
            'binaural' : self.binaural,
            # coordinates attributes
            'coordinates' : self.coordinates,
            # spatializer related            
            'target_source' : self.target_source, 
            }
        if slice is None:
            return res
        else:
            # kwdargs slicing
            # binaural: the tf stays binaural iff all the channels are taken
            if _slice  != None:
                 test = np.zeros(self.shape[1])
                 if not (test[_slice].shape == test.shape):
                     res['binaural'] = False
            # coordinates
            if self._has_coordinates:
                res['coordinates'] = self.coordinates[_slice].flatten()
            # spatializer internals
            if self._has_sources:
                res['target_source'] = self.target_source[_slice].flatten()
            return res

    ################### ndarray internals
    ## NDARRAY stuff
    def __array_wrap__(self, obj, context=None):
        handled = False
        x = np.ndarray.__array_wrap__(self, obj, context)
        if not hasattr(x, 'samplerate') and hasattr(self, 'samplerate'):
            x.samplerate = self.samplerate
        if context is not None:
            ufunc = context[0]
            args = context[1]
        return x

    def __array_finalize__(self,obj):
        if obj is None: return
        # major attribute
        self.samplerate = getattr(obj, 'samplerate', None)
        # binaural flag
        self.binaural = getattr(obj, 'binaural', False)
        # coordinates
        self.coordinates = getattr(obj, 'coordinates', None)
        # spatializer internals
        self.target_source = getattr(obj, 'target_source', None)
        
    ################## SLICING #######################
    # Notes:
    # - getitem is critical because it affects the type of IR
    # review the slicing, it just dosn't work ok anymore
    # always yields non binaural IR
    # - Fancy slicing with times is disabled for now, I may need to
    # rewrite the one in the Sound class, cause its a bit messy
    def __getitem__(self, key):
        return asarray(self).__getitem__(key)
    
        # data = np.ndarray.__getitem__(self, key)
        # if type(key) == int and self.ndim>1:
        #     # data returns a full row of values
        #     data.shape = (1,len(data))
        #     kwdargs = self.get_kwdargs()
        #     return TransferFunction(data, **kwdargs)
        # if data.ndim == 1 and type(key) == (tuple or int):
        #     # we have to see which dim was squeezed
        #     # TODO
        #     if type(key[0]) == int or type(key[0]) == np.float64:
        #         data.shape = (1, len(data))
        #     elif type(key[1]) == int or type(key[1]) == np.float64:
        #         data.shape = (len(data), 1)
        # if type(key) == tuple:
        #     channels = key[1]
        #     kwdargs = self.get_kwdargs(_slice = channels)
        # else:
        #     kwdargs = self.get_kwdargs()
        # return TransferFunction(data, **kwdargs)
    
    def __setitem__(self, key, val):
        asarray(self).__setitem__(key, val)

################## binaural stuff
    @property
    def left(self):
        if not self.binaural:
            return self
        data = asarray(self[:, :self.shape[1]/2], dtype = complex)
        kwdargs = self.get_kwdargs(_slice = slice(None, self.shape[1]/2))
        kwdargs['binaural'] = False
        return TransferFunction(data, **kwdargs)

    @property
    def right(self):
        if not self.binaural:
            return self
        data = asarray(self[:, self.shape[1]/2:], dtype = complex)
        kwdargs = self.get_kwdargs(_slice = slice(self.shape[1]/2, None))
        kwdargs['binaural'] = False
        return TransferFunction(data, **kwdargs)


################## Coordinates
    @property
    def _has_coordinates(self):
        return not (self.coordinates == None)

    @property
    def ncoordinates(self):
        if self._has_coordinates:
            if self.binaural:
                return self.shape[1]/2
            else:
                return self.shape[1]
        else:
            raise AttributeError('Impulse Response doesnt have coordinates')

    def forcoordinates(self, *args):
        '''
        Should return an impulseResponse containing all the rays headed towards source args
        SPEC:
        value peut etre un int -> indexation
        value peut etre une rotation? peut etre pas mal, mais bof NONONON
        value peut etre un tuple (len=2) az, el
        value peut etre une condition: mieux!
        '''
        # argument parsing
        if len(args) == 1 and isinstance(args[0], int):
            # indexing is just the i'th IR
            i = args[0]
            # kwdargs handling
            kwdargs = self.get_kwdargs(_slice = i)
        
            # data collection
            if self.binaural:
                data = np.zeros((self.shape[0], 2))
                data[:, 0] = self.left[:, i].flatten()
                data[:, 1] = self.right[:, i].flatten()
                kwdargs['binaural'] = True
            else:
                data = self[:,i]

            return TransferFunction(data, **kwdargs)

        if len(args) == 1 and callable(args[0]):
            fun = args[0]
            
            idx = []
            for i, (az, el) in enumerate(self.coordinates[:self.shape[1]/2]):
                if fun(az, el):
                    idx.append(i)
            if len(idx)>0:
                idx = np.array(idx)
                data = np.hstack((self[:, idx], self[:, idx + self.shape[1] /2]))

            
                kwdargs = self.get_kwdargs(_slice = idx)
                kwdargs['binaural'] = True
                return TransferFunction(data, **kwdargs)
            else:
                db.log_debug('No HRTF found matching condition')
                return None
            
        if len(args) == 2:
            azs = args[0]
            els = args[1]
            if not iterable(azs):
                azs = [azs]
            if not iterable(els):
                els = [els]
            f = lambda azim, elev: azim in azs and elev in els
            return self.forcoordinates(f)

            
    #################### Spatializer features
    # this includes:
    # attrs:
    # - incoming_direction,source
    # methods:
    # - forsource()
    # flags for different types of TransferFunction
    
    @property
    def _has_sources(self):
        return not (self.target_source == None)
    
    @property
    def nsources(self):
        if self._has_sources:
            return len(np.unique(self.target_source))
        else:
            log_debug('TransferFunction has no target_source, falling back, you should use .shape! or .ncoordinates')
            if self.binaural:
                return self.shape[1]/2
            else:
                return self.shape[1]


    def forsource(self, k):
        '''
        Returns the part of the impulse response that is related to source k
        Only implemented for impulse response that are the result of computations in the spatializer
        '''
        if not self._has_sources:
            raise AttributeError('This method isnt implemented for this type of TF')
        else:
            if (self.target_source != k).all():
                raise ValueError('No data for this source')
            if not self.binaural:
                return self[:, self.target_source == k]
            else:
                data = np.zeros((self.nsamples, 2))
                data[:, 0] = self[:, k].flatten()
                data[:, 1] = self[:,i + self.shape[1]/2].flatten()
                kwdargs = self.get_kwdargs(_slice = k)
#                 kwdargs['binaural'] = self.binaural
                return TransferFunction(data, **kwdargs)

    @property
    def sources(self):
        return np.unique(self.target_source)

    def collapse(self, other = None):
        '''
        An TransferFunction that is the result from Spatializer computations has sources attached to it and possibly multiple rays per source.
        Using collapse on it yields a new TransferFunction with only one TransferFunction per source.
        '''
        res = zerosTF(self.shape[0], 
                      target_source = self.sources, binaural = self.binaural)
        for k, id in enumerate(self.sources):
            chunk = self.forsource(id)
            #chunki = onesTF
            if (np.sum(asarray(chunk.left), axis = 1) ==0).any():
                print 'why?'
            res[:, k] = np.sum(chunk.left, axis = 1).flatten()
            if self.binaural:
                res[:, k+self.nsources/2] = np.sum(chunk.right, axis = 1)
        return res

    ################### DSP Features
    # -listening of TFs
    # -applying to sounds
    def listen(self, sound = None, sleep = True, reference = True):
        '''
        If impulse response is more than one binaural or monaural,
        then all the sounds are played in a sequence
        
        TDO: WRITE DOC
        '''
        if sound is None:
            soundfile = prefs.get_pref('DEFAULT_SOUND', default = -1)
            if not soundfile == -1:
                sound = Sound.load(soundfile)
            else:
                sound = pinknoise(500*ms, samplerate = self.samplerate)
        else:
            if type(sound) == str:
                sound = Sound.load(sound)

        if reference:
            db.log_debug('Listening to original sound')
            sound.atlevel(60*dB).play(sleep = sleep)

        if self._has_coordinates:
            for i in range(self.ncoordinates):
                out = self.forcoordinates(i).apply(sound)
                db.log_debug('Listening to TF for coordinates '+str(self.coordinates[i]))
                out.atlevel(60*dB).play(sleep = sleep)
        else:
            out = self.apply(sound)
            db.log_debug('Listening to TF')
            out.atlevel(60*dB).play(sleep = sleep)

    def apply(self, other, outlevel = 60*dB):
        '''
        Applies the current Impulse responses to a sound.
        Either yields a single sound in the case where the Impulse response is only relative to a single source. Otherwise outputs a list of sounds for each source.
        
        Gain normalisation (outlevel kwdarg): by default the gain is adjusted to that the loudest channel is at the level specified by outlevel. Hence the difference in level between the two channels is preserved. This is a different behavior from the one in brian.hears.sounds        
        '''
        ir = ImpulseResponse(self)
        return ir.apply(other, outlevel = outlevel)

    def smooth(self, b = .54, octavefraction = 1./3):
        res = np.zeros_like(self)
        for channel in range(self.nchannels):
            x = self[:,channel]
            amps = nonuniform_spectralsmoothing(np.abs(x),
                                                self.samplerate, width = octavefraction, b = b)
            phasepart = np.exp(1j*np.angle(x))
            res[:,channel] = amps*phasepart
        kwdargs = self.get_kwdargs()
        return TransferFunction(res, **kwdargs)

    
######################### Plotting
    def plot(self, dB = True, display = False, label = ''):
        '''
        Plots the spectrum of the TF
        TODO: whould simply call TF().plot()
        '''
        conv = dBconv
        if not dB:
            conv = lambda x: x
        
        offset = 0
        if argmin(self) == 0:
            offset = 1
        subplot(211)
        semilogx(self.freqs[offset:], conv(self.amplitudes[offset:,:]), label = label)
        xlabel('Frequency (Hz)')
        if dB:
            ylabel('Amplitude (dB)')
            ax = list(axis())
            ax[2] = -100
            axis(ax)
        else:
            ylabel('Amplitude (gain)')
            ylim(0, 1)
        subplot(212)
        semilogx(self.freqs[offset:], np.unwrap(self.phases[offset:,:]), label = label)
        ylabel('Phase (rad)')
        xlabel('Frequency (Hz)')
        ylim(-np.pi, np.pi)
        if label != '':
            legend()
        if display:
            show()
    

    def ild(self, indices = None):
        '''
        Returns the interaural level difference in dB
        '''
        if indices is None:
            return dBconv(self.left/self.right)
        else:
            return dBconv(self.left[indices,:]/self.right[indices,:])


    def itd(self, indices = None):
        '''
        Returns the interaural phase delay in ms
        '''
        if indices is None:
            return np.nan_to_num(1e3*(np.unwrap(np.angle(self.left/self.right), axis = 0)/(2*np.pi*self.freqs.reshape((self.nfft, 1)))))
        else:
            curfreqs = self.freqs[indices]
            return np.nan_to_num(1e3*(np.unwrap(np.angle(self.left[indices,:]/self.right[indices,:]), axis = 0)/(2*np.pi*curfreqs.reshape((len(curfreqs), 1)))))


    def igd(self, indices = None):
        '''
        Returns the interaural group delay in ms,
        only for positive frequencies by default (because of the diff)
        '''
        curfreqs = self.freqs[self.freqs > 0]
        deltafs = np.tile(np.atleast_2d(np.diff(curfreqs, axis = 0)).T, (1, self.shape[1]/2))
        tmp = np.nan_to_num(1e3 * np.diff(np.unwrap(self.ipd(indices = self.freqs > 0), axis = 0), axis = 0))/(2*np.pi*deltafs)
        
        return tmp[indices[self.freqs > 0][:-1]]

    def idi(self, indices = None):
        '''
        Returns the interaural diffraction index in radians
        '''
        curfreqs = self.freqs[self.freqs > 0]

        curfreqs = np.tile(np.atleast_2d(curfreqs).T, (1, self.shape[1]/2))

        tmp =  np.mod(curfreqs[:-1] * (self.itd(indices = self.freqs > 0)[:-1] - self.igd(indices = self.freqs > 0)) * 2 * np.pi / 1000. + np.pi, 2 * np.pi) - np.pi
        
        return tmp[indices[self.freqs > 0][:-1]]

    def ipd(self, indices = None):
        '''
        Returns the interaural phase difference in radians
        '''
        if indices is None:
            return np.mod(np.angle(self.left/self.right) + np.pi, 2*np.pi) - np.pi
        else:
            curfreqs = self.freqs[indices]
            return np.mod(np.angle(self.left[indices,:]/self.right[indices,:]) + np.pi, 2*np.pi) - np.pi

    def plot_binaural_cues(self, frange = (500, 20000), display = False, smoothing = 0.):
        '''
        Plots the binaural cues
        '''
        if not self.binaural:
            raise AttributeError('Cannot plot binaural cues of non-binaural TF')
        posfreqs_idx = (self.freqs>0)*(self.freqs > frange[0])*(self.freqs<frange[1])
        posfreqs = self.freqs[posfreqs_idx]
        
        subplot(211)
        ilds = self.ild(indices = posfreqs_idx)
        ax = gca()
        color = np.linspace(0, 1., ilds.shape[1])
        for kpos in range(ilds.shape[1]):
            if smoothing != 0:
                ax.semilogx(posfreqs, octaveband_smoothing(ilds[:,kpos], smoothing), color = cmap.jet(color[kpos]))
            else:
                ax.semilogx(posfreqs, ilds[:,kpos], color = cmap.jet(color[kpos]))

        # for kposition in xrange(self.ncoordinates):
        #     print ilds[1, kposition]
        #     annotate('%.1f' % (self.coordinates['azim'][kposition]),
        #              xy = (0, ilds[1, kposition]))
            
        xlabel('Frequency (Hz)')
        ylabel('ILD (dB)')

        xlim(posfreqs[0], posfreqs[-1])
        
        subplot(212)
        itds = self.itd(indices = posfreqs_idx)
        ax = gca()
        for kpos in range(ilds.shape[1]):
            if smoothing != 0:
                ax.semilogx(posfreqs, octaveband_smoothing(itds[:,kpos], smoothing), color = cmap.jet(color[kpos]))
            else:
                ax.semilogx(posfreqs, itds[:,kpos], color = cmap.jet(color[kpos]))

        xlabel('Frequency (Hz)')
        ylabel('ITD (ms)')
        xlim(posfreqs[0], posfreqs[-1])

        if display:
            show()
        
    def convolve(self, other):
        ir = ImpulseResponse(self)
        return ir.convolve(other)

    # ndarray stuff
    def copy(self):
        kwdargs = self.get_kwdargs()
        data = np.ndarray.copy(self)
        return TransferFunction(data, **kwdargs)

    ##### TODO : check why repr and str don't work without this
    def __str__(self):
        return asarray(self, dtype = complex).__str__()

    def __repr__(self):
        return asarray(self, dtype = complex).__repr__()

    # global properties
    @property
    def nchannels(self):
        return self.shape[1]
    
    @property
    def duration(self):
        return (self.shape[0] / self.samplerate)
    
    @property
    def nfft(self):
        return self.shape[0]

    @property
    def freqs(self):
        return fftfreq(self.nfft)*self.samplerate
    
    @property
    def fbin(self):
        return self.samplerate/self.nfft
    
    @property
    def amplitudes(self):
        return np.abs(self)
    
    @property
    def phases(self):
        return np.angle(self)

    # load/save
    def save(self, filename):
        '''
        Uses pickle.dump on ``filename`` to dump the current ir.
        '''
        f = open(filename, 'wb')
        pickle.dump(self, f)
        f.close()

    @staticmethod
    def load(filename):
        '''
        Uses pickle.load on ``filename`` to load a pickled ir.
        '''
        f = open(filename, 'rb')
        res = pickle.load(f)
        f.close()
        return res
    
    # constructors
    @staticmethod
    def zerosTF(shape, **kwdargs):
        shape = _shape_from_kwdargs(shape, kwdargs)
        return TransferFunction(np.zeros(shape), **kwdargs)

    @staticmethod
    def delayTF(*args, **kwdargs):
        return TransferFunction(ir.delayIR(*args, **kwdargs))
    
    
    @staticmethod
    def onesTF(shape, **kwdargs):
        shape = _shape_from_kwdargs(shape, kwdargs)
        return TransferFunction(np.ones(shape), **kwdargs)

    @staticmethod
    def binauralTF(*args, **kwdargs):
        kwdargs['binaural'] = True
        return TransferFunction(*args, **kwdargs)

######## Yes do that
zerosTF = TransferFunction.zerosTF        
onesTF = TransferFunction.onesTF
delayTF = TransferFunction.delayTF        
binauralTF = TransferFunction.binauralTF

######################################## UTILS

def dur2sample(duration, samplerate):
    return int(rint(duration*samplerate))

def sample2dur(n, samplerate):
    return float(n)/samplerate

def isduration(x):
    return units.have_same_dimensions(x, second)

def getnsample(x, samplerate):
    if not (x is None) and isduration(x):
        return dur2sample(x, samplerate)
    else:
        return x

def ensure_Hz(samplerate):
    if units.is_dimensionless(samplerate):
        return samplerate*Hz
    else:
        return samplerate
        

########### coordinates, target_source
def _makecoordinates(shape, coords, binaural = False):
    # utility to construct the coordinate array
    if coords is None:
        return None
    
    if len(shape) == 1:
        shape = (shape[0], 1)
    
    dtype_coords = [('azim','f8'), ('elev','f8')]
    
    if type(coords) == tuple:
        coords = np.array([coords], dtype = dtype_coords).reshape((1,1))

    n = len(coords)
    if binaural and n == shape[1]/2:
        coordinates = np.zeros(2*n, dtype = dtype_coords)
        coordinates[:n] = coords
        coordinates[n:] = coords
    else:
        if n == shape[1]*2:
            coordinates = np.zeros(shape[1], dtype = dtype_coords)
            coordinates = coords[:n/2]
        else:
            coordinates = np.zeros(n, dtype = dtype_coords)
            coordinates[:n] = coords


    if coordinates.shape[0] != shape[1]:
        if not binaural:
            raise ValueError('Couldnt retrieve the right number of coordinates '+str(coordinates.shape)+' vs. '+str(shape))
        else:
            pass
#            if (coordinates[:n] != coordinates[n:]).all():
#                raise ValueError('Couldnt retrieve the right number of coordinates, coordinates not repeated!')
    return coordinates

def _make_target_source(shape, ts, binaural = False):
    if ts is None:
        return None

    if isinstance(ts, np.ndarray) or isinstance(ts, list):
        n = len(ts)
    else:
        n = 1

    if len(shape) == 1:
        shape = (shape[0], 1)

    if binaural and n == shape[1]/2:
        target_source = np.zeros(2*n)
        target_source[:n] = ts
        target_source[n:] = ts
    else:
        target_source = np.zeros(n)
        target_source[:n] = ts

    if target_source.shape[0] != shape[1]:
        if not binaural:
            print 'got ', target_source.shape[0]
            print 'expected', shape[1]
            raise ValueError('Couldnt retrieve the right number of target_source')
        else:
            pass
#            if (target_source[:n] != target_source[n:]).all():
#                raise ValueError('Couldnt retrieve the right Number Of target_source, target_source not repeated!')

    return target_source

def _shape_from_kwdargs(shape, kwdargs):
    # this is used for the initialization through staticmethods.
    # it guesses the shape out of the provided shape (which might be a
    # single Quantity)
    #
    ###### dimension 0
    if not type(shape) == tuple:
        shape = (shape, 1)

    shape = list(shape)
    if isinstance(shape[0], Quantity):
        if not units.have_same_dimensions(shape[0], second):
            raise DimensionMismatchError('Impulse response length must be specified in samples or seconds')
        else:
            try:
                
                samplerate = ensure_Hz(kwdargs['samplerate'])
            except KeyError:
                samplerate = prefs.get_pref('DEFAULT_SAMPLERATE', default = 44100*Hz)
        shape[0] = round(shape[0] * samplerate)
    
    ###### dimension 1
    n = shape[1]
    if 'coordinates' in kwdargs:
        if type(kwdargs['coordinates']) == tuple:
            n = 1
        else:
            n = len(kwdargs['coordinates'])
    if 'target_source' in kwdargs:
        n = len(kwdargs['target_source'])
    if 'binaural' in kwdargs:
        if kwdargs['binaural']:
            n *= 2
    shape[1] = n

    shape = tuple(shape)
    return shape
