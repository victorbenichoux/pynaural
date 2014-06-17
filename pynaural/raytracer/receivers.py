from pynaural.raytracer.geometry.base import Point, Vector, FRONT, BACK, LEFT, RIGHT, UP, DOWN, ORIGIN
from pynaural.utils.debugtools import log_debug


import scenes 
#import spatializer.utils.debugtools as db
import numpy as np
import scipy as sp


__all__ = ['Receiver', 'HeadReceiver',
           'OrientedReceiver',
           'ArrayReceiver',
           'HRTFReceiver', 'SphericalHeadReceiver',
           'IRCAMSubjectReceiver', 'IRCAMInterpolatingReceiver']

########################### Receivers ##################

class Receiver(object):
    '''
    Parent class for receiver objects. Unless subclassed, it only requires a position attribute.
    '''
    def __init__(self, loc):
        if isinstance(loc, Point):
            self.position = loc
        elif isinstance(loc, Vector):
            self.position = Point(loc)
        else:
            raise TypeError('Position wasnt point or vector, '+str(loc))
    
    def __str__(self):
        return 'Receiver:\n'+str(self.position)+'\n'
    
    def getSphericalBeam(self, n):
        return SphericalBeam(self.position, n)
    
    def computeIRs(self, *args, **kwdargs):
        scene = args[0]
        rest = list(args)[1:]
        return scene.computeIRs(self, *rest, **kwdargs)

    @property
    def binaural(self):
        return False

    def get_positions(self):
        return self.position.array()

############# Parent classes for HRTF integration

class OrientedReceiver(Receiver):
    '''
    OrientedReceiver is just a regular receiver but oriented. This provides the reference for the azimuth and elevation in the HRTF computations
    '''
    def __init__(self, position, orientation):
        super(OrientedReceiver,self).__init__(position)
        if not isinstance(orientation, Vector):
            raise ValueError('Orientation should be specified as a Vector object')
        self.orientation = orientation/orientation.norm()

    def placeSource(self, distance, rotation):
        '''
        Yields a source at a certain distance, az and el relative to the oriented receiver object.
        '''

        return SphericalSource(sourceposition)

########################## HRTF enabled receiver #################

class HRTFReceiver(OrientedReceiver):
    '''
    A single-point receiver that uses HRTF to finally filter IRs. The
    position is the head center.

    ** Initialization ** 

    ``position`` can be a Quantity or numeric object. If so the
    Receiver is positioned at position*UP. Alternatively it can be
    just a Vector or Point giving the real position.

    `` orientation = FRONT `` Orientation of the Receiver object. 
    
    ** Methods ** 
    
    .. automethod :: get_beam
    
    ** Advanced **
    
    An HRTFReceiver has to implement the ``get_hrir'' method.
    ``get_hrir(az, el [,distance])'' returns the hrir pair for the given direction.
    
    Note that if the ``distance'' argument is supported by ``get_hrir'' then the ``is_distancedependent'' property must be set to True.
    '''

    def __init__(self, position, orientation = FRONT, is_distancedependent = False):
        if isinstance(position, (Quantity, float, int)):
            position = position*UP
        
        OrientedReceiver.__init__(self, position, orientation)
        
        self.samplerate = self.hrtfset.samplerate
        
        self.is_distancedependent = is_distancedependent
        
    def get_beam(self):
        '''
        Returns a beam centered at the position with directions
        through all available coordinates.
        TODO
        '''
        pass
    
    def computeHRTF(self, *args, **kwdargs):
        '''
        Should return an array containing HRTFs for each beam direction.
        Should be overriden by more specialized classes
        
        Note: this is the "vectorized" version of get_hrir and is not used for the moment
        TODO
        '''
        pass

    def get_hrir(self, az, el):
        return self.hrtfset.get_hrir(az, el)

    def computeHRIRs(self, *args, **kwdargs):
        '''
        Used to fetch HRIRs relative to a result from an AcousticScene, 
        
        can be used either with a scene + beam (result of rendering)
        or simply with a list of coordinates of the form (az,el) where
        az and el are in my convention, but for that rather use get_hrir
        
        also the method kwargs can switch between interpolation
        (bilinear) or closest behavior, 
        '''
        if len(args) == 2 or isinstance(args[0], Beam):
            beam = args[0]
            beam = beam[beam.get_reachedsource_index()]
            print spatdb.log_debug.__module__
            spatdb.log_debug('Fetching HRTFs')
            if isinstance(self.hrtfset, HRTFModel):
                data = np.zeros((self.hrtfset.nfft, beam.nrays*2))
            elif hasattr(self, 'nsamples'):
                data = np.zeros((self.nsamples, beam.nrays*2))
            else:
                data = np.zeros((len(self.hrtfset[0].left), beam.nrays*2))

            # preparing target_source, coordinates
            target_source = np.zeros(beam.nrays, dtype = int)
            # coordinates
            coordinates = np.zeros(beam.nrays, dtype = [('azim','f8'),('elev','f8')])
            d, az, el = cartesian2spherical(beam.directions, unit = 'deg')

            coordinates['azim'] = az
            coordinates['elev'] = el
            for i in range(beam.nrays):
                if self.is_distancedependent:#isinstance(self.hrtfset, HRTFModel):
#                    print 'Is, indeed, distancedependent (in receivers)'
#                    print 'az, el, d', (az[i], el[i], d[i])
                    cur_hrir = self.get_hrir(az[i], el[i], d[i])
                else:
                    cur_hrir = self.get_hrir(az[i], el[i])
                data[:, i] = cur_hrir.left.flatten()
                data[:, i + beam.nrays] = cur_hrir.right.flatten()
                target_source[i] = beam.target_source[i]

            spatdb.log_debug('Finished fetching HRTFs')
            HRIRs = ImpulseResponse(data,
                samplerate = self.samplerate,
                binaural = True,
                coordinates = coordinates,
                target_source = target_source)
            return HRIRs
        elif len(args) == 1:
            coords = args[0]
            # TODO
            # do something better and move it somewhere else
            coordinates = _makecoordinates(coords, binaural = True)
            leftdata = np.zeros((len(self.hrtfset[0].left),
                                  len(coords)))
            rightdata = np.zeros((len(self.hrtfset[0].right),
                                  len(coords)))
            for i,(az, el) in enumerate(coords):
                leftdata[:,i] = self.hrtfset.get_hrir(az, el, method =
                                                      method).left.flatten()
                rightdata[:,i] = self.hrtfset.get_hrir(az, el, method =
                                                       method).right.flatten()
            data = np.hstack((leftdata, rightdata))
            HRIRs = ImpulseResponse(data,
                samplerate = self.samplerate,
                binaural = True,
                coordinates = coordinates)
            return HRIRs

    def computeIRs(self, *args, **kwdargs):
        '''
        Returns the scene response convolved by the hrtfs.
        depends on the form of the HRTFs.
        
        Computes the IRs from the scene given as a first argument, and convolves it correctly with the Receiver.
        
        Sample usage:
        receiver.computeIRs(scene, 
        '''
        if len(args) == 1 and isinstance(args[0], Beam):
            beam = args[0]
        else:
            scene = args[0]
            args = args[1:]
            beam = scene.render(self)
            beam = beam[beam.get_reachedsource_index()]
        
        newargs = [None]*(len(args)+1)
        newargs[1:] = args
        newargs[0] = beam
        
        HRIRs = self.computeHRIRs(beam)
        spatdb.log_debug('HRIRs are of length (%i)' % (HRIRs.shape[0]))
        
        if isinstance(scene, scenes.VoidScene):
            # in this case nothing has to be done with the environment,
            # so we just output the HRTFs
            # Hey maybe not! this shouldn't be used to fetch hrtfs, maybe something directly linked to the receiver (another method) would be better
            return HRIRs

        kwdargs['binaural'] = False

        kwdargs['collapse'] = False
        
        IRs = scene.computeIRs(*newargs, **kwdargs)

        # if isinstance(scene, SimpleScene):
        #     kwdargs['binaural'] = False
        # else:
        IRs = np.tile(asarray(IRs), (1, 2))

        print IRs.shape

        spatdb.log_debug('Scene IRs are of length (%i)' % (IRs.shape[0]))
        spatdb.log_debug('Convoluting '+str(beam.nrays)+' scene responses with HRIRs')

        ir_offset0 = np.min(np.argmin(1.0*(np.abs(IRs) < 1e-10), axis = 0))
        ir_offset1 = IRs.shape[0]-np.min(np.argmin(1.0*(np.abs(IRs[::-1,:]) < 1e-10), axis = 0)) + 1
        if ir_offset0 == ir_offset1 - 2:
            spatdb.log_debug('Environment impulse response is just one delay + gain')
            # simple case, with only a delay and possibly a gain, so we just multiply
            gains = np.tile(IRs[ir_offset0, :].reshape(1, IRs.shape[1]), (HRIRs.shape[0], 1))
            convolution = HRIRs * gains
        else:
            # we do a real linear convolution, mostly because lengths never match
            # it is costly so we try and trim the scene IRs that are quite often sparse
            nir = ir_offset1 - ir_offset0
            N =  nir + HRIRs.shape[0] - 1

            IRs_padded = np.vstack((IRs[ir_offset0:ir_offset1, :], 
                                    np.zeros((N - nir, IRs.shape[1]))))
            HRIRs_padded = np.vstack((HRIRs, 
                                      np.zeros((IRs_padded.shape[0] - HRIRs.shape[0], HRIRs.shape[1]))))

            convolution = np.zeros(HRIRs_padded.shape, dtype = complex)
            convolution = ifft(fft(IRs_padded, axis = 0)*fft(HRIRs_padded, axis = 0), axis = 0).real

        res = vstack((zeros((max(ir_offset0-1,0), 2*beam.nrays)), convolution))
        spatdb.log_debug('Collapsing final responses')
        
        if scene.nsources > 1:
            # More than one source
            # TODO 
            allhrirs = zerosIR((res.shape[0], 2*scene.nsources), binaural = True)
            relativecoordinates = []
            target_source = unique(HRIRs.target_source)
            for i in range(scene.nsources):
                relativecoordinates.append(scene.sources[i].getRelativeSphericalCoords(
                        self, unit = 'deg'))
                allhrirs[:, i] = np.sum(res[:, i*beamspersource:(i+1)*beamspersource], axis = 1)/float(beamspersource)#left
                allhrirs[:, i+scene.nsources] = np.sum(res[:, beam.nrays+i*beamspersource:beam.nrays+(i+1)*beamspersource], axis = 1)/float(beamspersource)#right
            if np.isnan(allhrirs).any():
                spatdb.log_debug('Output of getIRs will containt nans')
            allhrirs.target_source = target_source
            allhrirs.coordinates = relativecoordinates
            return allhrirs
        
        else:
            # exactly one source
            left = np.sum(res[:,:2], axis = 1)/beam.nrays
            right = np.sum(res[:,2:], axis = 1)/beam.nrays
            data = np.hstack((
                    left.reshape((len(left),1)), right.reshape((len(right), 1))
                                                               ))
            coordinates = scene.sources[0].getRelativeSphericalCoords(self)
            return ImpulseResponse(data, 
                                   binaural = True,
                                   samplerate = HRIRs.samplerate,
                                   target_source = scene.sources[0].get_id(), coordinates = coordinates[1:])
                                   

    @property
    def binaural(self):
        return True

################### IRCAM receivers #################################
# One of them is exact (IRCAMSubjectReceiver) no interpolation...
# One of them is interpolated IRCAMInterpolatedRreceiver
# TODO: rethink/rewrite HRTFSets so that the only diff between the two is that.
        
class IRCAMSubjectReceiver(HRTFReceiver):
    '''
    Is used to place a know subject of the IRCAM database in a scene.
    Initialized with the subject number, height, inter aural difference and orientation.
    You must instanciate the IRCAMpath preference variable for it to work.
    '''
    def __init__(self, height, orientation = FRONT, subject = None):
        if subject is None:
            subject = prefs.get_pref('IRCAM_DEFAULTSUBJECT', default = -1)
            if subject == -1:
                raise AttributeError('You didn\'t specify a subject, maybe you should instantiat IRCAM_DEFAULTSUBJECT in localprefs.py')

        IRCAMpath = spatprefs.get_pref('IRCAMpath', default = '')

        hrtfset = IRCAM_LISTEN(IRCAMpath, 
                              compensated = False).load_subject(subject)

        self.hrtfset = NewNewInterpolatingHRTFSet(hrtfset,
                                                  interpolation = 'closest')

        HRTFReceiver.__init__(self, height, orientation = orientation)


    def get_normalization_factors(self):
        '''
        Returns the normalisation factors alphaleft alpharight so that all the filters on the left (resp. right) have a gain equals to 1. 
        
        The gain of a filter being defined as the sqrt of the sum of the square of the amplitudes of the tf.
        '''
        tfs_left = self.hrtfset.data[0,:,:]
        tfs_left = fft(tfs_left, axis = 1)
        gains_left = np.sum(np.abs(tfs_left), axis = 1)
        gain_left = self.nfft/np.max(gains_left)

        tfs_right = self.hrtfset.data[1,:,:]
        tfs_right = fft(tfs_right, axis = 1)
        gains_right = np.sum(np.abs(tfs_right), axis = 1)
        gain_right = self.nfft/np.max(gains_right)
        return gain_left, gain_right
        
    def get_hrir(*args, **kwdargs):
#        kwdargs['method'] = 'closest'
        return HRTFReceiver.get_hrir(*args, **kwdargs)
        
    ################# Utils for relative coordinates
    # Applies to OrientedReceiver

    def get_coordinates(self, coordsfilter):
        res = []
        for (az, el) in self.hrtfset.compensatedcoordinates:
            if coordsfilter(az, el):
                res.append((az, el))
        return res

    def plot_forEl(self, el, nfft = 8192, fig = None, display = True):
        coordsfilter = lambda azim, elev: elev==el
        coords = self.get_coordinates(coordsfilter)
        HIRs = self.computeHRIRs(coords)
        azs = np.array([az for (az,el) in coords])
        els = coords[0][1]
        freqs = fftfreq(nfft) * self.hrtfset.samplerate
        
        leftHRTFs = np.zeros((nfft/2, HRIRs.shape[1]/2))
        rightHRTFs = np.zeros((nfft/2, HRIRs.shape[1]/2))
        for i in range(HRIRs.shape[1]/2):
            leftHRTFs[:,i] = np.abs(fft(HRIRs[:,i].flatten(), n = nfft))[:nfft/2]
            rightHRTFs[:,i] = np.abs(fft(HRIRs[:,HRIRs.shape[1]/2+i].flatten(), n = nfft))[:nfft/2]
        figure(fig)
        ax1 = subplot(211)
        ax1.set_title('Left HRTFs, el='+str(el)+' deg')
#        ax = imshow(leftHRTFs.T, aspect = 'auto', interpolation
#        ='nearest')
        pcolor(freqs[:nfft/2], azs , leftHRTFs.T)
#        imshow(azs, freqs, leftHRTFs.T)
        ylabel('Azimuth')
        xlabel('Frequency (Hz)')
        colorbar()
        ax2 = subplot(212)
        ax1.set_title('Right HRTFs, el='+str(el)+' deg')
        imshow(rightHRTFs.T, aspect = 'auto')
        ylabel('Azimuth')
        xlabel('Frequency (Hz)')
#        imshow(freqs, azs, rightHRTFs.T)
        colorbar()
        if display:
            show()

    @property
    def binaural(self):
        return True

    @property
    def nfft(self):
        return self.hrtfset.data.shape[2]

class IRCAMInterpolatingReceiver(HRTFReceiver):
    '''
    Implements a spatializer counter part of the hrtfsets in Brian.
    Is used to place a know subject of the IRCAM database in a scene.
    Initialized with the subject number, height, inter aural difference and orientation.
    You must instantiate the IRCAMpath preference variable for it to work.
    
    has an attribute hrtfset from which it gets the hrirs through hrtfset.get_hrir(az, el, distance = ...)
    '''
    def __init__(self, height, orientation = FRONT, subject = None, 
                 compensated = False):
        if subject is None:
            subject = get_pref('IRCAM_DEFAULTSUBJECT', default = -1)
            if subject == -1:
                raise AttributeError('You didn\'t specify a subject, maybe you should instantiat IRCAM_DEFAULTSUBJECT in localprefs.py')
        IRCAMpath = spatprefs.get_pref('IRCAMpath', default = '')

        hrtfset = IRCAM_LISTEN(IRCAMpath, 
                              compensated = compensated).load_subject(subject)
        self.hrtfset = NewNewInterpolatingHRTFSet(hrtfset,
                                                  interpolation = 'linear')

        HRTFReceiver.__init__(self, height, orientation = orientation)

    @property
    def binaural(self):
        return True

######################## Dummy binaural receiver
class HeadReceiver(OrientedReceiver):
    '''
    Initialized with the height of the head center and interaural distance.
    Is a simple two-points receiver that represents an empty head.
    
    ** Initialization ** 
    
    '''
    def __init__(self, height, iad, orientation = FRONT):
        position = Point(height * UP)
        OrientedReceiver.__init__(self, position, orientation)
        self.iad = iad

    def getEarPosition(self, whichone):
        d = UP.vectorial_product(self.orientation)#vector from center to left ear
        if whichone == 'left':
            return Point(self.iad/2.0*d+self.position)
        elif whichone == 'right':
            return Point(-self.iad/2.0*d+self.position)
        else:
            ValueError('Fetched ear position must be left or right, it was '+str(whichone))
    
    def getSphericalBeam(self, n):
        '''
        Returns a spherical beam with 2*n rays, the first half coming from the left ear and the scond from the right ear.
        '''
        b1 = SphericalBeam(self.getEarPosition('left'), n)
        b2 = SphericalBeam(self.getEarPosition('right'), n)
        return b1.cat(b2)

    @property
    def binaural(self):
        return True

    def get_positions(self):
        d = UP.vectorial_product(self.orientation)#vector from center to left ear
        positions = np.tile(self.position.array(), (1, 2))
        positions[:, 0] += self.iad/2.0*d.array().flatten()
        positions[:, 1] -= self.iad/2.0*d.array().flatten()
        return positions
    
    


########################## Multiple receiver #################

class ArrayReceiver(Receiver):
    def __init__(self, positions):
        self.positions = positions
        raise NotImplementedError
    
    @property
    def npositions(self):
        return self.positions.shape[1]
    
    def get_positions(self):
        return self.positions

################### Spherical Model Receiver ########################

class SphericalHeadReceiver(HRTFReceiver, HeadReceiver):
    def __init__(self, height, iad, earelevation = 5, subject = None, 
                 samplerate = 44100., nfft = 1024):
        self.hrtfset = SphericalHead(iad, earelevation,
                                     samplerate = samplerate, nfft = nfft)
        HeadReceiver.__init__(self, height, iad)
        HRTFReceiver.__init__(self, height)

