from pynaural.raytracer.geometry.rays import SphericalBeam, Beam
from pynaural.raytracer.geometry.base import Point, Vector, FRONT, BACK, LEFT, RIGHT, UP, DOWN, ORIGIN, \
    cartesian2spherical
from pynaural.signal.impulseresponse import ImpulseResponse, _makecoordinates, zerosIR
from pynaural.signal.sphericalmodel import SphericalHead
from pynaural.utils.spatprefs import get_pref
from pynaural.utils.debugtools import log_debug
from pynaural.io.hrtfs.ircam import ircamHRIR
import numpy as np


__all__ = ['Receiver',
           'OrientedReceiver',
           'HRTFReceiver', 'SphericalHeadReceiver',
           'IRCAMSubjectReceiver']

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
            raise TypeError("Position was neither point nor vector, "+str(loc))
    
    def __str__(self):
        return 'Receiver:\n'+str(self.position)+'\n'
    
    def getSphericalBeam(self, n):
        return SphericalBeam(self.position, n)
    
    def computeIRs(self, *args, **kwdargs):
        scene = args[0]
        rest = list(args)[1:]
        return scene.computeIRs(self, *rest, **kwdargs)

    def get_positions(self):
        return self.position.array()

# Parent classes for HRTF integration
class OrientedReceiver(Receiver):
    '''
    OrientedReceiver is just a regular receiver but oriented. This provides the reference for the azimuth and elevation in the HRTF computations
    '''
    def __init__(self, position, orientation):
        super(OrientedReceiver,self).__init__(position)
        if not isinstance(orientation, Vector):
            raise ValueError('Orientation should be specified as a Vector object')
        self.orientation = orientation/orientation.norm()

# HRTF enabled receiver
class HRTFReceiver(OrientedReceiver):
    '''
    A single-point receiver that uses HRTF to finally filter IRs. The
    position is the head center.

    ** Initialization ** 

    ``position`` can be a Quantity or numeric object. If so the
    Receiver is positioned at position*UP. Alternatively it can be
    just a Vector or Point giving the real position.

    `` hrtfset `` An ImpulseResponse object containing the HRIR data

    `` orientation = FRONT `` Orientation of the Receiver object.

    
    ** Methods ** 
    
    .. automethod :: get_beam
    
    ** Advanced **
    
    An HRTFReceiver has to implement the ``get_hrir'' method.
    ``get_hrir(az, el [,distance])'' returns the hrir pair for the given direction.
    
    Note that if the ``distance'' argument is supported by ``get_hrir'' then the ``is_distancedependent'' property must be set to True.
    '''

    def __init__(self, position, hrtfset, orientation = FRONT, is_distancedependent = False):
        if isinstance(position, (float, int)):
            position = position*UP
        
        OrientedReceiver.__init__(self, position, orientation)

        self.hrtfset = hrtfset
        self.samplerate = self.hrtfset.samplerate
        self.nsamples = self.hrtfset.nsamples

        self.coordinates = self.hrtfset.coordinates

        self.is_distancedependent = is_distancedependent

    def get_hrir(self, az, el):
        idmin = get_closest_coords(self.coordinates, (az, el), retval = 'idmin')
        hrir = self.hrtfset.forcoordinates(idmin)
        return hrir

    def collapse(self, irs):
        data = np.zeros((self.nsamples + irs.nsamples - 1, 2))
        ncoordinates = irs.ncoordinates
        for k in xrange(ncoordinates):
            az, el = irs.coordinates['azim'][k], irs.coordinates['elev'][k]
            hrir = self.get_hrir(az, el)
            cur_ir = irs.forcoordinates(k).apply(hrir)
            data[:,0] += cur_ir[:,0].flatten()
            data[:,1] += cur_ir[:,1].flatten()
        return ImpulseResponse(data, binaural = True)

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
            print log_debug.__module__
            log_debug('Fetching HRTFs')
            if hasattr(self, 'nsamples'):
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
                if self.is_distancedependent:
                    cur_hrir = self.get_hrir(az[i], el[i], d[i])
                else:
                    cur_hrir = self.get_hrir(az[i], el[i])
                data[:, i] = cur_hrir.left.flatten()
                data[:, i + beam.nrays] = cur_hrir.right.flatten()
                target_source[i] = beam.target_source[i]

            log_debug('Finished fetching HRTFs')
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
        log_debug('HRIRs are of length (%i)' % (HRIRs.shape[0]))

        kwdargs['binaural'] = False

        kwdargs['collapse'] = False
        
        IRs = scene.computeIRs(*newargs, **kwdargs)

        # if isinstance(scene, SimpleScene):
        #     kwdargs['binaural'] = False
        # else:
        IRs = np.tile(np.asarray(IRs), (1, 2))

        print IRs.shape

        log_debug('Scene IRs are of length (%i)' % (IRs.shape[0]))
        log_debug('Convoluting '+str(beam.nrays)+' scene responses with HRIRs')

        ir_offset0 = np.min(np.argmin(1.0*(np.abs(IRs) < 1e-10), axis = 0))
        ir_offset1 = IRs.shape[0]-np.min(np.argmin(1.0*(np.abs(IRs[::-1,:]) < 1e-10), axis = 0)) + 1
        if ir_offset0 == ir_offset1 - 2:
            log_debug('Environment impulse response is just one delay + gain')
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
            convolution = np.ifft(np.fft.fft(IRs_padded, axis = 0)*np.fft.fft(HRIRs_padded, axis = 0), axis = 0).real

        res = np.vstack((np.zeros((max(ir_offset0-1,0), 2*beam.nrays)), convolution))
        log_debug('Collapsing final responses')
        
        if scene.nsources > 1:
            # More than one source
            # TODO 
            allhrirs = zerosIR((res.shape[0], 2*scene.nsources), binaural = True)
            relativecoordinates = []
            target_source = np.unique(HRIRs.target_source)
            for i in range(scene.nsources):
                relativecoordinates.append(scene.sources[i].getRelativeSphericalCoords(
                        self, unit = 'deg'))
                allhrirs[:, i] = np.sum(res[:, i*beamspersource:(i+1)*beamspersource], axis = 1)/float(beamspersource)#left
                allhrirs[:, i+scene.nsources] = np.sum(res[:, beam.nrays+i*beamspersource:beam.nrays+(i+1)*beamspersource], axis = 1)/float(beamspersource)#right
            if np.isnan(allhrirs).any():
                log_debug('Output of getIRs will containt nans')
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
                                   

######################## IRCAM HRIR
class IRCAMSubjectReceiver(HRTFReceiver):
    '''
    A receiver based on a IRCAM HRIR set
    '''
    def __init__(self, position, orientation = FRONT, path=None, subject=get_pref('IRCAM_DEFAULTSUBJECT', 1066)):
        if isinstance(position, (float, int)):
            position = position*UP

        HRTFReceiver.__init__(self, position, ircamHRIR(subject, path=path), orientation=orientation, is_distancedependent=False)

######################## Spherical head model
class SphericalHeadReceiver(HRTFReceiver):
    '''
    Initialized with the height of the head center and interaural distance.
    Is a simple two-points receiver that represents an empty head.

    ** Initialization **

    '''
    def __init__(self, height, iad, orientation = FRONT, samplerate = 44100.,
                    nfft = 1024, pre_delay = 128):
        if isinstance(height, float):
            position = Point(height * UP)
        else:
            position = height
        OrientedReceiver.__init__(self, position, orientation)
        self.headmodel = SphericalHead(iad, (0,0), samplerate = samplerate, nfft = nfft)
        self.pre_delay = pre_delay
        self.iad = iad
        self.nsamples = self.headmodel.nfft
        self.samplerate = samplerate

    def get_ear_position(self, whichone):
        d = UP.vectorial_product(self.orientation)#vector from center to left ear
        if whichone == 'left':
            return Point(self.iad/2.0*d + self.position)
        elif whichone == 'right':
            return Point(-self.iad/2.0*d + self.position)
        else:
            ValueError('Fetched ear position must be left or right, it was '+str(whichone))

    def get_hrir(self, az, el, d = 20):
        return self.headmodel.get_hrir(az, el, pre_delay = self.pre_delay)


################### closest coordinates #################################
def get_closest_coords(coords, arg, retval = ''):
    if len(arg) == 2:
        az, el = arg
        ddist = np.zeros(len(coords))
    else:
        az, el, d = arg
        ddist = (coords['dist']-d)**2

    azdist = (coords['azim']-az)**2
    eldist = (coords['elev']-el)**2

    idmin = np.argmin(azdist+eldist+ddist)

    if retval == 'idmin':
        return idmin
    else:
        if len(arg) == 2:
            return coords['azim'][idmin], coords['elev'][idmin]
        else:
            return coords['azim'][idmin], coords['elev'][idmin], coords['dist'][idmin]

