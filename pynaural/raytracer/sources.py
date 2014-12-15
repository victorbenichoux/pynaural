from pynaural.raytracer.geometry.rays import RandomSphericalBeam
from pynaural.raytracer.geometry.base import FloatTriplet, FRONT, Point, Vector, degAz, degEl, ORIGIN, orthodromic_distance
from pynaural.raytracer.geometry.surfaces import Sphere
from pynaural.utils.debugtools import log_debug
from pynaural.raytracer.receivers import Receiver, OrientedReceiver

import numpy as np

__all__=['BaseSource',
         'Source', 'EqualizedSource',
         'pair_sources', 'linspace_sources', 'relative_sources']

########################## Sources

class BaseSource(object):
    '''
    Class to work with acoustic sources, practically one never quite
    uses BaseSource, but Source instead.
    
    ** Identification **
    
    When added to a scene, any Source object is given an id that
    corresponds to its position in the ``surfaces`` attribute of the
    GeometricScene.
    '''
    def __init__(self, loc):
        # may be plain useless, 
        # should consider adding cardiod etc...
        self._id = None
        if isinstance(loc, Point):
            self.position = loc
        elif isinstance(loc, Vector):
            self.position = Point(loc)
        else:
            raise TypeError

    ################ ID ###############
    def set_id(self, id):
        '''
        Sets an Id relative to a surface for simple use of targetted beams
        '''
        self._id = id
    
    def get_id(self):
        if not (self._id is None):
            return self._id
        else:
            raise AttributeError('Current source not linked to a scene(no id)')
        
    @property
    def id(self):
        return self.get_id()

    def __str__(self):
        return 'Source:\n '+str(self.position)+'\n'
    
    def getRelativeCoords(self, receiver):
        if isinstance(receiver, Receiver):
            d = Vector(self.position.array() - receiver.position.array())
            return d
        else:
            raise TypeError
        
    def getRelativeSphericalCoords(self, receiver, unit = 'deg'):
        if isinstance(receiver, Receiver):
            direction = Vector(self.position.array() - receiver.position.array())
            (d, az, el) = direction.toSphericalCoords(unit = unit)
            if isinstance(receiver, OrientedReceiver):
                (_, azref, elref) = receiver.orientation.toSphericalCoords(unit = unit)
                az -= azref
                el -= elref
            return (d, az, el)
        else:
            raise TypeError
 

################## Spherical Source is the only source #######################
class Source(BaseSource, Sphere):
    '''
    Spherical Source, should be used instead of Source because
    otherwise numerical problems in raytracing arise.
    reference distance is of 1cm.
    Initialized with a location.
    '''
    def __init__(self, loc, ref = .01, model = None):
        Sphere.__init__(self, loc, ref)
        self.model = model
        super(Source,self).__init__(loc)


# positioning methods 
# maybe add sources for hrtf here ? (i.e move them away from scene.placeSources
    @staticmethod
    def pair_sources(position, isd, orientation = FRONT):
        '''
        Places two sources separated by a certain distance, with a certain orientation... Both in the same plane
        '''
        if isinstance(position, Vector):
            position = Point(position)
        d = Vector(1, -orientation.x/orientation.y, 0)
        positionright = position + isd/2.*d
        positionleft = position + (-isd/2.)*d
        return [Source(positionleft), Source(positionright)]

    @staticmethod
    def linspace_sources(start, end, n, **kwdargs):
        res = []
        d = Vector((end-start).array())/n
        for i in range(n):

            res.append(Source(start+i*d, **kwdargs))
        return res

    @staticmethod
    def relative_sources(*args, **kwdargs):
        '''
        Constructor to place sources positioned relatively to any
        object with a position attribute.

        
        Sample usage, with a function as a filter, relatively to a
        receiver
        
        relative_sources(receiver, 1*meter, lambda azim,elev: azim ==0)
        
        This will place Sources where the receiver has HRTFs (no interpolation)
        
        Sample usage, with lists of coordinates:
        relative_sources(receiver, [1*meter 2*meter], [90 180], 15)
        
        This places 4 sources at 1 meter, 90 Az, 15 El, etc...
        
        Additionally, one can pass a list of [(d, az, el), ...] which will place only one source for each d, az, el triplet.
        '''
        if isinstance(args[0], FloatTriplet):
            position = args[0].array()
            receiver = None
        
        if hasattr(args[0], 'position'):
            receiver = args[0]
            position = args[0].position.array()
        
        distances = ensure_iterable(args[1])

        res = []

        if len(args) == 2 and isinstance(args[1], list):
            for d, az, el in args[1]:
                rotation = az * degAz + el * degEl
                sourceposition = Vector(position) + float(d)*(rotation*receiver.orientation)
                res.append(Source(sourceposition, **kwdargs))
            return res
                

        if len(args) <= 2:
            coordsfilter = lambda azim, elev: True
        else:
            coordsfilter = args[2]

        if callable(coordsfilter):
            if not (receiver is None) and hasattr(receiver, 'get_coordinates'):
                for d in distances:
                    coords = receiver.get_coordinates(coordsfilter)
                    for (az, el) in coords:
                        rotation = az*degAz + el*degEl
                        sourceposition = Vector(position) + float(d)*(rotation*receiver.orientation)
                        res.append(Source(sourceposition, **kwdargs))
            else:
                raise ValueError('Cannot build sources because the Receiver doesnt have fixed coordinates')
        else:
            azimuths = ensure_iterable(coordsfilter)
            if len(args) == 3:
                # no el asked for, going with 0
                elevations = [0.]
            else:
                # make sure its iterable
                elevations = ensure_iterable(args[3])
            
            for d in distances:
                for az in azimuths:
                    for el in elevations:
                        rotation = az*degAz + el*degEl
                        sourceposition = Vector(position) + float(d)*(rotation*receiver.orientation)
                        res.append(Source(sourceposition, **kwdargs))
        return res

pair_sources = Source.pair_sources      
relative_sources = Source.relative_sources
linspace_sources = Source.linspace_sources

################## Source Equalization #######################
class EqualizedSource(Source):   
    '''
    ** Source equalization ** 
    
    When rays reach the source, they can be equalized according to a
    scheme defined in model (not yet implemented). Until this feature
    appears, the only equalization done is uniform. To do so, one can
    compute a voronoi partition of the rays arriving on the sphere.
    

    .. automethod:: voronoi_partition
    
    By default an EqualizedSource is omnidirectional, so
    get_equalization yields directly the result of voronoi_partition.

    .. automethod:: get_equalization
    
    Other types may come in the future.
    '''
    def get_equalization(self, beam):
        '''
        Returns the equalization weights from the beam, representing
        rays arriving to the source. Since the source is
        omnidirectional, the arriving rays are weigh according to the
        surface of the sphere they account for, so it is the result of
        voronoi_parittion.
        '''
        return self.voronoi_partition(beam)
    
    
    def voronoi_partition(self, beam, method = 'directions'):
        '''
        Returns an array of weights (0<weight<1) that correspond to
        the surface area covered by the voronoi partition of the
        sphere around the source given by the rays' arrival
        directions.
        This is further used as a cue of the amount of acoustic power
        this ray is carrying.
        
        *advanced*
        ``method = 'direction'`` which data to use to construct the
        voronoi partition. If it is 'directions' then the directions
        of the incoming rays is used (actually -directions). If it is
        'origins', then the vectors used are vectors from center of
        the source to the origins of the rays (that lay on the surface
        of the source).
        '''
        log_debug('Computing Voronoi partition')
        if method == 'origins':
            positions = np.tile(self.position.array(), (1, beam.nrays))
            data = beam.origins - positions
        elif method == 'directions':
            data = -beam.directions
        else:
            raise AttributeError('Unrecognized partition method')
        
        try:
            N = data.shape[1] * 5
            res = voronoi_partition(data, N)
            return res
        except ValueError:
            log_debug('Voronoi estimation requires less data, sequentializing...')
            
            N /= 50
            res = np.zeros(beam.nrays)
            for i in range(50):
                log_debug('Chunk '+str(i+1)+'/50')
                res += voronoi_partition(data, N)/50
            return res


#################### UTILS
# make an object iterable
def ensure_iterable(obj):
    try:
        [x for x in obj]
        return obj
    except TypeError:
        return [obj]


# Voronoi partitioning
def voronoi_partition(data, N, probe = None, distance = 'orthodromic'):
    '''
    Uses a Monte Carlo method to estimate a Voronoi diagram on a
    sphere. Distance criterion is orthodromic


    ** Arguments **
    `` data `` Data points, laying on a sphere. The voronoi partition
    is the distribution of non overlapping areas containing points
    that are the closest to those provided here.

    `` N `` Number of probe positions (number of points used to seed
    the montecarlo method)
    
    ** Keyword arguments **
    
    `` probe = None`` One can supply manually the probe positions
    (discrete positions on the sphere) that the algorithm uses to
    generate the voronoi diagram. If it is None, a monte carlo method
    is used. If it is set, then this overrides the ``N`` argument.
    '''
    if probe == None:
        probe = RandomSphericalBeam(ORIGIN, N).directions
    else:
        N = probe.shape[1]
        
    dists = np.zeros((data.shape[1], N))
    for i in range(data.shape[1]):
        dists[i,:] = orthodromic_distance(probe, data[:,i])
    closest = np.argmin(dists, axis = 0)
    
    # fetching final values
    res = np.zeros(data.shape[1])
    for i in range(data.shape[1]):
        res[i] = len(np.nonzero(closest == i)[0]) / float(N)
    return res
