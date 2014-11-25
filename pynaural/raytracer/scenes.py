import matplotlib.pyplot as plt
import numpy as np

from pynaural.utils.debugtools import log_debug
from pynaural.raytracer.geometry.base import FloatTriplet, ORIGIN, Point, Vector, FRONT, BACK, LEFT, RIGHT, UP, DOWN, ORIGIN
from pynaural.raytracer.geometry.surfaces import Plane, Surface, Sphere
from pynaural.raytracer.geometry.rays import Beam
from pynaural.raytracer.sources import Source
from pynaural.raytracer.receivers import Receiver
from pynaural.raytracer.acoustics import c


DEFAULT_PRECISION = 1e-2 #precision, to prevent self intersection, here 1cm
DEFAULT_STOPCONDITION = 30 # default number of reflections
DEFAULT_NRAYS = 1e6 # default number of rays

N_MAXREFLECTIONS = 10e5
MAX_ARRAY_SIZE = 5*10**6

__all__ = ['GeometricScene',
           'VoidScene',
           'RoomScene',
           'GroundScene']

#####################################################################################################
########################################## GEOMETRIC SCENE ##########################################
#####################################################################################################

class GeometricScene(object):
    '''
    Class that encapsulates all the geometric data of a scene, and
    takes care of the raytracing itself.
    
    GeometricScenes may contain any number of Surface and Source
    objects. Those are used for raytracing a Beam through the
    ``render`` method.
    
    The raytracing algorithm can be controlled in one of two ways: by
    changing the stop condition, or by changing the precision
    attribute.
    
    ** Initialization **

    To initialize a GeometricScene, one can use any number of
    arguments, corresponding to objects to be added to the scene. 

    ``stopcondition = 30`` defines the stopcondition of the raytracing algorithm.

    ``precision = 1e5`` defines the threshold of precision for
    avoiding autointersection during raytracing. Due to numerical
    (in)accuracy issues, sometimes a ray that just reflected on a
    surface will come from the back of said surface and reintersect
    it. That's why we use this threshold that prevents this by
    forbidding any intersection with a surface closer than ``precision``

    `` nrays`` Defines the default number of rays to be generated in
    that scene (see get_beam)
    
    ** Object Management **

    .. automethod:: add
    .. autoattribute:: surfaces
    .. autoattribute:: nsurfaces
    .. autoattribute:: sources
    .. autoattribute:: nsources

    ** Raytracing ** 

    .. automethod :: render

    .. autoattribute :: stopcondition
    .. automethod :: set_stopcondition

    .. autoattribute :: precision

    ** Beam handling ** 

    .. automethod :: get_beam
    .. automethod :: get_beam_onereceiver

    ** Geometry **
    
    .. automethod :: volume

    ** Misc. **
    .. automethod :: plotsources

    ** Acoustical measures **

    .. automethod :: modedensity
    .. automethod :: eyring
    .. automethod :: sabine
    .. automethod :: reflectionnumber
    

    '''
    def __init__(self, *args, **kwdargs):
        stopcondition = kwdargs.get('stopcondition', DEFAULT_STOPCONDITION)
        self.nrays = kwdargs.get('nrays', DEFAULT_NRAYS)
        self.precision = kwdargs.get('precision', DEFAULT_PRECISION)

        self.set_stopcondition(stopcondition)

        self.surfaces = []
        self.nsurfaces = 0
        self.sources = []        
        self.nsources = 0
        for arg in args:
            self.add(arg)

    def add(self, *obj):
        '''
        Adds the given object, or list of objects to the
        GeometricScene. Objects may be either Sources or Surfaces.
        '''
        if len(obj) > 1:
            for o in obj:
                self.add(o)
            return
        else:
            obj = obj[0]
            
        if isinstance(obj, list):
            for o in obj:
                self.add(o)
            return


        if isinstance(obj, Surface):
            obj.set_id(self.nsurfaces)
            self.surfaces.append(obj)
            self.nsurfaces += 1
            if isinstance(obj, Source):
                if not self.is_inside(obj):
                    log_debug('Source may not be inside the Scene')
                self.sources.append(obj)
                self.nsources += 1
                obj.set_id(self.nsources + (self.nsurfaces - self.nsources - 1))
            return
        
        raise ValueError(
            """Only Surface or Source objects may be added to a
            scene, here """+str(type(obj)))

    def __str__(self):
        res='SCENE\n  Surfaces:\n'
        for obj in self.surfaces:
            res+=str(obj)+'\n'
        return res

    def preparesurfaces(self, nrays):
        # calls .prepare on all the surfaces. This prepares for the
        # hardcore vectorization of intersection...
        for surf in self.surfaces:
            surf.prepare(nrays)


    ########################## PLOTTING ##########################

    def plotsources(self, display = False):
        '''
        Plots the sources in a 3D axes object. 

        ``display = False`` If set to True, then show() is called.

        '''
        ax = plt.subplot(111, projection = '3d')
        X = np.zeros(self.nsources)
        Y = np.zeros(self.nsources)
        Z = np.zeros(self.nsources)
        for i,source in enumerate(self.sources):
            (x,y,z) = source.position.tuple()
            X[i] = x
            Y[i] = y
            Z[i] = z

        ax.scatter(X,Y,Z, '*y')
        if display:
            plt.show()

    ########################## GEOMETRY ##########################

    def compute_volume_areas(self, **kwdargs):
        if 'Niter' in kwdargs:
            Niter = float(kwdargs['Niter'])
            del kwdargs['Niter']
        else:
            Niter = 1.

        self._volume = 0.
        self._areas = np.zeros(self.nsurfaces)
        
        for i in range(int(Niter)):
            v, a = self.compute_volume_areas_once(**kwdargs)
            if False:
                print v
                print a
            self._volume += v/Niter
            self._areas += a/Niter

    def compute_volume_areas_once(self, 
                                  ref_point = ORIGIN, 
                                  abs_max = 10., Npoints = 1e7,
                                  debug = False):
        '''
        Monte Carlo method for estimation of the volumes and surface areas of an enclosed GeometricScene.
        
        '''

        log_debug('Computing volumes and areas')
        if not isinstance(abs_max, np.ndarray):
            max_coords = abs_max*np.ones((3,2))
            max_coords[:, 0] *= -1
        
        if not isinstance(ref_point, np.ndarray):
            if isinstance(ref_point, Receiver):
                ref_point = ref_point.position
            ref_point = ref_point.array()
            
            
        ## Volume estimation
        # compute total volume of the slab
        extent = max_coords[:,1] - max_coords[:,0]
        vtot = np.prod(extent)

        # generate uniformly distributed points in the slab
        points = np.zeros((3, Npoints))
        tmprand = np.random.rand(Npoints)
        points[0, :] = max_coords[0, 0] + tmprand*extent[0] + ref_point[0,0]
        tmprand = np.random.rand(Npoints)
        points[1, :] = max_coords[1, 0] + tmprand*extent[1] + ref_point[1,0]
        tmprand = np.random.rand(Npoints)
        points[2, :] = max_coords[2, 0] + tmprand*extent[2] + ref_point[2,0]

        # check which points are actually inside the volume
        res = self.is_inside(points)
        n_inside = len(np.nonzero(res)[0])        

        if n_inside == 0:
            log_debug('No points were detected in a space')
        if n_inside == Npoints:
            log_debug('All points were generated in the space, try using a bigger slab')

        # and guess the volume
        p = float(n_inside)/Npoints
        volume = p*vtot

        if debug:
            ax = plt.subplot(111, projection = '3d')
            ax.scatter(points[0,res],points[1,res], points[2,res], color = 'g')
            ax.scatter(points[0,-res],points[1,-res], points[2,-res], color = 'r', alpha = 0.1)
            plt.show()
        

        ## Surfaces estimation
        # This here is a bit tricky. to estimate the surfaces
        # we displace the points generated in the volume following the
        # normal of the considered surface. We can then guess the
        # area of that surface by how many points are now outside the room!
        in_points = points[:,res]
        # we have to figure out the magnitude of the shift (alpha), 
        density = n_inside/volume
        # say we want to lose 10 percent of the points, 
        frac = .1
        # since we expect the surface to be about (if its square)
        expected_surf = (volume ** (1./3))**2
        # so the expected shift to lose frac*n_inside points is
        alpha = frac*n_inside/(density*expected_surf)
        # we'll use that value for shifts
        areas = np.zeros(self.nsurfaces)
        for i,surface in enumerate(self.surfaces):
            # only works with planes
            if isinstance(surface, Plane):
                # we take all the points inside the volume, and 
                # shift them by -alpha * normal, 
                shift = np.tile(-surface.norm.array(), (1, n_inside))
                shifted = in_points + alpha*shift
                # figure out how many we lost, that is how many
                # crossed this surface
                still_inside = surface.is_in_updomain(shifted)
#                still_inside = self.is_inside(shifted)
                # the fraction 
                frac_displaced = 1-float(len(np.nonzero(still_inside)[0]))/n_inside
                # the lost volume
                volume_lost = frac_displaced * volume
                if False:
                    # debug
                    print 'frac', frac_displaced
                    print 'volume_lost',volume_lost
                    print 'alpha',alpha
                # hence the surface!
                areas[i] = (volume_lost/alpha)
            else:
                if not isinstance(surface, Source):
                    log_debug('Can only compute surfaces of planes')    
                areas[i] = 0
        return volume, areas

    def volume(self, **kwdargs):
        '''
        Estimates the volume of the enclosed space in the
        GeometricScene around the ref_point using a monte carlo method.

        More precisely, draws Npoints random points within the slab
        centered on ref_point, with coordinates within abs_max of this
        points. Then evaluates the volume by testing whether or not
        each point is in the closed space around the reference point. 
        
        Keyword arguments:
        
        ``abs_max = 10`` Can be a (3x2) ndarray with min/max values
        along each dimension, or just a single value that defines the
        maximum absolute value of the generate coordinates

        ``Npoints = 1e7`` Defines the number of points to generate, a
        higher value may lead to more precise estimation, but too high
        values raise a MemoryError.
        
        ``Niter = 1`` If set to more than one then the volume
        computations is run Niter times and averaged.
        
        ``ref_point = ORIGIN`` Volume will be measured around the
        point ref_point.
        '''
        if not hasattr(self, '_volume') or len(kwdargs)>0:
            self.compute_volume_areas(**kwdargs)
        return self._volume
    
    def area(self, i, **kwdargs):
        '''
        Returns the surface of the i'th Surface object in the
        scene. It is computed alongside the volume using a Monte Carlo method.
        '''
        if not hasattr(self, '_volume') or len(kwdargs)>0:
            self.compute_volume_areas(**kwdargs)
        if isinstance(i, Surface):
            i = i.id
        return self._areas[i]

    def total_enclosure_area(self, **kwdargs):
        if not hasattr(self, '_volume') or len(kwdargs)>0:
            self.compute_volume_areas(**kwdargs)
        return np.sum([self.area(i) for i in range(self.nsurfaces)])

    def is_inside(self, obj):
        '''
        Method to tell whether an object is inside a convex scene or
        not. It may not work for non-convex geometries.
        
        `` obj`` may be any object with a ``position`` attribute, or a
        (3xn) ndarray, or a Vector/Point.
        
        '''
        if hasattr(obj, 'position'):
            obj = obj.position
            
        if isinstance(obj, Vector) or isinstance(obj, Point):
            points = obj.array()
        
        if isinstance(obj, np.ndarray):
            points = obj
            
        res = np.ones(points.shape[1], dtype = bool)
        for surface in self.surfaces:
            if not isinstance(surface, Source):
                tmp = surface.is_in_updomain(points)
                res *= tmp
        return res
        
        
    
    ########################## STOP CONDITION ##########################
    @property
    def stopcondition(self):
        '''
        Returns the current value of the stop condition.
        '''
        return self.stopcondition_value

    def set_stopcondition(self, val):
        '''
        Sets the stop condition of the raytracing algorithm to the
        value given as an argument. If the value is an integer, then
        the stop condition will be the maximum number of reflections
        computed. Otherwise, it is the longest time-of-flight of the rays.
        '''
        if type(val) == int:
            ## the stop condition is specified as a number of reflections.
            def intStopCondition(n, distances):
                return (n < val) * np.ones(distances.shape, dtype = bool)
            self.stopcondition_value = val
            self._stopcondition = intStopCondition
        else:
            ## the stop condition is specified as a time of flight.
            val *= c
            def distanceStopCondition(n, distances):
                return (n < N_MAXREFLECTIONS) * np.array(distances < val, dtype = bool)
            self.stopcondition_value = val
            self._stopcondition = distanceStopCondition

    ########################## BEAM HANDLING ############################

    def get_beam(self, position, nrays = DEFAULT_NRAYS):
        '''
        Returns a Beam object originating at a given 3D position and all the
        sources in the GeometricScene.
        
        ``nrays = 1e5`` Defines the number of rays to be generated per
        couple (receiver, source). Note: This is ignored when the rays
        are known exactly (see SquareRoomScene for example).
        '''

        beam = Beam(0)

        if isinstance(position, FloatTriplet):
            position = position.array().reshape((3,1))

        elif isinstance(position, np.ndarray) & position.ndim == 1:
            position = position.reshape((3, 1))
        
        for source in self.sources:
            cur_nrays = float(nrays)/self.nsources
            beam = beam.cat(RandomSphericalBeam(position, 
                                                int(cur_nrays),
                                                target_source = source.id))
        return beam        

    def render(self, obj, **kwdargs):
        '''
        Raytracing algorithm. 

        ``obj`` Defines on which object to do the rendering. 
        For Receiver inputs, the Scene decides
        which Beam to generate, generates one beam for all source objects
        and renders it.
        If ``obj`` is a beam, then the beam is rendered in the scene and
        returned.

        Keyword Arguments:

        ``cpu = 1`` If set to more than 1 the algorithm will be
        distributed across CPUs using playdoh. (experimental)
        
        ``debug = False`` If set to True, will be *very* verbose.
        
        ``nrays = None`` passed to get_beam.
        '''
        if isinstance(obj, Receiver):
            nrays = kwdargs.get('nrays', self.nrays)
            beam = self.get_beam(obj.position, nrays = nrays)
        elif isinstance(obj, Beam):
            beam = obj
        else:
            return TypeError("Doesn't recognize given Beam/Receiver for rendering")
        return self._render(beam, **kwdargs)


    def _render(self, beam, **kwdargs):
        # this function is just a wrapper for the _renderstep method
        # where the raytracing actually happens. her

        log_debug('Preparing surfaces for '+str(beam.nrays)+' rays')
        self.preparesurfaces(beam.nrays)
        
        log_debug('Starting simulation')
        out = self._renderstep(beam, 0, **kwdargs)

        reached_source = out[np.nonzero(out.get_reachedsource())[0]]
        log_debug(str(float(reached_source.nrays)/out.nrays*100)+'% of rays reached the source ('+str(reached_source.nrays)+')')
        return out
                    
    def _renderstep(self, beam, n, **kwdargs):
        # here is where the magic happens. it's a bit complicated, but
        # intuitively it's straightforward. i'll try to comment you
        # through it, dear (crazy) reader.

        debug = kwdargs.get('debug', False)
        exp_parallel = kwdargs.get('exp_parallel', False)
        
        # active rays match 3 properties:
        # - they aren't going away to infinity (its what surfacehit ==
        # inf means)
        # - they haven't reached their source
        # - they match the stopcondition
        active_rays = np.nonzero(np.isfinite(beam.surfacehit) * (-
                beam.sourcereached) * self._stopcondition(n,
                                                          beam.traveleddistance))[0]
        propactive = (len(active_rays)/float(beam.nrays))
        log_debug('Step '+str(n)+', '+str((1-propactive)*100)+'% rays reached the source')

        if len(active_rays) != 0:
            # computing all the distances
            dists = np.inf*np.ones((len(self.surfaces), beam.nrays))
            for i in range(len(self.surfaces)):
                if isinstance(self.surfaces[i], Source):
                    # if it has reached its source, (or if source is
                    # undefined = -1)
                    condition = np.array((beam.target_source == i), dtype = bool) + np.array((beam.target_source == -1), dtype = bool)
                    curr_target_rays = np.nonzero(condition * np.isfinite(beam.surfacehit) * - beam.sourcereached)[0]
                    # keep track of the distances
                    dists[i, curr_target_rays] = self.surfaces[i].intersection(beam[curr_target_rays])
                else:
                    # keep track of the distances
                    dists[i, active_rays] = self.surfaces[i].intersection(beam[active_rays])
                if not (dists >= 0).all():
                    log_debug('WARNING, negative distances ?')
                    print dists[dists < 0]
            # distance correction
            dists = correct_dists(dists, epsilon = self.precision)
            # minimum distances along the columns
            mindists = np.min(dists, axis = 0)
            # not infinite distance means something was hit
            notinf_idx = np.nonzero(np.isfinite(mindists))[0]
            # what was hit?
            intersurfs = np.argmin(dists[:, notinf_idx], axis = 0)

            if debug:
                print 'DEBUG, step ',n
                print 'active_rays: ',active_rays
                print 'dists: ',dists
                print 'mindists: ',mindists
                print 'notinf_idx: ',notinf_idx
                print 'intersurfs: ',intersurfs

            
            # starting to create following beam
            beam.surfacehit = np.inf * np.ones(beam.nrays, dtype = int)
            next = Beam(beam.nrays)
            next.directions = np.nan*np.ones((3,beam.nrays)) # no more intersection
            next.origins = np.nan*np.ones((3,beam.nrays)) # no more intersection
            # those attributes transfer down the list
            next.sourcereached = beam.sourcereached.copy()
            next.surfacehit = beam.surfacehit
            for i, surface in enumerate(self.surfaces):
                # now, for every surface, who hit it?
                subbeam_idx = notinf_idx[np.nonzero(intersurfs == i)]
                if len(subbeam_idx != 0):
                    # if someone has...
                    if debug:
                        print 'surface '+str(i)+' was hit, by rays '+str(subbeam_idx)
                    surface = self.surfaces[i]
                    beam.surfacehit[subbeam_idx] = i
                    beam.distances[subbeam_idx] = dists[i,subbeam_idx]
                    intersectionpoints = beam[subbeam_idx].atDist(beam.distances[subbeam_idx])
                    beam.incidences[subbeam_idx] = surface.get_incidence(
                        beam.directions[:,subbeam_idx], 
                        intersectionpoints)
                    # geometric computation
                    if isinstance(surface, Source):
                        # if it was a source, remember it...
                        next.sourcereached[subbeam_idx] = True
                        next.directions[:, subbeam_idx] = 0
                        next.origins[:, subbeam_idx] = intersectionpoints
                    else:
                        # if not reflect it...
                        reflected_beam = surface.reflect(
                            beam.directions[:,subbeam_idx],
                            intersectionpoints)
                        next.sourcereached[subbeam_idx] = False
                        next.directions[:,subbeam_idx] = reflected_beam.directions
                        next.origins[:,subbeam_idx] = reflected_beam.origins
                        next.target_source[subbeam_idx] = beam.target_source[subbeam_idx]
                    # in any event, keep the traveleddistance
                    next.traveleddistance[subbeam_idx] = beam.traveleddistance [subbeam_idx] + dists[i, subbeam_idx]
            # make sure that the output Beam has its directions normalized        
            next.normalize()
            if debug:
                print 'next beam'
                print 'origins:',next.origins
                print 'directions:',next.directions
                print 'sourcereached:',next.sourcereached
                print 'surfacehit:', beam.surfacehit
            # chain it! pfiou...
            beam.next = self._renderstep(next, n+1,
                                         precision = self.precision, debug = debug)
        else:
            log_debug('No more active rays, stopping render')
        return beam
    

    ###################### ACOUSTICAL MEASURES ######################

    def modedensity(self, freq):
        '''
        Computes the mode density ratio, ie deltaN/deltaf, the number
        of modes per bandwidth of the room (intr.acous. p 111) at a
        given frequency. 
        
        NB: 
        - approx, works for high freqs with regards to f0
        - homogene a un temps
        
        For a sound that lasts t0, with fundamental at f0, one must
        find about 10 modes in the frequency band centered around f0
        of width 2*pi/t0.
        That is the number of modes for a signal defined as up, in a
        band of width deltaf is:
        room.modedensity(f)*deltaf
        for the sound to be intelligible, one must have

        Notes:
        - wtf?
        '''
        return 4*pi*self.volume*freq**2*c**-3
    
    def modes(self, bound = 1000.):
        '''
        Computes the possible room mode frequencies up to a certain
        ``bound`` (in Hz) given as an argument.
        
        Note that since there are a huge number of room modes in high
        frequencies, imposing a high bound (more than a few kHz) will
        most likely raise a MemoryError.
        '''
        n = 1
        flag = True
        tmp = -1
        while flag:
            n *= 2
            combinations = np.arange((n)**3)
            nL = np.mod(combinations, n)
            nl = np.mod(combinations - nL, (n)**2)/n
            nh = np.mod(combinations - nl*(n) - nL, (n)**3)/n**2
            freqs = (c / 2)*((nL/L)**2  + (nl/l)**2 + (nh/h)**2)**.5
            goodfreqs = np.nonzero(freqs <= bound)[0]
            if len(goodfreqs) == tmp:
                flag = False
                res = freqs[goodfreqs]
            print 'max',np.max(freqs)
            tmp = len(goodfreqs)
        return freqs

    def reflectionnumber(self, t):
        '''
        Returns the number of reflections expected in a rectangular
        hall before time t.
        Follows the formula 2pi(ct)**3/(3V) (Dalenback 96, ref12)
        '''
        return 4*np.pi*(c*t)**3/(3*self.volume())
    
    def sabine(self):
        '''
        Evaluates Sabine's formula (RT60) for the current scene.
        '''
        if not hasattr(self, '_volume'):
            self.compute_volume_areas(**{})
        tmp = np.sum([self.surfaces[i].model.alpha*self.area(i) 
                      for i in range(self.nsurfaces)])
        return 0.161*self.volume()/tmp
    
    def eyring(self):
        '''
        Evaluates Eyring's formula (RT60) for the current scene.
        '''
        if not hasattr(self, '_volume'):
            self.compute_volume_areas(**{})
        tmp = np.sum([self.surfaces[i].model.alpha*self.area(i) 
                      for i in range(self.nsurfaces)])
        S = self.total_enclosure_area()
        tmp = -S*np.log(1-tmp/S)
        return 0.161*self.volume()/tmp


#####################################################################################################
########################################## GEOMETRIC SCENE ##########################################
#####################################################################################################
  
###################### VoidScene ######################

class VoidScene(GeometricScene):
    '''
    Class to work in anechoic environments.
    
    In this setting the environment impulse response will aways be dirac-like.
    '''
    def __init__(self):
        super(VoidScene, self).__init__(stopcondition = 1, model = None)
        self.set_stopcondition(1)

    def get_beam(self, pos, **kwdargs):
        # Here the beam is known exactly
        beam = Beam(self.nsources)
        if isinstance(pos, Point):
            pos = pos.array().reshape((3, 1))
        elif isinstance(pos, np.ndarray) & pos.ndim == 1:
            pos = pos.reshape((3, 1))
        beam.origins = np.tile(pos, (1, self.nsources))
        sourcepos = np.hstack(tuple([
                    self.sources[k].position.array().reshape((3, 1)) for k in range(self.nsources)]))
        # beam towards the source (direct)
        beam.directions = sourcepos - pos#).squeeze()
        beam.directions /= np.sqrt(np.sum(beam.directions**2, axis = 0))
        beam.target_source = np.array([source.id for source in self.sources])
        return beam

    def volume(self):
        raise AttributeError('Volume is infinite for a VoideScene')
      
###################### GroundScene ######################
    
class GroundScene(GeometricScene):
    '''
    Class to work with simple settings in which only a ground/wall is present.

    The environment is modeled using a single plane, with arbitrary
    model. 

    ** Initialization ** 
    
    ``mode = 'ground'`` By default the Plane object added to the Scene
    is GROUND, that is the plane at z = 0, one can set this mode to
    WALL in which case the wall is placed at x = 0, facing right.


    ** Sample usage **
    
    To model a simple situation where the ground absorbs 3dB of
    signal, one can do:
    
    outdoor = GroundScene(model = RigidReflectionModel(3*dB))

    '''
    def __init__(self, mode = 'ground'):
        if mode == 'ground':
            super(GroundScene, self).__init__(
                Plane(ORIGIN, UP), stopcondition = 2)
            self.ground = self.surfaces[0]
        elif mode == 'wall':
            super(GroundScene, self).__init__(WALL, stopcondition = 2)
            self.wall = self.surfaces[0]
    
    def get_beam(self, position, nrays = DEFAULT_NRAYS):
        if isinstance(position, FloatTriplet):
            position = position.array().reshape((3,1))

        elif isinstance(position, np.ndarray) & position.ndim == 1:
            position = position.reshape((3,1))

        # Here the beam is known exactly
        res = Beam(0)
        for k in range(self.nsources):
            beam = Beam(2)
            beam.set_target(self.sources[k])
            

            
            beam.origins = np.tile(position, (1,2))
            sourcepos = self.sources[k].position.array().reshape((3,1))
            # first beam towards the source (direct)
            beam.directions[:,0] = (sourcepos - position).squeeze()
            beam.directions[:,0] /= np.sqrt(np.sum(beam.directions[:,0]**2))
            # second beam towards the virtual source (reflected)
            tmp = Beam(1)
            tmp.origins = sourcepos
            tmp.directions = -self.ground.normal_at(tmp.origins)
            dh = self.ground.intersection(tmp)
            sprime = tmp.atDist(2*dh)
            beam.directions[:,1] = (sprime - position).squeeze()
            beam.directions[:,1] /= np.sqrt(np.sum(beam.directions[:,1]**2))
            res = res.cat(beam)
        return res

    def volume(self):
        raise AttributeError('Volume is infinite for a GroundScene')


###################### Room Scene ###################

class RoomScene(GeometricScene):
    '''
    Class for working with rectangular room shapes.
    
    It creates a cuboid-shaped acoustic scene, the ground being the z = 0 plane, with the origin in the middle of the ground. The computations are much faster here since they are base on the image method described in the 1979 Allen and Berkley paper ("Image Method for Efficiently Simulating Small-room Acoustics").
    
    

    ** Sample usage ** 
    
    room = RoomScene(3*m, 4*m, 2.5*m)
    
    '''
    def __init__(self, l, L, h, **kwdargs):
        if 'nreflections' in kwdargs:
            self.nreflections = kwdargs['nreflections']
            kwdargs['stopcondition'] = self.nreflections + 1
            
        self.l = float(l)
        self.L = float(L)
        self.h = float(h)

        self.leftwall = Plane(Point(-l/2., 0, 0), RIGHT, label = 'left wall')
        self.rightwall = Plane(Point(l/2., 0, 0), LEFT, label = 'right wall')
        self.backwall = Plane(Point(0, -L/2., 0), FRONT, label = 'back wall')
        self.frontwall = Plane(Point(0., L/2., 0), BACK, label = 'front wall')
        self.floor = Plane(Point(0, 0, 0), UP, label = 'floor')
        self.ceiling = Plane(UP * h, DOWN, label = 'ceiling')
        
        super(RoomScene,self).__init__(*[self.leftwall,self.rightwall,
                                         self.backwall,self.frontwall,
                                         self.floor, self.ceiling], **kwdargs)
        
    def get_beam(self, position, nrays = DEFAULT_NRAYS):
        if isinstance(position, FloatTriplet):
            position = position.array().reshape((3, 1))
        elif isinstance(position, np.ndarray) & position.ndim == 1:
            position = position.reshape((3, 1))

        nreflections = self.nreflections
        res = Beam(0)
        for k in range(self.nsources):
            # total number of real+virtual sources
            # TODO:
            # find out the analytical value, see test_sum (in old spatializer files)
            def n2d(i):
                return 2*i*(i+1) + 1
            def n3d(i):
                sumvect = n2d(np.arange(i))
                return n2d(i) + 2 * np.sum(sumvect)
            N = n3d(nreflections)

            tmp = Beam(N, target_source = self.sources[k].id)
            
            tmp.origins = np.tile(position, (1, tmp.nrays))
            tmp.nreflections = np.zeros(tmp.nrays)
            
            # room center
            Oroom = (self.h/2 * UP).array().flatten()
            Xprime = (position.flatten() - Oroom) # microphone location in room
            X = (self.sources[k].position.array().flatten() - Oroom) # talker location
            lLh = np.array([self.l, self.L, self.h])

            # signature function
            sigma = lambda n: -2.*(np.mod(n,2)-.5)

            i = 0
            for n in range(-nreflections, nreflections+1):
                for m in range(-nreflections, nreflections+1):
                    for p in range(-nreflections, nreflections+1):
                        nrefs = abs(m) + abs(n) + abs(p)
                        if nrefs <= nreflections:
                            tmp.nreflections[i] = nrefs
                            Rp = lLh.copy()
                            Rp[0] *= n
                            Rp[1] *= m
                            Rp[2] *= p
                            Vsource = np.array([sigma(n)*X[0], 
                                                sigma(m)*X[1], 
                                                sigma(p)*X[2]]) + Rp

                            if n == 0 and m == 0 and p == 0:
                                tmp.directions[:,i] = X - Xprime
                            else:
                                tmp.directions[:,i] = (Vsource - Xprime).flatten()
                            i = i + 1
            res = res.cat(tmp)
        res.normalize()
        return res

    # Geometrical Attributes
    def volume(self):
        '''
        Here it is computed exactly.
        '''
        return self.l*self.h*self.L
    
    def area(self, i):
        '''
        Here it is computed exactly.
        '''
        # surf order:
        # l r b f f c
        {
            'left': 0,
            'right': 1,
            'back': 2,
            'front': 3,
            'floor': 5,
            'ceiling': 6
            }
        areas = np.array([self.L*self.h, self.L*self.h, 
                          self.l*self.h, self.l*self.h,
                          self.l*self.L, self.l*self.L])
        return areas

# distance correction to avoid self-reflection
def correct_dists(dists, epsilon = DEFAULT_PRECISION):
    if not epsilon == 0:
        ncorr = len(np.nonzero(dists < epsilon)[0])
        if ncorr > 0:
            log_debug('Corrected '+str( ncorr / float(dists.shape[1]) * 100)+'% of distances')
            dists[dists < epsilon] = np.inf
        return dists
    else:
        return dists

