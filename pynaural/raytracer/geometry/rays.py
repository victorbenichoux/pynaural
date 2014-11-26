import numpy as np
import warnings
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from pynaural.raytracer.geometry.base import Point, Vector, FRONT, BACK, LEFT, RIGHT, UP, DOWN, ORIGIN
from pynaural.raytracer.acoustics import c

__all__ = ['Beam',
           'ScreenBeam', 'SphericalBeam',
           'DummyBeam', 'BaseBeam', 'RandomSphericalBeam']

########### Beams #########################
class Beam(object):
    '''
    Class that describes a chained list of bundle of rays. It is both
    the input and the output of the GeometricScene's render method.
    
    ** Initialization ** 

    Initializes a blank bundle of `` nrays `` rays.
    
    ** Initial Attributes **
    
    Those attributes form the minimal set of attributes that a Beam
    requires. They describe simple properties of the Beam object.
    The following two attributes are (3 x nrays) ndarrays.

    `` origins `` The coordinates of the origins of each ray.
    `` directions `` The direction of the ray.
    
    `` target_source `` Contains the Source id of the target of the given
    ray. If it is not set then the ray will stop whenever it reaches a
    source during rendering.

    ** Rendered Attributes **

    Those attributes are instantiated when the beam is the result of
    rendering in a GeometricScene. 
    
    `` sourcereached `` Whether of not the ray has reached its source.
    `` surfacehit `` Which surface was hit by the current ray.
    `` traveleddistance `` Total distance traveled by the ray
    sequence since the first one

    .. autoattribute:: reflectiondepth
    


    Those last two attributes are those handled by the Model classes:
    `` distances `` Travelled distance (in meters) since last Surface
    was hit.
    `` incidences `` Incidence angle of the ray on the Surface it just hit.

    Those attributes and methods find the rays that reached their
    intended source (or not).
    .. autoattribute :: nreachedsource    
    .. automethod:: get_reachedsource_index
    .. automethod:: get_reachedsource
    .. automethod:: get_notreachedsource_index
    .. automethod:: get_notreachedsource
    
    .. automethod:: get_reachedsource_depth
    .. automethod:: get_finalbeam

    ** Chained list structure ** 
    
    Beams, when rendered through the render method of a
    GeometricScene, are arranged as a chained list. Hence each Beam
    object points to a next Beam until rendering depth is reached.

    `` next `` Points to the next Beam object in the sequence.
    .. autoattribute :: depth

    Fetching values further in the list structure
    .. automethod :: get_origins
    .. automethod :: get_directions

    ** Plotting, Misc. ** 

    .. automethod :: plot
    .. automethod :: plot_echogram
    .. automethod :: cat


    '''    
    
    def __init__(self, nrays, other = None, target_source = -1):
        self.nrays = nrays
        # INIT geometric data
        self.directions = np.nan*np.ones((3,nrays))
        self.origins = np.nan*np.ones((3,nrays))
        self.target_source =  target_source*np.ones(nrays, dtype = int)
        # other geometric data
        self.sourcereached = np.zeros(nrays, dtype = bool)
        self.surfacehit = -1*np.ones(nrays, dtype = int)
        self.traveleddistance = np.zeros(nrays)
        # data for model computation
        self.distances = np.nan*np.ones(nrays)
        self.incidences = np.nan*np.ones(nrays)
        # chained list structure
        self.next = other
    
    def hasNext(self):
        return not(self.next is None)
        
    def append(self,other):
        self.next = other
    
    def get_totaldists(self):
        d = self.distances.copy()
        d[np.nonzero(np.isnan(d))[0]] = 0
        if not self.hasNext():
            return d
        else:
            return d + self.next.get_totaldists()
    
    def get_totaldelays(self):
        return self.get_totaldists()/c

    ################# nfft detection    
    def detect_nfft(self, samplerate):
        # automatic detection of nfft
        tmax = np.max(self[self.get_reachedsource_index()].get_totaldelays())
        nfft = np.ceil(tmax*samplerate)
        return nfft
    
    def __getitem__(self,rayslice):
        if isinstance(rayslice, np.ndarray):
            if rayslice.dtype == bool:
                nrays = len(np.nonzero(rayslice)[0])
            else:
                nrays = len(rayslice)
        else: 
            try:
                nrays = len(self.directions[0,rayslice])
            except TypeError:
                nrays = 1
        res = Beam(nrays)    
        res.directions = np.array(self.directions[:,rayslice]).reshape((3,nrays))
        res.origins = np.array(self.origins[:,rayslice]).reshape((3,nrays))
        res.surfacehit =  np.array([self.surfacehit[rayslice]]).flatten()
        res.sourcereached = np.array([self.sourcereached[rayslice]]).flatten()
        res.incidences = np.array([self.incidences[rayslice]]).flatten()
        res.distances = np.array([self.distances[rayslice]]).flatten()
        res.target_source = np.array([self.target_source[rayslice]]).flatten()
        if self.hasNext():
            res.next = self.next[rayslice]
        else:
            res.next = None
        return res

    def forsource(self, k):
        '''
        Returns the part of the beam that is targetted towards source with id k
        '''
        return self[self.target_source == k]
    
    def atDist(self,dists):
        if len(dists) != self.nrays:
            raise ValueError('wrong number of distances')
        else:
            return self.origins + self.directions * np.tile(dists, (3, 1))

    # Sure this needs to be here??
    def norm(self):
        ''' 
        Returns the norms of the directions
        '''
        return np.sqrt(np.sum(self.directions*self.directions,axis=0))
    
    def normalize(self):
        self.directions /= np.tile(self.norm(),(3,1))

    #################### DEPTH
    @property    
    def depth(self):
        '''
        The depth of the Beam, that is the length of the chained list
        structure. This doesn't take into account wheter or not the
        ray reached the source, please note the difference with reflectiondepth.
        '''
        if self.next is None:
            return 1
        else:
            return 1 + self.next.depth

    def _get_rs_depth(self):
        res = np.ones(self.nrays, dtype = int)
        res[self.sourcereached] = 0
        if self.hasNext():
            return res + self.next._get_rs_depth()
        else: 
            return 0
    
    def get_reachedsource_depth(self):
        rs = self.get_reachedsource()
        if not rs.all():
            res = np.inf*np.ones(self.nrays)
            res[rs] = self[rs]._get_rs_depth()
        else:
            res = self._get_rs_depth()
        return res

    @property
    def reflectiondepth(self):
        '''
        Returns the reflection depth for each ray of the beam, that is
        the number of reflections before it actually reached its
        targetted source plus one. It is np.inf if the source wasn't reached.
        '''
        return self.get_reachedsource_depth()

    ################### REACHED SOURCE
    def get_reachedsource(self):
        '''
        Wether of not the rays in the Beam reached their source at any
        depth in the chained list. Note the difference with the
        attribute sourcereached.
        Returns an array with boolean values.
        '''
        if self.hasNext():
            return self.sourcereached + self.next.get_reachedsource()
        else: 
            return self.sourcereached

    def get_reachedsource_index(self):
        '''
        Returns the indices of the Trues in self.get_reachedsource()
        '''
        return np.nonzero(self.get_reachedsource())[0]
    
    def get_notreachedsource(self):
        '''
        Returns -self.get_reachedsource()
        '''
        return -self.get_reachedsource()

    def get_notreachedsource_index(self):
        '''
        Returns the indices of the Trues in self.get_notreachedsource()
        '''
        return np.nonzero(self.get_notreachedsource())[0]

    @property
    def nreachedsource(self):
        '''
        Returns the number of rays that reached their intended source
        in this Beam.
        '''
        return len(self.get_reachedsource_index())

    def get_finalbeam(self):
        '''
        Returns a beam containing the last step for all paths that
        reached the source. Those may not be at the same depth, of
        course, and all rays returned are pointing TO the source.
        '''
        # who reached the source?
        rs = self[self.get_reachedsource()]
        # at what depth was source reached?
        depth = rs.get_reachedsource_depth()
        print depth
        # result
        res = Beam(rs.nrays)
        res.origins = rs.get_origins(depth)
        # we want the origins of the last ray, but the directions
        # of the previous one!
        ds = np.zeros((2, depth.shape[0]))
        ds[0,:] = depth - 1
        depth = np.max(ds, axis = 0)
        res.directions = rs.get_directions(depth)
        return res

    ######################### Chained List stuff
    # helper function
    def _get_at_depth(self, depth, attribute):
        if isinstance(depth, (int, float)):
            depth = np.array([depth])
        
        if (depth == 0).all():
            return getattr(self, attribute)
        else:
            if not self.hasNext():
                raise IndexError('Depth specified is too high')
            res = np.zeros((3, self.nrays))
            zero_depth = (depth == 0)
            zero_depth_idx = np.nonzero(zero_depth)[0]
            one_depth_idx = np.nonzero(-zero_depth)[0]
            res[:, zero_depth_idx] = getattr(self[zero_depth_idx], attribute)
            res[:, one_depth_idx] = self[one_depth_idx].next._get_at_depth(
                depth[one_depth_idx]-1, attribute)
            return res
        
    def get_origins(self, depth):
        '''
        Gets the values of the origins attribute of the depth'th
        Beam in the chained list.
        '''
        return self._get_at_depth(depth, 'origins')
            
    def get_directions(self, depth):
        '''
        Gets the values of the directions attribute of the depth'th
        Beam in the chained list.
        '''
        return self._get_at_depth(depth, 'directions')
        
    def getpath(self, i, depth = None):
        path=[self.origins[:,i]]
        next=self.next
        i=1
        while (not next is None):
            if depth is not None and i>=depth:
                break
            path.append(next.origins[:,i])
            next=next.next
            i+=1
        n=len(path)
        res=np.zeros((3,n))
        for i in range(n):
            res[:,i]=path[i]
        return res
    
    def cat(self,other):
        '''
        An important feature for the inner working of the spatializer!
        It concatenates two Beams together. 
        '''
        if self.nrays == 0:
            return other
        res = Beam(self.nrays+other.nrays)
        res.directions = np.hstack((self.directions, other.directions))
        res.origins = np.hstack((self.origins, other.origins))
        res.sourcereached = np.hstack((self.sourcereached, other.sourcereached))
        res.surfacehit = np.hstack((self.surfacehit, other.surfacehit))
        res.distances = np.hstack((self.distances, other.distances))
        res.incidences = np.hstack((self.incidences, other.incidences))
        res.target_source = np.hstack((self.target_source, other.target_source))
        if self.next is not None:
            res.next = self.next.cat(other.next)
        return res


    def set_target(self, source):
        '''
        argument: source
        Sets the target source of all rays to the source in question
        '''
        self.target_source = source.get_id() * np.ones(self.nrays, dtype = int)

    def __eq__(self,other):
        if isinstance(other,Beam):
            return ((other.target_source == self.target_source).all() and (other.directions == self.directions).all() and (other.origins == self.origins).all() and (other.sourcereached == self.sourcereached).all()) and (self.next == other.next)
        else:
            return False

    ######################### Plotting
    def plot(self, recursive = False, axis = None, display = False, 
             colors = None):
        '''
        Plots the current Beam object. The origins are plotted with a
        round tick and the direction is drawn with a segment going
        away from it.
        
        Please note that the plotting of very big Beams may not work
        due to matplotlib issues.
        
        Arguments:

        `` recursive = False `` Recursively call the next beam's plot method
        if set to True

        `` axis = None `` Precises which Axes object to plot on.

        `` display = False `` Calls show() if set to True
        
        '''
        
        if axis is None:
            ax = plt.subplot(111, projection = '3d')
        else:
            ax = axis
        ax.scatter(self.origins[0,:], self.origins[1,:], self.origins[2,:], '.')
        dirs = self.origins + self.directions
        try:
            sr = self.get_reachedsource()
        except AttributeError:
            sr = self
        for i in range(int(self.nrays)):
            if not self.sourcereached[i]:
                if self.hasNext():
                    Xs = np.array([self.origins[0,i], self.next.origins[0,i]])
                    Ys = np.array([self.origins[1,i], self.next.origins[1,i]])
                    Zs = np.array([self.origins[2,i], self.next.origins[2,i]])
                else:
                    Xs = np.array([self.origins[0,i], dirs[0,i]])
                    Ys = np.array([self.origins[1,i], dirs[1,i]])
                    Zs = np.array([self.origins[2,i], dirs[2,i]])
                if colors is None:
                    if sr[i]:
                        col = 'g'
                    else:
                        col = 'r'
                    ax.plot(Xs, Ys, Zs, col+'-')
                else:
                    ax.plot(Xs, Ys, Zs, c = colors[i])
        plt.ylabel('Front/Back')
        plt.xlabel('Left/Right')
#        plt.zlabel('Up/Down')
        if recursive:
            if self.hasNext():
                self.next.plot(recursive = True, display = display)
        if display:
            plt.show()
        return ax
    
    def plot_echogram(self, display = False):
        '''
        Plots a simple echogram derived from the Beam object only,
        that is it will display the histogram of reflectiondepths.
        '''
        reached_id = self.get_reachedsource_index()
        reached = self[reached_id]
        reached_depth = reached.get_reachedsource_depth()
        reached_time = reached.get_totaldelays()
        plt.plot(reached_time*1e3, reached_depth, 'o')
        plt.xlabel('Propagation time to source (ms)')
        plt.ylabel('Number of reflections')
        
        if display:
            plt.show()
    
########### Geometric Beams

class SphericalBeam(Beam):
    '''
    Generates a spherical beam using a method based on the golden number!
    It is completely deterministic.
    
    ** Initialization **

    `` position `` The position of the origin of all rays in the Beam
    `` nrays `` Number of rays to generate
    '''
    def __init__(self, position, nrays, **kwdargs):
        if 'vectorized' in kwdargs:
            vectorized = kwdargs['vectorized']
            del kwdargs['vectorized']
        else:
            vectorized = True

        super(SphericalBeam,self).__init__(nrays, **kwdargs)

        self.directions = np.zeros((3,nrays))
        if isinstance(position, Point) or isinstance(position, Vector):
            position = position.array()
        position = position.reshape((3,1))
        self.origins = np.tile(position, (1,nrays))
        self.nrays = float(nrays)
        self.compute_rays(vectorized = vectorized)

    def compute_rays(self, vectorized = True):
        inc = np.pi*(3-np.sqrt(5))
        off = 2/self.nrays
        if not vectorized:
            # VERY unefficient
            for k in range(0, int(self.nrays)):
                y = k*off-1+off/2
                r = np.sqrt(1-y*y)
                phi = k*inc
                v = Vector(np.cos(phi)*r , y, np.sin(phi)*r)
                self.directions[:,k] = (v / v.norm()).array().flatten()
        else:
            ks = np.arange(self.nrays)
            self.directions[1,:] = ks*off - 1 + off/2
            r = np.sqrt(1-self.directions[1,:]**2)
            phi = ks*inc
            self.directions[0,:] = np.cos(phi)*r
            self.directions[2,:] = np.sin(phi)*r


class RandomSphericalBeam(Beam):
    '''
    Generates a beam with rays starting at point position and directions uniformly distributed on a unit sphere.
    

    ** Initialization **

    `` position `` The position of the origin of all rays in the Beam
    `` nrays `` Number of rays to generate

    '''
    def __init__(self, position, nrays, **kwdargs):
        super(RandomSphericalBeam, self).__init__(nrays, **kwdargs)

        self.directions = np.zeros((3, nrays))
        if isinstance(position, Point) or isinstance(position, Vector):
            position = position.array()
        position = position.reshape((3,1))
        self.origins = np.tile(position, (1,nrays))
        self.nrays = nrays
        self.compute_rays()

    def compute_rays(self):
        directions = (np.random.rand(3, self.nrays) - 1./2)*2
        norm = np.sqrt(np.sum(directions ** 2, axis = 0))
        while not (norm <= 1).all():
            bad_idx = np.nonzero(norm >1)[0]
            directions[:, bad_idx] = (np.random.rand(3, len(bad_idx)) - 1./2)*2
            norm = np.sqrt(np.sum(directions ** 2, axis = 0))
        self.directions = directions/norm

class ScreenBeam(Beam):
    '''
    Initialized with four vectors defining the screen corners
    '''
    def __init__(self, position, tl, tr, br, bl,nraysx,nraysy):
        super(SphericalBeam,self).__init__(nraysx*nraysy)
        nraysx=float(nraysx)
        nraysy=float(nraysy)
        self.directions=np.zeros((3,self.nrays))  
        self.origins=np.tile(position.array(),(1,nrays))
        for x in range(nraysx):
            for y in range(nraysy):
                v=tl*(1-x/nraysx)*(1-y/nraysy)+tr*x/nraysx*(1-y/nraysy)+br*x/nraysx*y/nraysy+tr*(1-x/nraysx)*y/nraysy
                self.directions[x+y*nraysx]=(v/v.norm()).array()

class DummyBeam(Beam):
    '''
    A *dummy* Beam, with all rays having the same starting point and direction.
    
    ** Initialization **

    `` position`` starting point of the rays
    `` direction`` direction of the rays
    `` nrays = 1 `` Number of rays
    '''
    def __init__(self, position, direction, nrays = 1):
        super(DummyBeam, self).__init__(nrays)
        self.directions=np.tile(direction.array(),(1,nrays))
        self.origins=np.tile(position.array(),(1,nrays))
        self.normalize()

class BaseBeam(Beam):
    '''
    A base Beam, with all rays having the same starting point and directions are the base vectors with their opposite, arranged in the order of dimensions (that is RIGHT / LEFT, FRONT / BACK, UP / DOWN).
    Initialized with `` position``, the starting point of the rays.
    '''
    def __init__(self, position):
        super(BaseBeam, self).__init__(6)
        self.origins=np.tile(position.array(),(1,self.nrays))
        self.directions=np.hstack((RIGHT.array(), LEFT.array(),
                                FRONT.array(), BACK.array(),
                                UP.array(), DOWN.array()))
        self.normalize()

