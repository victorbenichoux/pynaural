import numpy as np
from scipy import weave
from scipy.weave import converters

from pynaural.raytracer.geometry.rays import Beam
from pynaural.raytracer.geometry.base import Point, Vector, FRONT, BACK, LEFT, RIGHT, UP, DOWN, ORIGIN
from pynaural.utils.debugtools import log_debug
from pynaural.utils.spatprefs import get_pref


__all__=['Surface',
         'Sphere',
         'Plane',
         'GROUND']


########################## SURFACES #########################

class Surface(object):
    '''
    Parent class to work with geometrical surfaces.

    ** Methods ** 
    
    .. automethod :: reflect

    ** Advanced: Primary Methods ** 

    Those two methods should be instantiated when creating a new
    Surface object (and please feel free to do so).
    
    .. automethod :: intersection
    .. automethod :: normal_at

    This might also be of some interest in (advanced) applications:
    .. automethod :: prepare

    ** Secondary methods ** 

    .. automethod :: contains
    .. automethod :: is_in_updomain

    '''
    def __init__(self, model = None, label = None):
        self.model = model
        self.label = label
    
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

    def contains(self,other):
        '''
        Returns whether or not other is contained on that surface
        '''
        pass

    def is_in_updomain(self, other):
        '''
        Returns whether or not other is contained in the 'up domain'
        of the surface. This method is used to compute the volume of
        the inside of a closed GeometricScene. 
        '''
        pass
    
    def normal_at(self,point):
        '''
        Returns the normal vector at the given point (given as a
        Point) or points (given as (3xn) ndarrays) on the surface. 
        '''
        pass

    def intersection(self, beam):
        '''
        Returns the distances at which the rays in the Beam reach the
        surface. If the ray doesn't intersect the surface, then np.inf
        must be returned for the rendering to work.
        '''
        pass
    
    def reflect(self, *args):
        ''' 
        Returns a new Beam object that represents the efferent
        reflected Beam. This method uses the specialized methods
        intersection and normal_at.
        
        Usage:
        either reflect(beam) 
        or 
        reflect(directions, intersectionpoints)
        
        The second form is used internally to avoid re-computing the
        intersection distances.
        
        '''
        if len(args) == 1:
            beam = args[0]
            dists=self.intersection(beam)
            if not np.isfinite(dists).all():
                print 'intersection with wrong beam'
            nrays = beam.nrays
            res = Beam(nrays)
            res.origins = beam.origins + beam.directions*np.tile(dists,(3,1))
            d = self.normals[:,:nrays]*np.tile(np.sum(beam.directions*self.normals[:,:nrays],axis=0),(3,1))
            res.directions = beam.directions-2*d
            return res
        elif len(args) == 2:
            directions = args[0]
            intersectionpoints = args[1]
            nrays = directions.shape[1]
            if directions.shape[1] != intersectionpoints.shape[1]:
                raise ValueError('must specify the same number of points and directions')
            res = Beam(directions.shape[1])
            res.origins = intersectionpoints
            d = self.normals[:,:nrays]*np.tile(np.sum(directions*self.normals[:,:nrays],axis=0),(3,1))
            resdirections = np.zeros_like(directions)
            res.directions = directions-2*d
            return res


    def get_incidence(self, *args):
        '''
        Returns the incidence angle of the ray/beam object with the surface.
        To avoid recomputing the intersection points, one can call
        this function with two  (ndarray) arguments, first one being
        the directions, and the second one being the intersection
        points.
        
        '''
        if len(args)==1:
            ray=args[0]
            if isinstance(ray,Beam):
                dists = self.intersection(ray)
                return self.get_incidence(ray.directions, ray.atDist(dists))
        elif len(args) == 2:
            directions = args[0]
            intersectionpoints = args[1]
            if not directions.shape == intersectionpoints.shape:
                raise ValueError('must specify the same number of points and directions')
            normals = self.normal_at(intersectionpoints)
            return (np.arccos( np.abs( np.sum( normals * directions ,axis=0)) ) 
                    % np.pi ) / np.pi*180
        
    def prepare(self, nrays):
        '''
        This method *prepares* the surface for vectorized rendering with a given
        number of rays. Indeed to avoid the time loss of
        re-initializing big vectors (for example repeated
        versions of the normal in a Place) at each rendering step,
        those are initialized once in for all. 
        
        
        The argument is the number of rays that will be passed every
        time. One may want to use that when creating new Surface
        objects, for example in the Plane object, calling prepare
        creates a (3 x nrays) array with each column being the normal.
        '''
        pass
    
    def __str__(self):
        lbl = self.label or str(self.id) or ''
        return 'Surface '+lbl

class Sphere(Surface):
    '''
    Spherical surface.
    
    Initialization:
    `` center `` self explanatory
    
    `` radius `` self explanatory
    '''
    def __init__(self, center, radius, **kwargs):
        self.center=center
        self.radius=radius
        super(Sphere, self).__init__(**kwargs)

    def contains(self,point):
        if Vector(self.center,point).norm()==self.radius:
            return True
        else:
            return False

    def normal_at(self,point):
        if isinstance(point,Point):
            n=Vector(self.center,point)
            return n.normalize()
        elif isinstance(point, np.ndarray):
            tmp = np.tile(self.center.array(),(1,point.shape[1]))
            normals = point-tmp       
            normals = normals/np.tile(np.sqrt(np.sum(normals*normals,axis=0)),(3,1))
            return normals
    
    def intersection(self, ray, use_weave = None):
        '''
        Computes the intersection of the incoming Beam object with the sphere
        '''
        if use_weave is None:
            use_weave = get_pref('USE_WEAVE', False)

        if isinstance(ray, Beam):
            if not use_weave:
                res = np.inf*np.ones(ray.nrays)
                tmp = np.tile(self.center.array().reshape((3,1)), (1, ray.nrays))
                v = ray.origins - tmp
                b = 2*(np.sum(v*ray.directions, axis=0))
                c = np.sum(v*v,axis=0) - self.radius**2
                delta = b**2-4*c
                deltapos = np.nonzero(delta>=0)[0]
                alpha = (-b[deltapos]-np.sqrt(delta[deltapos]))/2.
                alphasneg = np.nonzero(alpha<0)
                alpha[alphasneg] = np.inf
                # is the alpha> condition really necessary? could be handled later
                alphapos = np.nonzero(np.isfinite(alpha))[0]
                res[deltapos[alphapos]] = alpha[alphapos]
                return res
            else:
                nrays = ray.origins.shape[1]
                ncoords = 3
                
                origins = ray.origins
                directions = ray.directions
                radius = self.radius
                center = self.center.array().flatten()
                
                res = np.inf*np.ones(ray.nrays)
                probe = 0
                if not hasattr(self, '_intersect_code'):
                    log_debug('Using weave')
                    self._intersect_code = """
#include <math.h>
int coord, ray;
double v, b, c;
double alpha, delta;
double sum, sum2;
double tmp;

for(ray = 0; ray < nrays; ray++)
{

sum = 0;
sum2 = 0;

for(coord = 0; coord < ncoords; coord++) 
{

v = origins(coord, ray) - center(coord);
sum += v*directions(coord, ray);
sum2 += v*v;

}

b = 2*sum;
c = sum2 - radius*radius;
delta = b*b - 4*c;


if (delta >= 0)
{

alpha = (- b - sqrt(delta))/2.;

if (alpha >0)
{

res(ray) = alpha;

}

}

}

"""
                    
                err = weave.inline(self._intersect_code, ['nrays', 'ncoords',
                                                          'radius',
                                                          'origins', 'directions', 
                                                          'center', 'res'],
                                   type_converters = converters.blitz, compiler='gcc')
                return res
            

class Plane(Surface):
    '''
    Plane surface.
       
    Initialization:

    `` orig `` self explanatory
    
    `` normal `` self explanatory

    '''
    def __init__(self, orig, normal, **kwargs):
        self.norm = normal.normalize()
        self.orig = orig
        super(Plane, self).__init__(**kwargs)
        
    def __str__(self):
        if self.label == None:
            return 'Plane: \n orig:' + str(self.orig) + '\n normal:' + str(self.norm)
        else:
            return super(Plane, self).__str__()

    def contains(self, point):
        if type(point) == Point:
            if Vector(self.orig,point)*self.norm==0:
                return True
            else:
                return False
        elif type(point) == np.ndarray:
            if not hasattr(self, 'origins'):
                self.prepare(point.shape[1])
            sp = point - self.origins
            res = np.sum(sp*self.normals, axis = 0) == 0
            return res
    
    def normal_at(self,point):
        if isinstance(point,Point):
            return self.norm
        elif isinstance(point,np.ndarray):
            return np.tile(self.norm.array(),(1,point.shape[1]))
    
    def intersection(self, ray, use_weave = None):
        if use_weave is None:
            use_weave = get_pref('USE_WEAVE', False)


        if not hasattr(self, 'origins'):
            self.prepare(ray.nrays)

        if False:
            res = np.inf*np.ones(ray.nrays)
            print "using this"
            Nv = np.sum(self.normals[:,:ray.nrays] * ray.directions, axis = 0)
            Nqp = np.sum(self.normals[:,:ray.nrays] * (self.origins[:,:ray.nrays]-ray.origins), axis = 0)
            d = Nqp/Nv
            dpos = np.nonzero(d > 0)[0]
            noninf = np.nonzero(np.isfinite(d[dpos]))[0]
            res[dpos[noninf]] = d[dpos[noninf]]
            res[res == -1] = np.inf
            res[res == -np.inf] = np.inf
            return res
        if not use_weave:
            nrays = ray.nrays
            res = np.inf*np.ones(ray.nrays)
            sa = ray.origins - self.origins[:,:nrays]
            sn = self.origins[:,:nrays] - self.normals[:,:nrays]
            d = -np.sum(sa*sn, axis = 0)/(np.sum(ray.directions*sn,axis=0))
            dpos = np.nonzero(d > 0)[0]
            noninf = np.nonzero(np.isfinite(d[dpos]))[0]
            res[dpos[noninf]] = d[dpos[noninf]]
            res[res == -1] = np.inf
            res[res == -np.inf] = np.inf
            return res

        else:
            nrays = ray.origins.shape[1]
            ncoords = 3
                                
            origins = ray.origins
            directions = ray.directions
            plane_origins = self.origins
            normals = self.normals

            res = np.inf*np.ones(nrays)

            if not hasattr(self, '_intersect_code'):
                log_debug('Using weave')
                self._intersect_code = """
int coord, ray;
double sa, sn;
double d;
double sum, dirsum;

for(ray = 0; ray < nrays; ray++)
{

sum = 0;
dirsum = 0;

for(coord = 0; coord < ncoords; coord++) 
{

sa = origins(coord, ray) - plane_origins(coord);
sn = plane_origins(coord) - normals(coord);
dirsum += directions(coord, ray)*sn;
sum += -sa*sn;

}

d = sum/dirsum;

if (d > 0)
{

res(ray) = d;

}

}


"""
            err = weave.inline(self._intersect_code, ['nrays', 'ncoords',
                                                      'plane_origins', 'normals',
                                                      'origins', 'directions', 
                                                      'res'],
                               type_converters = converters.blitz, compiler='gcc')
            return res
    
    def prepare(self, nrays):
        self.normals = np.tile(self.norm.array(),(1,nrays))
        self.origins = np.tile(self.orig.array(),(1,nrays))

    def is_in_updomain(self, points):
        '''
        plane.is_in_updomain(points) where points is (3,n) points returns a (n,) array with 
        True is the corresponding point was in the domain where point.normal >=0
        '''
        normals = np.tile(self.norm.array(), (1, points.shape[1]))
        origins = np.tile(self.orig.array(),(1, points.shape[1]))
        return (np.sum((points-origins)*(normals-origins), axis = 0) >=0 )

GROUND = Plane(ORIGIN,UP)
