import numpy as np
from brian.units import *

__all__=[
    'Vector','Point',
    'getAzEl',
    'UP','DOWN','LEFT','RIGHT','FRONT','BACK',
    'ORIGIN',
    'az_spat2ircam', 'az_ircam2spat',
    'deg2rad', 'rad2deg',
    'crossproduct',
    'orthodromic_distance',
    'Az', 'degAz',
    'El', 'degEl',
    'Rotation', 'AzimuthRotation', 'ElevationRotation',
    'cartesian2spherical','spherical2cartesian']

########### Spherical Coords

def getAzEl(source,receiver):
    v = Vector(receiver.position,source.position)
    (r,theta,phi) = v.toSphericalCoords()
    return (theta,phi)
    
def getAzimuth(source,receiver):
    (theta,phi) = getAzEl(source,receiver)
    return theta
    
def getElevation(source,receiver):
    (theta,phi) = getAzEl(source,receiver)
    return phi

def spherical2cartesian(r,az,el):
    x = r*np.cos(np.pi/2+az)
    y = r*np.sin(np.pi/2+az)
    z = r*np.cos(np.pi/2-el)
    return (r*x, r*y, r*z)

def cartesian2spherical(*args, **kwdargs):
    '''
    Does the conversion between cartesian and spherical coordinates.
    Returns d, az, el, by default in radians but one can specifically ask for degrees by passing unit = 'deg'
    '''
    vectorized = kwdargs.get('vectorized', True)

    unit = kwdargs.get('unit', 'rad')
    conv = lambda x: x
    if unit == 'deg':
        conv = rad2deg
    
    if len(args) == 3:
        directions = np.array([args[0], args[1], args[2]])
        directions.shape = (3, 1)
        
    elif len(args) == 1:
        directions = args[0]
    else:
        raise ValueError('Wrong argument')
    
    if not vectorized:
        x = args[0]
        y = args[1]
        z = args[2]
        # this i leave in cause the algo is easier to read!
        r = np.sqrt(x**2 + y**2 + z**2)
        el = np.angle(np.sqrt(x**2 + y**2)/r+1j*z/r)
        az = cartesian2polar(x,y)
        return (float(r), float(conv(az)), float(conv(el)))
    else:
        d2 = directions**2
        dists = np.sqrt(np.sum(d2, axis = 0))
        tmp = np.sqrt(d2[0,:] + d2[1,:])
        el = np.angle((tmp+1j*directions[2,:])/dists)
        az = np.angle(np.exp(-np.pi/2*1j)*(directions[0,:] + 1j*directions[1,:]))
        return dists, conv(az), conv(el)

def cartesian2polar(x,y):
    z = x+1j*y
    z *= np.exp(-np.pi/2*1j)
    return np.angle(z)


######### Interpolation stuff
# radians/degrees conv
deg2rad = lambda x: x/180.*np.pi
rad2deg = lambda x: x/np.pi*180.

def vectfromspherical(az,el):
    return Vector(*spherical2cartesian(1,az,el))

deg2rad = lambda theta: theta/180.*np.pi
rad2deg = lambda theta: theta*180./np.pi

def az_spat2ircam(az): 
    '''
    Converts azimuth positions between spatializer and ircam systems
    '''
    if az < -180 or az > 180:
        raise ValueError('Wrong azimuth for conversion spatializer->IRCAM')
    if az == -180:
        az = 180
    return np.mod(-az,360)

def az_ircam2spat(az):
    '''
    Converts azimuth positions between ircam and spatializer systems
    '''
    if az < 0 or az > 360: 
        raise ValueError('Wrong azimuth for conversion IRCAM->spatializer')
    if az <= 180:
        return az
    else:
        return -(360-az)

def orthodromic_distance(a, b):
    '''
    computes the orthodromic distances of all the vectors in a (3xn) to the vector in b.
    '''
    # vect prod axb
    #
    # a2 b3 - a3 b2
    # a3 b1 - a1 b3
    # a1 b2 - a2 b1
    #
    # u1 u2 - u3 u4
    if isinstance(a, FloatTriplet):
        a = a.array()
    else:
        a = np.array(a)
    if isinstance(b, FloatTriplet):
        b = b.array()
    else:
        b = np.array(b)
    b = np.tile(b.reshape((3,1)), (1, a.shape[1]))
    
    crossprod = crossproduct(a, b)
    
    crossprodnorm = np.sqrt(np.sum(crossprod**2, axis = 0))
    return np.arctan2(crossprodnorm, np.sum(a*b, axis = 0))

def angular_distance(x, y):
    '''
    returns the angular distance of points in x relative to y
    output has len of x (as orthodromic_distance)
    '''
    y = np.tile(y.reshape((3,1)), (1, x.shape[1]))
    x_norm = np.sqrt(np.sum(x**2, axis = 0))
    y_norm = np.sqrt(np.sum(y**2, axis = 0))
    xy = np.sum(x*y, axis = 0)
    angle = arccos(xy/(x_norm*y_norm))
    return angle

def eculidean_distance(x, y):
    '''
    Returns the usual euclidean distance of points in y relative to x
    output has len of x
    '''
    y = np.tile(y.reshape((3,1)), (1, x.shape[1]))
    return np.sqrt(np.sum((x - y) ** 2, axis = 0))

def crossproduct(a, b):
    '''
    computes the cross product of the vectors in a and b, arranged by columns
    '''
    u1 = np.vstack((a[1, :],  a[2, :], a[0, :]))
    u2 = np.vstack((b[2, :],  b[0, :], b[1, :]))
    u3 = np.vstack((a[2, :],  a[0, :], a[1, :]))
    u4 = np.vstack((b[1, :],  b[2, :], b[0, :]))
    
    crossprod = u1*u2 - u3*u4
    
    return crossprod
    
    
########################## GENERAL GEOMETRY #########################

class FloatTriplet(object):
    '''
    Simple class to hold 3d coordinates, only used as a superclass for Points and Vectors that should not in principle be confounded.
    '''
    def __init__(self, input):
        if isinstance(input, FloatTriplet):
            self.coords = np.array([input.x, input.y, input.z]).reshape((3,1))
        elif len(input) == 3:
            self.coords = np.array(input).reshape((3,1))
        else:
            raise TypeError
        self.x = self.coords[0,0]
        self.y = self.coords[1,0]
        self.z = self.coords[2,0]



    def norm(self):
        return np.sqrt(self.x**2+self.y**2+self.z**2)

    def __str__(self):
        return '(%s,%s,%s)' % (self.x, self.y, self.z)
    
    def __repr__(self):
        return self.__str__()

    def __len__(self):
        return 3

    def array(self):
        return self.coords

    def tuple(self):
        return (self.x, self.y, self.z)
########### Vectors

class Vector(FloatTriplet):
    '''
    
    The vector object describes a Vector in 3D space.
    It can be initialized in various ways.
    
    By default uses usual 3D coordinate system. ``coordinates='cartesian'``
    But one can also set the ``cordinates`` keyword for other type of
    intialization, 
    'spherical' for spherical coordinates of the form (r,az,el), here
    azimuth vary from -pi (left) to +pi (right)
    'spherical_IRCAM' is provided to compensate for the IRCAM
    coordinate system (0-360 degrees, counterclockwise)
    
    Additionally, If the constructor is passed tow Points objects
    (A,B), then the vector represents the vector AB.
    
    '''
    def __init__(self, *args, **kwargs):
        try: kwargs['coordinates'] 
        except: kwargs['coordinates']='cartesian'
        if len(args)==3:
            if kwargs['coordinates']=='spherical':
                (x,y,z)=spherical2cartesian(args[0],args[1],args[2])
            elif kwargs['coordinates']=='spherical_IRCAM':
                (x,y,z)=spherical2cartesian(args[0],correct_az(args[1]),args[2])
            else:
                x=args[0]
                y=args[1]
                z=args[2]
            super(Vector,self).__init__([x,y,z])
        
        elif len(args)==2:
            if isinstance(args[0],Point) and isinstance(args[1],Point):
                self.x=args[1].x-args[0].x
                self.y=args[1].y-args[0].y
                self.z=args[1].z-args[0].z
            else:
                raise ValueError("Vector wasn't properly initialized ",*args)
        else:
            super(Vector,self).__init__(*args)

    def __str__(self):
        return 'Vector '+super(Vector,self).__str__()

    def __add__(self,other):
        if type(other)==Vector or type(other)==Point:
            return Vector(self.x+other.x,self.y+other.y,self.z+other.z)
        else:
            return TypeError

    def __sub__(self, other):
        return self.__add__(-other)

    def __mul__(self, other):
        if type(other) == Vector:
            return self.x*other.x+self.y*other.y+self.z*other.z
        elif isinstance(other, Rotation):
            return other.__mul__(self)
        else:
            return Vector(self.x*other,self.y*other,self.z*other)
    
    def __rmul__(self,other):
        return self.__mul__(other)

    def __div__(self,other):
        return Vector(self.x/other, self.y/other, self.z/other)

    def __neg__(self):
        return Vector(-self.x,-self.y,-self.z)
    
    def norm(self):
        return super(Vector,self).norm()
    
    def normalize(self):
        return self/self.norm()

    def reflectThrough(self,normal):
        d = normal.normalize()*(self*normal.normalize())
        return self - 2*d
    
    def toSphericalCoords(self, unit = 'rad'):
        conv = lambda x: x
        if unit == 'deg':
            conv = rad2deg
        d, az, el = cartesian2spherical(self.x, self.y, self.z)
        return (d, conv(az), conv(el))
        # #phi elevation
        # #theta azimuth
        # #r radius
        # # relou! see test_sphericalcoords
        # r=self.norm()
        
        # phi = np.pi/2-np.arccos(self.z/r )
        # if np.isnan(phi):
        #     phi=0
        
        # if self.y>=0:
        #     theta= np.pi/2-np.arccos(self.x/np.sqrt(self.x**2+self.y**2) )
        # else:
        #     if self.x<0:
        #         theta= -np.arccos(self.x/np.sqrt(self.x**2+self.y**2) ) 
        #     elif self.x>0:
        #         theta=np.pi-np.arccos(self.x/np.sqrt(self.x**2+self.y**2) ) 
        #     else:
        #         theta=-np.pi
        
        # if np.isnan(theta):
        #     theta=0
        # return (r,theta,phi)
    
    def vectorial_product(self, other):
        return Vector(self.y*other.z - self.z*other.y,
                      self.z*other.x - self.x*other.z,
                      self.x*other.y - self.y*other.x)
    
    def angle(self, other):
        if not isinstance(other, Vector):
            raise ValueError
        else:
            return np.arccos((self * other) / (self.norm() * other.norm()))
            

# useful vectors
   
UP=Vector(0,0,1)
DOWN=Vector(0,0,-1)
FRONT=Vector(0,1,0)
BACK=Vector(0,-1,0)
LEFT=Vector(-1,0,0)
RIGHT=Vector(1,0,0)


################## Angles

class Rotation(object):
    def __init__(self, value):
        self.value = value
        
    def __mul__(self, other):
        if isinstance(other, Vector):
            return Vector(self.value*other.array())
            
class AzimuthRotation(Rotation):
    def __init__(self, az, unit = 'rad'):
        if unit == 'deg':
            az = deg2rad(az)
        self.unit = unit
        self.az = az
        self.value = np.zeros((3,3))
        self.value[0,0] = np.cos(self.az)
        self.value[1,1] = np.cos(self.az)
        self.value[0,1] = -np.sin(self.az)
        self.value[1,0] = np.sin(self.az)
        self.value[2,2] = 1
        self.value = np.mat(self.value)
        
    def __mul__(self, other):
        if isinstance(other, Vector):
            return Vector(self.value*other.array())
        elif type(other) in [float, int]:
            if self.unit == 'deg':
                other = deg2rad(other)
            out = AzimuthRotation(other + self.az)
            return out

    def __lmul__(self, other):
        return self.__mul__(other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __add__(self, other):
        if isinstance(other, AzimuthRotation):
            return AzimuthRotation(self.az + other.az)
        if isinstance(other, ElevationRotation):
            return Rotation(self.value * other.value)
        if isinstance(other, Vector):
            return self.value*other.array()

class ElevationRotation(Rotation):
    def __init__(self, el, unit = 'rad'):
        if unit == 'deg':
            el = deg2rad(el)
        self.unit = unit
        self.el = el
        self.value = np.zeros((3,3))
        self.value[1,1] = np.cos(self.el)
        self.value[2,2] = np.cos(self.el)
        self.value[1,2] = -np.sin(self.el)
        self.value[2,1] = np.sin(self.el)
        self.value[0,0] = 1
        self.value = np.mat(self.value)
        
    def __mul__(self, other):
        if isinstance(other, Vector):
            return Vector(self.value*other.array())
        elif type(other) in [float, int]:
            if self.unit == 'deg':
                other = deg2rad(other)
            out = ElevationRotation(other + self.el)
            return out

    def __lmul__(self, other):
        return self.__mul__(other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __add__(self, other):
        if isinstance(other, ElevationRotation):
            return ElevationRotation(self.el + other.el)
        if isinstance(other, AzimuthRotation):
            return Rotation(self.value * other.value)

Az = AzimuthRotation(0)
degAz = AzimuthRotation(0, unit = 'deg')
El = ElevationRotation(0)
degEl = ElevationRotation(0, unit = 'deg')



########### Points

class Point(FloatTriplet):
    '''
    Simple class to represent points
    '''
    def __init__(self, input):
        super(Point,self).__init__(input)

    def __str__(self):
        return 'Point '+super(Point,self).__str__()
    
    def __add__(self,other):
        if type(other)==Vector:
            return Vector(self.x+other.x,self.y+other.y,self.z+other.z)
        else:
            return TypeError

ORIGIN=Point([0,0,0])


