from scipy.special import *
from numpy.fft import *
from ..geometry import *
from ..acoustics import c # speed of sound
from .misc import zeropad, dBconv, fullrangefreq #
from .smoothing import apply_windowing
from .impulseresponse import ImpulseResponse, TransferFunction
from brian.stdunits import *
import numpy as np
from brian import *

MAX_ITER = 1e7

class HRTFModel(object):
    pass

class SphericalHead(HRTFModel):
    '''
    * Spherical Head Model * 
    
    Implements the Spherical Head model by (Duda and Mertens 1998). 

    Initialized with the iad and ear elevation and azimuths in degrees, as follows:
    
    head = SphericalHead(20*cm, (3, 3))

    * Initialization *
    
    ``head = SphericalHead(radius, (earazimuth, earelevation))``
    Where:
    
    ``earazimuth`` is the displacement in degrees of the ear with regards to the usual (+-90 degrees) position. Positive values will put the ears more in the front.
    
    ``earelevation`` is the elevation of the ears, in degrees as well
    
    The following two keyword arguments may be overriden by the
    user later in the get_hrtf method.
        
    `` samplerate `` Sets the samplerate of all the output TF/IR s
    `` nfft `` Sets the nfft of all the output TF/IR s

    * Methods *
    
    .. automethod:: get_hrtf
    .. automethod:: get_hrir
    
    * Attributes *
    
    .. autoattribute:: normalizedtime
    .. autoattribute:: normalizedfrequency
    
    '''
    def __init__(self, iad, earpos,
                 nfft = 1024, samplerate = 44100.*Hz):
        
        if isinstance(earpos, tuple):
            earelevation, earazimuth = earpos
        else:
            earelevation = 0
            earazimuth = earpos
            
        self.a = float(iad)/2
        self.elear = float(earelevation)
        self.azear = float(earazimuth)
        
        self.nfft = nfft
        self.samplerate = samplerate

    def get_hrtf(self, az, el, **kwdargs):
        '''
        Returns the HRTF at the position specified in the
        argument. The output is a TransferFunction object. One can get
        an ImpulseResponse with get_hrir.
        
        Arguments: 
        
        `` az, el`` azimuth and elevation of the source, in degrees.
        
        Additional keywords:

        `` distance = 2*meter `` The distance at which the source is placed.
        
        `` nfft `` Overrides the nfft attribute of the SphericalHead
        if set.
        
        `` samplerate `` Overrides the samplerate attribute of the SphericalHead
        if set.
        
        Suggested usage:
        get_hrtf(az, el, distance = 2*meter, method = 'new', order =
        1e-20)
        '''
        nfft = kwdargs.get('nfft', self.nfft)
        
        kwdargs['ear'] = 'left'
        left = self._get_single_tf(az, el, **kwdargs)

        kwdargs['ear'] = 'right'
        right = self._get_single_tf(az, el, **kwdargs)

        both = np.hstack((left.reshape((nfft, 1)), right.reshape((nfft, 1))))

        tf = TransferFunction(both, 
                              samplerate = self.samplerate,  
                              binaural = True,
                              coordinates = (az, el))
        return tf

    def get_hrtfs(self, azs, **kwdargs):
        '''
        generates hrtfs for the given list of azimuhts.
        elevation is by default 0 but can be specified (albeit constant) with elev = ...
        '''
        el = kwdargs.pop('elev', 0)
        nfft = kwdargs.get('nfft', self.nfft)

        data = np.zeros((nfft, len(azs)*2), dtype = complex)
        coords = np.zeros(len(azs), dtype = [('azim','f8'), ('elev','f8')])
        for kaz,az in enumerate(azs):
            kwdargs['ear'] = 'left'
            data[:,kaz] = self._get_single_tf(az, el, **kwdargs).flatten()
            kwdargs['ear'] = 'right'
            data[:,kaz+len(azs)] = self._get_single_tf(az, el, **kwdargs).flatten()

        coords['azim'] = azs
        coords['elev'] = np.ones(len(azs)) * el

        tf = TransferFunction(data, 
                              samplerate = self.samplerate,  
                              binaural = True,
                              coordinates = coords)
        return tf

    def get_hrir(self, az, el, **kwdargs):
        '''
        Time domain counterpart of get_hrtf. See the documentation for
        get_hrtf for more details.
        
        * Output delay * 
        
        In order to avoid aliasing in the output, one can delay the ImpulseResponse
        output with the following keyword argument:
        
        `` pre_delay = 0`` Delays the IR output by an integer number
        of samples. Uses numpy's roll function on the IR
        '''
        h = self.get_hrtf(az, el, **kwdargs)

        pre_delay = kwdargs.get('pre_delay', 64)

        ir = ifft(asarray(h), axis = 0).real
        ir = roll(ir, pre_delay, axis = 0)
        
        irp = vstack((zeros((1,2)), ir[:-1,:]))
        irm = vstack((ir[1:,:], zeros((1,2))))
        
        ir = 0.5*(2*ir+irp+irm)
        
        return ImpulseResponse(ir, **h.get_kwdargs())
        
    def _get_single_tf(self, az, el, distance = 2*meter, 
                       ear = 'left', 
                       pre_delay = None,
                       nfft = None, samplerate = None,
                       method = 'old', order = 75, threshold = 1e-15):
        ## This is where the computation is actually carried out
        # KWDARG parsing
        if nfft is None:
            nfft = self.nfft

        if samplerate is None:
            samplerate = self.samplerate
        
        ## main parameters of the model
        # cosine of incidence angle
        x = self._cos_incidence(az, el, ear = ear)
        # normalized distance
        rho = float(distance) / self.a
        # frequencies
        f = fftfreq(nfft)*self.samplerate
        
        f[f<0] = -f[f<0]
        
        a = float(self.a)
        r = float(distance)

        mu = (2 * pi * f * a) / c
        rho = r / a
        zr = 1. / (1j * mu * rho)
        za = 1. / (1j * mu)
        Qr2 = zr
        Qr1 = zr * (1. - zr)
        Qa2 = za
        Qa1 = za * (1. - za)
        P2 = 1.
        P1 = x
        sum = 0.
        term = zr / (za * (za - 1.))
        sum = sum + term
        term = (3 * x * zr * (zr - 1) ) / (za * (2 * za**2 - 2 * za + 1) )
        sum = sum + term
        oldratio = 1.
        newratio = abs(term)/abs(sum)
        m = 2.
        
        condition = ones(f.shape, dtype = bool)
        
        while condition.any():
            if m> MAX_ITER:
                print 'Maximum iteration number reached'

            Qr = - (2 * m - 1) * zr * Qr1 + Qr2
            Qa = - (2 * m - 1) * za * Qa1 + Qa2
            P = ( (2 * m - 1) * x * P1 - (m - 1) * P2) / m
            term = ( (2 * m + 1) * P * Qr) / ( (m + 1) * za * Qa - Qa1)
            sum = sum + term
            m = m + 1
            Qr2 = Qr1
            Qr1 = Qr
            Qa2 = Qa1
            Qa1 = Qa
            P2 = P1
            P1 = P
            oldratio = newratio
            newratio = abs(term)/abs(sum)
            condition = (oldratio > threshold) + (newratio > threshold)
        H = (rho * exp(- 1j * mu) * sum) / (1j * mu)

        H[isnan(H)] = 0
        
        testf = fftfreq(nfft)
        H[testf>0] = conj(H[testf>0])
        
        return TransferFunction(H, 
                                samplerate = self.samplerate,  
                                coordinates = (az, el))
    
    def _cos_incidence(self, az, el, ear = 'left'):
        '''
        Returns the cosine of the incidence of an incoming ray at given az/el with
        the ear ('left'/'right', kwdarg). Used internally.
        '''
        if ear == 'left':
            azear = 90. - self.azear
        else:
            azear = -90. + self.azear
        center_ear = (azear * degAz + self.elear * degEl) * FRONT
        center_source = (az * degAz + el * degEl) * FRONT
        return float((center_source*center_ear)/(center_source.norm()*center_ear.norm()))
        
    def _incidence(self, az, el, ear = 'left'):
        '''
        Returns the incidence of an incoming ray at given az/el with
        the ear ('left'/'right', kwdarg). Used internally.
        '''
        return np.arccos(self._cos_incidence(az, el, ear = 'left'))
    
    @property
    def normalizedfrequency(self):
        '''
        The normalized frequency defined as 2 * pi * f * a / c
        '''
        f = linspace(0, 1, self.nfft) * self.samplerate
        return 2 * np.pi * f * self.a / c
    
    @property
    def normalizedtime(self):
        '''
        The normalized times defined as 2 * pi * t * c / a
        '''
        t = linspace(0, 1, self.nfft) / self.samplerate
        return 2 * np.pi * t * c / self.a


############################ DEPRECATED ###################################
# def _get_single_tf(self, az, el, distance = np.inf, 
#                        ear = 'left', 
#                        pre_delay = None,
#                        nfft = None, samplerate = None,
#                        method = 'old', order = 75, threshold = 1e-15):
#         ## This is where the computation is actually carried out

#         # KWDARG parsing
#         if nfft is None:
#             nfft = self.nfft

#         if samplerate is None:
#             samplerate = self.samplerate
        
#         ## main parameters of the model
#         # cosine of incidence angle
#         costheta = self._cos_incidence(az, el, ear = ear)
#         # normalized distance
#         rho = float(distance) / self.a
#         # frequencies
#         f = fftfreq(nfft)*self.samplerate
        
#         f[f<0] = -f[f<0]
        
#         # normalized frequencies
#         mu = (f * 2 * pi * self.a)/c

#         if distance == np.inf:
#             ## aproximation for infinite distances (at least d>>a)
#             tmp_Psi = np.zeros((nfft, order), dtype = complex)
        
#             for i in range(nfft):
#                 jn_prime = sph_jn(order-1, mu[i])[1]
#                 yn_prime = sph_yn(order-1, mu[i])[1]
#                 ratio = 1/(jn_prime+1j*yn_prime)
#                 tmp_Psi[i, :] = (2*arange(order)+1)*(-1j)**(arange(order)-1)*ratio
        
#             Pm_costheta = lpn(order-1, costheta)[0]
#             tmp_Psi *= np.tile(Pm_costheta.reshape((1, order)), (nfft, 1))
            
#             Psi_prime = np.sum(tmp_Psi, axis = 1)
            
#             H = (1 / mu**2) * Psi_prime 
#             H[0] = H[1]
#         else:
#             # otherwise, two different methods may be used
#             if method == 'old':
#                 # This here is the *dummy* implementation of the model using scipy's spherical functions
#                 # its a bit more compact, and because it's vectorized it's more complicated to understand
#                 # An important point is that the sph_jn (etc) functions in scipy return all the values of the function
#                 # up to a certain order. In the end it means this code
#                 # is vectorized in ``order`` but not in frequency
#                 tmp_Psi = np.zeros((nfft, order), dtype = complex)

#                 for i in range(nfft):
#                     jn = sph_jn(order-1, mu[i]*rho)[0]
#                     jn_prime = sph_jn(order-1, mu[i])[1]
#                     yn = sph_yn(order-1, mu[i]*rho)[0]
#                     yn_prime = sph_yn(order-1, mu[i])[1]
#                     ratio = (jn+1j*yn)/(jn_prime+1j*yn_prime)
#                     tmp_Psi[i, :] = (2*arange(order)+1)*ratio
        
#                 Pm_costheta = lpn(order-1, costheta)[0]
#                 tmp_Psi *= np.tile(Pm_costheta.reshape((1, order)), (nfft, 1))#/2
                
#                 Psi = np.sum(tmp_Psi, axis = 1)
                
#                 H = (-rho / mu) * np.exp(-1j * mu * rho ) * Psi #/ nfft
#                 H[0] = H[1]
                
#             elif method == 'new': 
#                 # This here implements the method for approximating H given in Duda Mertens 1998
#                 # - It basically does the same thing, except that the number of terms  in the 
#                 # series evaluation may depend on frequency
#                 # - Also, it uses a different stop condition (see threshold)
#                 # - Note that the implemetnation is straightforward, I have just renamed x by costheta
#                 zr = 1./(1j*mu*rho)
#                 za = 1./(1j*mu)

#                 Qr2 = zr
#                 Qr1 = zr*(1. - zr)
#                 Qa2 = za
#                 Qa1 = za*(1. - za)
                
#                 P2 = 1.
#                 P1 = costheta
                
#                 _sum = 0.
                
#                 term = zr/(za*(za - 1.))
#                 _sum = _sum + term
                
#                 term = (3*costheta*zr*(zr-1))/(za*(2*za**2-2*za+1))
                
#                 _sum = _sum + term
                
#                 newratio = abs(term)/abs(_sum)
#                 oldratio = ones_like(newratio)
#                 m = 2
                
#                 condition = ones(f.shape, dtype = bool)
                
#                 while condition.any():
#                     if m > MAX_ITER:
#                         print 'Maximum iterations reached'
#                         break
                    
#                     Qr = -(2*m - 1)*zr*Qr1 + Qr2
#                     Qa = -(2*m - 1)*za*Qa1 + Qa2
#                     P = ((2*m - 1)*costheta*P1 - (m-1)*P2)/m
#                     term = ((2*m + 1)*P*Qr)/((m+1)*za*Qa-Qa1)
                    
#                     _sum[condition] = _sum[condition] + term[condition]
                    
#                     m = m+1
#                     Qr2 = Qr1
#                     Qr1 = Qr
#                     Qa2 = Qa1
#                     Qa1 = Qa
#                     P2 = P1
#                     P1 = P
                    
#                     oldratio = newratio
#                     newratio = abs(term)/abs(_sum)
                    
#                     condition = (oldratio > threshold) + (newratio > threshold)
                    
#                 H = (float(rho)*exp(-1j*mu)*_sum )/ (1j*mu)

#         # remove nans? dunno
# #        H[np.isnan(H)] = 0
                
#         H[0] = 0
#         testf = fftfreq(nfft)
#         H[testf>0] = conj(H[testf>0])
        
        

#         # And in any event, if pre_delay is set then delay the TF
#         if pre_delay is None:
#             pre_delay = 0
#         else:
#             pre_delay = pre_delay
#         H *= exp(-1j*pre_delay*2*pi*f/self.samplerate)
#         # And return the TransferFunction object
#         return TransferFunction(H, 
#                                 samplerate = self.samplerate,  
#                                 coordinates = (az, el))
