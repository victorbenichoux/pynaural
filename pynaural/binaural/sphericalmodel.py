from pynaural.raytracer.geometry.base import degAz, degEl, FRONT
from numpy.fft import fft, ifft, fftfreq
import pynaural
import numpy as np

__all__ = ['SphericalHead']

MAX_ITER = 1e7
c = 342.



class SphericalHead(object):
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
                 nfft = 1024, samplerate = 44100.):
        
        if isinstance(earpos, tuple):
            earelevation, earazimuth = earpos
        else:
            earelevation = 0.
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

        tf = pynaural.signal.impulseresponse.TransferFunction(both,
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

        tf = pynaural.signal.impulseresponse.TransferFunction(data,
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

        ir = ifft(np.asarray(h), axis = 0).real
        ir = np.roll(ir, pre_delay, axis = 0)
        
        irp = np.vstack((np.zeros((1,2)), ir[:-1,:]))
        irm = np.vstack((ir[1:,:], np.zeros((1,2))))
        
        ir = 0.5*(2*ir+irp+irm)
        
        return pynaural.signal.impulseresponse.ImpulseResponse(ir, **h.get_kwdargs())
        
    def _get_single_tf(self, az, el, distance = 2.,
                       ear = 'left', 
                       pre_delay = None,
                       nfft = None, samplerate = None, threshold = 1e-15):
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
        
        f[f < 0] = -f[f < 0]
        
        a = float(self.a)
        r = float(distance)

        mu = (2 * np.pi * f * a) / c
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
        
        condition = np.ones(f.shape, dtype = bool)
        
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
        H = (rho * np.exp(- 1j * mu) * sum) / (1j * mu)

        H[np.isnan(H)] = 0
        
        testf = fftfreq(nfft)
        H[testf>0] = np.conj(H[testf>0])
        
        return pynaural.signal.impulseresponse.TransferFunction(H,
                                samplerate=self.samplerate,
                                coordinates=(az, el))
    
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
        f = np.linspace(0, 1, self.nfft) * self.samplerate
        return 2 * np.pi * f * self.a / c
    
    @property
    def normalizedtime(self):
        '''
        The normalized times defined as 2 * pi * t * c / a
        '''
        t = np.linspace(0, 1, self.nfft) / self.samplerate
        return 2 * np.pi * t * c / self.a