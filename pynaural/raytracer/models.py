'''
This is a new class designed to replace the old Model paradigm that is getting messy and outdated
'''
from pynaural.raytracer import cartesian2spherical
from pynaural.raytracer.acoustics import spherical_ref_factor
from pynaural.signal.impulseresponse import TransferFunction
import numpy as np

__all__ = ['NaturalGroundModel']

class NaturalGroundModel(object):
    def __init__(self, sigma = 20000, samplerate = 44100., nfft = 1024):
        self.sigma = sigma
        self.nfft = nfft
        self.samplerate = samplerate

    def apply(self, beam):
        # first check that the beam is consistent with a situation with only a ground
        if not (beam.depth == 3 and beam.nrays % 2 == 0):
            raise ValueError("NaturalGroundModel can only be used with scenes with only a ground")

        nsources = beam.nrays / 2
        reflected_tfs = np.zeros((self.nfft, nsources), dtype = complex)

        freqs = np.fft.fftfreq(self.nfft, d = 1/self.samplerate)
        f = freqs
        f[f<0] = -f[f<0]

        distances = beam.get_totaldists()

        for ksource in xrange(nsources):
            reflected_tfs[:, ksource] = spherical_ref_factor(distances[2*ksource], beam.incidences[ksource*2], f, sigma = self.sigma)
            reflected_tfs[freqs>0, :] = np.conj(reflected_tfs[freqs>0, :])

        tf_data = np.ones((self.nfft, nsources*2), dtype = complex)
        tf_data[:, 1::2] = reflected_tfs

        beam_coordinates = cartesian2spherical(beam.directions, unit = 'deg')

        dtype_coords = [('azim','f8'), ('elev','f8'), ('dist', 'f8')]
        coordinates = np.zeros(beam.nrays, dtype = dtype_coords)
        coordinates['dist'] = beam_coordinates[0]
        coordinates['azim'] = beam_coordinates[1]
        coordinates['elev'] = beam_coordinates[2]

        return TransferFunction(tf_data, coordinates = coordinates)

################## Delay + Global Attenuation Model ############################

class DelayAttenuationModel(object):
    '''
    Accumulates delays and attenuations for each rays, then if relevant also uses HRTFs
    '''
