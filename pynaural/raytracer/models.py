'''
This is a new class designed to replace the old Model paradigm that is getting messy and outdated
'''
from pynaural.raytracer import cartesian2spherical
from pynaural.raytracer.acoustics import spherical_ref_factor
import pynaural.signal.impulseresponse
import numpy as np

from matplotlib.pyplot import *

__all__ = ['NaturalGroundModel', 'DelayAttenuationModel']

class NaturalGroundModel(object):
    def __init__(self, sigma = 20000, samplerate = 44100., nfft = 1024):
        self.sigma = sigma
        self.nfft = nfft
        self.samplerate = samplerate

    def apply(self, in_beam):
        beam = in_beam[in_beam.get_reachedsource()]
        # first check that the beam is consistent with a situation with only a ground
        if not (beam.depth == 2 and beam.nrays % 2 == 0):
            raise ValueError("NaturalGroundModel can only be used with scenes with only a ground")

        nsources = beam.nrays / 2
        reflected_tfs = np.zeros((self.nfft, nsources), dtype = complex)

        freqs = np.fft.fftfreq(self.nfft, d = 1/self.samplerate)
        f = freqs
        f[f<0] = -f[f<0]

        distances = beam.get_totaldists()

        for ksource in xrange(nsources):
            reflected_tfs[:, ksource] = -spherical_ref_factor(distances[2*ksource], beam.incidences[ksource*2], f, sigma = self.sigma)
            #reflected_tfs[freqs<=0, :] = np.conj(reflected_tfs[freqs>0, :])

        tf_data = np.ones((self.nfft, nsources*2), dtype = complex)
        tf_data[:, 1::2] = reflected_tfs

        beam_coordinates = cartesian2spherical(beam.directions, unit = 'deg')

        dtype_coords = [('azim','f8'), ('elev','f8'), ('dist', 'f8')]
        coordinates = np.zeros(beam.nrays, dtype = dtype_coords)
        coordinates['dist'] = beam_coordinates[0]
        coordinates['azim'] = beam_coordinates[1]
        coordinates['elev'] = beam_coordinates[2]

        ir_data = np.fft.ifft(tf_data, axis = 0).real
        delays = np.rint(beam.get_totaldelays()*self.samplerate)
        print beam.get_totaldelays()

        ir_data_final = np.zeros((self.nfft + delays.max(), nsources*2))
        for k in range(nsources):
            ir_data_final[delays[2*k]:delays[2*k]+self.nfft,2*k] = ir_data[:,2*k]
            ir_data_final[:self.nfft,2*k+1] = ir_data[:,2*k+1]
        # print tf_data
        # figure()
        # subplot(211)
        # semilogx(20*np.log10(np.abs(tf_data)))
        # subplot(212)
        # plot(ir_data_final)
        # show()

        return pynaural.signal.impulseresponse.ImpulseResponse(ir_data,
                                samplerate = self.samplerate,
                                coordinates = coordinates)

################## Delay + Global Attenuation Model ############################

def generate_delayattenuation_irs(delays, gains, samplerate, nfft):
    delays_samples = np.array(np.rint(delays*samplerate), dtype = int)
    max_delay_samples = delays_samples.max()
    n = nfft + max_delay_samples

    data_ir = np.zeros((n, len(delays)))
    for kd in range(data_ir.shape[1]):
        data_ir[delays_samples[kd], kd] = gains[kd]

    return data_ir

class DelayAttenuationModel(object):
    '''
    Accumulates delays and attenuations for each rays, then if relevant also uses HRTFs
    '''
    def __init__(self, samplerate = 44100., nfft = 1024, scene = None):
        self.nfft = nfft
        self.samplerate = samplerate

        if not scene is None:
            self.prepare_surfaces(scene)

    def prepare_surfaces(self, scene):
        gains = []
        for surface in scene.surfaces:
            if surface.model:
                gains.append(surface.model['alpha'])

        self.gains = np.array(gains)

    def apply(self, in_beam, scene = None, collapse = True):
        beam = in_beam[in_beam.get_reachedsource()]
        if not scene is None:
            self.prepare_surfaces(scene)

        delays = beam.get_totaldelays()

        if len(np.unique(self.gains)) == 1:
            gains = self.gains[0]**(beam.get_reachedsource_depth()-1)
        else:
            raise NotImplementedError

        data_ir = generate_delayattenuation_irs(delays, gains, self.samplerate, self.nfft)

        beam_coordinates = cartesian2spherical(beam.directions, unit = 'deg')

        dtype_coords = [('azim','f8'), ('elev','f8'), ('dist', 'f8')]
        coordinates = np.zeros(beam.nrays, dtype = dtype_coords)
        coordinates['dist'] = beam_coordinates[0]
        coordinates['azim'] = beam_coordinates[1]
        coordinates['elev'] = beam_coordinates[2]

        return pynaural.signal.impulseresponse.ImpulseResponse(data_ir,
                                samplerate = self.samplerate,
                                coordinates = coordinates, is_delay = True)

