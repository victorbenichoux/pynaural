from unittest import TestCase
from matplotlib.pyplot import *
from pynaural.binaural.coherence import octaveband_coherence, gammatone_coherence
from pynaural.signal.sounds import Sound
import numpy as np

__author__ = 'victorbenichoux'


class TestCoherence(TestCase):
    def setUp(self):
        pass

    def test_gammatone_coherence(self):
        noise0 = Sound(np.random.randn(10000), samplerate = 44100.)
        noise1 = Sound(np.random.randn(10000), samplerate = 44100.)


        cfs = np.linspace(100, 10000, 10)
        bc01 = gammatone_coherence(Sound((noise0, noise1)), noise0.samplerate, cfs)
        bc00 = gammatone_coherence(Sound((noise0, noise0)), noise0.samplerate, cfs)

        if False:
            figure()
            plot(cfs, bc00)
            plot(cfs, bc01)
            xlabel('CF')
            ylabel('Gammatone coherence')
            ylim(0,1.2)

        self.assertTrue((1-bc00<1e10).all())
        self.assertTrue((bc01!=1).all())

    def test_octaveband_coherence(self):
        noise0 = Sound(np.random.randn(10000), samplerate = 44100.)
        noise1 = Sound(np.random.randn(10000), samplerate = 44100.)

        cfs = np.linspace(100, 10000, 10)
        bc01 = octaveband_coherence(Sound((noise0, noise1)), noise0.samplerate, cfs)
        bc00 = octaveband_coherence(Sound((noise0, noise0)), noise0.samplerate, cfs)

        if False:
            figure()
            plot(cfs, bc00)
            plot(cfs, bc01)
            ylabel('1/3 oct. coherence')
            xlabel('CF')
            ylim(0,1.2)
            show()
        self.assertTrue((1-bc00<1e10).all())
        self.assertTrue((bc01!=1).all())