from unittest import TestCase
from pynaural.signal.impulseresponse import *
import numpy as np
__author__ = 'victorbenichoux'


class TestImpulseResponse(TestCase):
    def setUp(self):
        pass

    def test_zeroIR(self):
        ir = zerosIR((100,2), samplerate = 44100)
        self.assertTrue((ir==0).all())

        ir = zerosIR((100.,2), samplerate = 44100)
        self.assertTrue((ir==0).all())

    def test_delayIR(self):
        ir = delayIR(0.1, (44100,1), samplerate = 44100.)
        res = ir.apply(delayIR(0, (44100,1), samplerate = 44100.))
        self.assertTrue(np.nonzero(np.asarray(ir))[0][0]==np.nonzero(np.asarray(res))[0][0])