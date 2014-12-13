from unittest import TestCase
from pynaural.binaural import *
from pynaural.signal.impulseresponse import delayIR


class TestImpulseResponse(TestCase):
    def test_zeroIR(self):
        itd = 0.001
        hrir = delayIR([2*itd, 3*itd], (44100, 2), samplerate = 44100.)