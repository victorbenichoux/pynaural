from pynaural.raytracer import *
from unittest import TestCase

__author__ = 'victorbenichoux'


class TestCoherence(TestCase):
    def setUp(self):
        pass

    def test_gammatone_coherence(self):
        nreflections = 3
        scene = RoomScene(2., 4., 4, nreflections = nreflections)

        source = Source(1.5*FRONT +0.23*LEFT+ UP, ref = 0.01)

        scene.add(source)

        b = scene.get_beam(UP)

        scene.render(b)

        self.assertTrue((b.get_reachedsource_depth() <= nreflections + 1).all())

