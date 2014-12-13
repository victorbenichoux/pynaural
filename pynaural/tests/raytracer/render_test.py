import numpy as np

from pynaural.raytracer.scenes import GeometricScene, RoomScene
from pynaural.raytracer.geometry.surfaces import GROUND
from pynaural.raytracer.geometry.base import UP, FRONT
from pynaural.raytracer.geometry.rays import *
from pynaural.raytracer.receivers import Receiver
from pynaural.raytracer.sources import Source
from pynaural.utils.spatprefs import set_pref

def basic_render_test():
    scene = RoomScene(4., 6., 2., stopcondition = 30)
    source = Source(1.5*FRONT + UP)
    scene.add(source)
    
    b = RandomSphericalBeam(UP, 1e4)
    print b.target_source
    b.set_target(source)
    print b.target_source
    
    scene.render(b)
    print b.get_reachedsource()
    assert b.get_reachedsource().any()

    
if __name__ == '__main__':
    basic_render_test()


