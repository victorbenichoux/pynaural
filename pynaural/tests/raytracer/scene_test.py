import numpy as np
from pynaural.raytracer import *

def GeometricScene_test():
    """
    Scenes initialization tests
    """
    scene = GeometricScene()
    scene = RoomScene(2., 4., 6)
    scene = GroundScene()
    pass

def RoomScene_volume_test():
    """
    Volume computation test
    """
    scene = RoomScene(2., 4., 4)    
    scene2 = GeometricScene(scene.leftwall, scene.rightwall, scene.backwall, scene.frontwall, scene.floor, scene.ceiling)

    vol2 = scene2.volume(abs_max=4)
    dv =  np.abs(vol2 - scene.volume())
    print vol2, dv
    assert dv < 1.

    
if __name__ == '__main__':
    RoomScene_beam_test()
    GeometricScene_test()
    RoomScene_volume_test()
        
