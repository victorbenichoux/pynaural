import numpy as np

from pynaural.raytracer import *

def room_render_test():
    nreflections = 3
    scene = RoomScene(2., 4., 4, nreflections = nreflections)

    source = Source(1.5*FRONT +0.23*LEFT+ UP, ref = 0.01)
    
    scene.add(source)
    
    b = scene.get_beam(UP)

    scene.render(b)

    assert (b.get_reachedsource_depth() <= nreflections + 1).all()

if __name__ == '__main__':
    room_render_test()
    
