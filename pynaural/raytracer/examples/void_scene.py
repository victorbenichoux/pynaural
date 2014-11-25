from pynaural.raytracer import *

scene = VoidScene()

source = Source(UP + FRONT + 1.5*LEFT)
source2 = Source(UP + FRONT + 1.5*RIGHT)
scene.add(source, source2)

receiver = Receiver(1*UP)

beam = scene.render(receiver)
