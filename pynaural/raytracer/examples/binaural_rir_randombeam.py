from pynaural.raytracer import *
from matplotlib.pyplot import *

scene = RoomScene(4., 6., 2., nreflections = 10)
source = Source(1.5*FRONT + UP + 0.1*LEFT)
scene.add(source)

receiver = IRCAMSubjectReceiver(1.5*UP, subject = 2031)
beam = scene.render(RandomSphericalBeam(receiver.position, 1e5))

model = DelayAttenuationModel(scene = scene)

irs = model.apply(beam)

#brir = receiver.collapse(irs)

if False:
    figure()
    subplot(211)
    plot(irs)
    subplot(212)
    plot(brir)
    show()