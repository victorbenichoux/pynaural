from pynaural.raytracer import *
from matplotlib.pyplot import *

scene = RoomScene(4., 6., 2., nreflections = 100, model = {'alpha':0.1})
sourcepos = 1.5*FRONT + UP + 0.1*LEFT
source = Source(sourcepos)
scene.add(source)

receiverpos = 1.2*UP
receiver = Receiver(receiverpos)
beam = scene.render(receiver)

model = DelayAttenuationModel(scene = scene)

irs = model.apply(beam)

tdirect = (receiverpos-sourcepos).norm()/342.

figure()
plot(irs.times, irs)
axvline(tdirect)

draw()