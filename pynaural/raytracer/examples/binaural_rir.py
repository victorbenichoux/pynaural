from pynaural.raytracer import *
from matplotlib.pyplot import *

scene = RoomScene(4., 6., 2., nreflections = 0, model = {'alpha' : 0.1})

sourcepos = 1.5*FRONT + UP + 0.1*LEFT
source = Source(sourcepos)
scene.add(source)

receiverpos = 1.5*UP
receiver = IRCAMSubjectReceiver(receiverpos, subject = 2031)
beam = scene.render(receiver)

model = DelayAttenuationModel(scene = scene)

irs = model.apply(beam)

brir = receiver.collapse(irs)

brir.listen(sleep = True)

tdirect = (receiverpos-sourcepos).norm()/342.
times = brir.times

figure()
plot(times, brir)
axvline(tdirect)

draw()

show()