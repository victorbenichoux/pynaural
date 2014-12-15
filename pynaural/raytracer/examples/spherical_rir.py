from pynaural.raytracer import *
from pynaural.binaural.sphericalmodel import *
from matplotlib.pyplot import *

scene = RoomScene(4., 6., 2., nreflections = 3, model = {'alpha' : 0.1})

sourcepos = 1.5*FRONT + UP + 0.1*LEFT
source = Source(sourcepos)
scene.add(source)

receiverpos = 1.2*UP
receiver = Receiver(receiverpos)#, 2*0.0103)
beam = scene.render(receiver)

print beam.get_reachedsource_depth()
beam[0].plot(recursive = True)
draw()
show()



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