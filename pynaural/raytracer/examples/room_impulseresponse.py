from pynaural.raytracer import *
from matplotlib.pyplot import *

scene = RoomScene(4., 6., 2., nreflections = 1)
source = Source(1.5*FRONT + UP + 0.1*LEFT)
scene.add(source)

receiver = Receiver(1.2*UP)
beam = scene.render(receiver)

model = DelayAttenuationModel(scene = scene)

irs = model.apply(beam)

rir = np.sum(np.array(irs), axis = 1)
#rir += +np.random.randn(len(rir))
subplot(211)
plot(np.log10(rir), '.')
subplot(212)
depths = beam.get_reachedsource_depth()
print len(depths)
print rir.shape
plot(depths)
show()