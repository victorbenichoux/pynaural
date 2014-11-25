from pynaural.raytracer import *

scene = GroundScene()

source_position = UP + FRONT + 1.5*LEFT
source = Source(source_position)
scene.add(source)

receiver_position = 1*UP
receiver = IRCAMSubjectReceiver(receiver_position, subject = 2031)
beam = scene.render(receiver)

print beam
sigma = 20000
model = NaturalGroundModel(sigma)
model.apply(beam)

#ir = scene.computeIRs(scene, nfft=4096)