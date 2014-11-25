from pynaural.raytracer import *

# First, set up a spatializer scene with only a ground
scene = GroundScene()

# place some sound sources
source = Source(UP + FRONT + 1.5*LEFT)
source2 = Source(UP + FRONT + 1.5*RIGHT)
scene.add(source, source2)

# add a simple receiver
receiver = Receiver(1*UP)

# render the ray paths
paths = scene.render(receiver)

# apply Mikki's reflection model
model = NaturalGroundModel()
tf = model.apply(paths)
