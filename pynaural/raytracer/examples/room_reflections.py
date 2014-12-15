from pynaural.raytracer import *
from matplotlib.pyplot import *

room_size = (4.,6.,2.)

def run(sourcepos, receiverpos, nreflections):
    scene = RoomScene(room_size[0], room_size[1], room_size[2], nreflections = nreflections, model = {'alpha' : 0.1})

    sourcepos = 1.5*FRONT + UP + 0.1*LEFT
    source = Source(sourcepos)
    scene.add(source)

    receiverpos = 1.5*UP
    receiver = IRCAMSubjectReceiver(receiverpos, subject = 2031)
    beam = scene.render(receiver)

    return beam.get_reachedsource_depth()

def n2d(i):
    return 2*i*(i+1) + 1
def n3d(i):
    sumvect = n2d(np.arange(i))
    return n2d(i) + 2 * np.sum(sumvect)

def test(Nreflections, res):
    out = True
    for k in range(Nreflections):
        cur = len(np.nonzero(res == k+1)[0]) == n3d(k)-n3d(k-1)
        out *= cur
        print "--"
        print cur
        print n3d(k)-n3d(k-1)
        print len(np.nonzero(res == k+1)[0])
    return out

Niter = 10

np.tile(np.array(room_size).reshape((3,1)), (1, Niter))

source_positions = np.zeros((3,Niter))
source_positions[0,:] = (np.random.rand(Niter)-0.5) * room_size[0]
source_positions[1,:] = (np.random.rand(Niter)-0.5) * room_size[1]
source_positions[2,:] = (np.random.rand(Niter)) * room_size[2]

receiver_positions = np.zeros((3,Niter))
receiver_positions[0,:] = (np.random.rand(Niter)-0.5) * room_size[0]
receiver_positions[1,:] = (np.random.rand(Niter)-0.5) * room_size[1]
receiver_positions[2,:] = (np.random.rand(Niter)) * room_size[2]

max_nreflections = 10
n_reflections = np.array(np.rint(np.random.rand(Niter)*max_nreflections)+1, int)

res = np.zeros(Niter)
for kiter in xrange(Niter):
    cur = run(source_positions[:,kiter], receiver_positions[:,kiter], nreflections = n_reflections[kiter])
    res[kiter] = int(test(n_reflections[kiter], cur))

print 'Success rate', res.mean()*100.