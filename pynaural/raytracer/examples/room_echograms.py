from pynaural.raytracer import *
from matplotlib.pyplot import *
"""
A simple example in which basic echograms are computed (i.e. propagation time to source and number of reflections per ray) for rooms of different sizes, with same proportions.
"""


Nrays = 1e7
alpha = 0.9

depths = []
times = []
factors = np.linspace(1, 2, 10)
for kf, fact in enumerate(factors):
    scene = RoomScene(fact*4., fact*6., fact*2., stopcondition = 10)
    source = Source(1.5*FRONT + UP)
    scene.add(source)
    
    b = RandomSphericalBeam(UP, 1e6)
    scene.render(b)

    reached_id = b.get_reachedsource_index()
    reached = b[reached_id]
    cur_depth = reached.get_reachedsource_depth()
    cur_time = reached.get_totaldelays()
    depths.append(cur_depth)
    times.append(cur_time*1e3)


    curcol = cm.jet(float(kf)/len(factors))
    subplot(211)
    plot(cur_time*1e3, cur_depth, 'o', color = curcol)
    subplot(212)
    for k in range(len(cur_time)):
        plot([cur_time[k]*1e3, cur_time[k]*1e3], 
             [0, alpha**(cur_depth[k])], color = curcol)

xlabel('Time (ms)')
show()
