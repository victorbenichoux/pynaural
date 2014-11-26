from pynaural.raytracer import *
from matplotlib.pyplot import *
from pynaural.signal.impulseresponse import TransferFunction
from pynaural.signal.binauralcues import *
# First, set up a spatializer scene with only a ground
scene = GroundScene()

# place some sound sources
source = Source(UP + 5*FRONT + 1.5*LEFT)
scene.add(source)

# add a simple receiver
receiver = SphericalHeadReceiver(2*UP, 0.1,
                                    samplerate = 192000.,nfft = 1024, pre_delay = 512)

# render the ray paths
paths = scene.render(receiver)

# apply Mikki's reflection model
model = NaturalGroundModel(samplerate = 192000.)
irs = model.apply(paths)

bir = receiver.collapse(irs)
bir = bir.ramp(32)
#bir.listen()

times_bir = np.linspace(0, float(bir.duration), bir.nsamples)
times_irs = np.linspace(0, float(irs.duration), irs.nsamples)

figure()
subplot(211)
plot(times_irs, irs)
subplot(212)
plot(times_bir, bir)

btf = TransferFunction(np.fft.fft(bir[:,:], axis = 0), samplerate = bir.samplerate, binaural = True)
cfs = np.linspace(600, 1800, 64)
itdp, itdg, idi = itdp_itdg_idi(btf.ipd(indices = btf.freqs >=0).flatten(), cfs, btf.samplerate,
                                    cut_lf = True, fcut_lf = 500, Q = 4)

figure()
subplot(211)
plot(cfs, itdp)
plot(cfs, itdg)
subplot(212)
plot(cfs, idi)
show()