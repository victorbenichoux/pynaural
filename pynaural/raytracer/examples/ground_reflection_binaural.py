from pynaural.raytracer import *
from matplotlib.pyplot import *
from pynaural.signal.impulseresponse import TransferFunction
from pynaural.signal.binauralcues import *
# First, set up a spatializer scene with only a ground
scene = GroundScene()

# place some sound sources
headposition = 0.08*UP
azimuth= 60*degAz
d = 1.5
source = Source(headposition + (d*FRONT)*azimuth)
scene.add(source)

# add a simple receiver
receiver = SphericalHeadReceiver(2*UP, 0.1,
                                    samplerate = 192000.,nfft = 4096, pre_delay = 512)

# render the ray paths
paths_left = scene.render(Receiver(receiver.get_ear_position('left')))
paths_right = scene.render(Receiver(receiver.get_ear_position('right')))

# apply Mikki's reflection model
model = NaturalGroundModel(samplerate = 192000., nfft = 4096)
irs_left = model.apply(paths_left)
irs_right = model.apply(paths_right)

bir_left = receiver.collapse(irs_left, monaural = 'left')
bir_right = receiver.collapse(irs_left, monaural = 'right')

bir = ImpulseResponse(np.hstack((bir_left, bir_right)), samplerate = bir_left.samplerate, binaural = True)

times_bir = np.linspace(0, float(bir.duration), bir.nsamples)
times_irs_left = np.linspace(0, float(irs_left.duration), irs_left.nsamples)
times_irs_right = np.linspace(0, float(irs_left.duration), irs_right.nsamples)

figure()
subplot(211)
plot(times_irs_left, irs_left)
plot(times_irs_right, irs_right)
subplot(212)
plot(times_bir, bir)

btf = TransferFunction(np.fft.fft(bir[:,:], axis = 0), samplerate = bir.samplerate, binaural = True)
cfs = np.linspace(600, 1800, 64)
itdp, itdg, idi = itdp_itdg_idi(btf.ipd(indices = btf.freqs >=0).flatten(), cfs, btf.samplerate,
                                    cut_lf = True, fcut_lf = 500, Q = 2)

figure()
subplot(211)
plot(cfs, itdp)
plot(cfs, itdg)
subplot(212)
plot(cfs, idi)
show()