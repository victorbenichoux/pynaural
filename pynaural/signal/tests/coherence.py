import numpy as np
from pynaural.signal.misc import gammatone_coherence
from pynaural.io.sounds.sounds import Sound
from matplotlib.pyplot import *

noise0 = Sound(np.random.randn(10000), samplerate = 44100.)
noise1 = Sound(np.random.randn(10000), samplerate = 44100.)


cfs = np.linspace(100, 10000, 10)
bc01 = gammatone_coherence(Sound((noise0, noise1)), noise0.samplerate, cfs)
bc00 = gammatone_coherence(Sound((noise0, noise0)), noise0.samplerate, cfs)


figure()
subplot(211)
plot(bc00)
plot(bc01)
show()