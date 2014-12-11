import numpy as np

from matplotlib.pyplot import *

from pynaural.signal.coherence import octaveband_coherence, gammatone_coherence
from pynaural.signal.sounds import Sound


noise0 = Sound(np.random.randn(10000), samplerate = 44100.)
noise1 = Sound(np.random.randn(10000), samplerate = 44100.)


cfs = np.linspace(100, 10000, 10)
bc01 = gammatone_coherence(Sound((noise0, noise1)), noise0.samplerate, cfs)
bc00 = gammatone_coherence(Sound((noise0, noise0)), noise0.samplerate, cfs)


figure()
subplot(211)
plot(cfs, bc00)
plot(cfs, bc01)
xlabel('CF')
ylabel('Gammatone coherence')
ylim(0,1.2)

cfs = np.linspace(100, 10000, 10)
bc01 = octaveband_coherence(Sound((noise0, noise1)), noise0.samplerate, cfs)
bc00 = octaveband_coherence(Sound((noise0, noise0)), noise0.samplerate, cfs)

subplot(212)
plot(cfs, bc00)
plot(cfs, bc01)
ylabel('1/3 oct. coherence')
xlabel('CF')
ylim(0,1.2)
show()