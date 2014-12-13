import numpy as np

from matplotlib.pyplot import *
import scipy as sp

from pynaural.signal.misc import octaveband_filterbank
from pynaural.signal.misc import fftxcorr
from pynaural.signal.sounds import Sound


samplerate = 44100.

noise = Sound(np.random.randn(44100), samplerate = samplerate).ramp(when = 'both', duration=0.02)

cfs = np.linspace(100, 1000, 1)

signal = octaveband_filterbank(noise, cfs, noise.samplerate)[:,0]
envelope = np.abs(sp.signal.hilbert(signal, axis = 0))

# fir filters with linear/affine phase
itdg = .05

phase = lambda x, itdg, idi: itdg * f + idi
f = np.fft.fftfreq(4096)*samplerate

# TF and IR with and without IDI
tf = np.exp(-2*np.pi*1j*phase(f, itdg, 0))
ir = np.fft.ifft(tf).real

tf1 = np.exp(-2*np.pi*1j*phase(f, itdg, 0.5))
ir1 = np.fft.ifft(tf1).real

ir_times = np.arange(4096)/samplerate

# filtering the signal

filtered_noise_0 = sp.signal.lfilter(ir, 1., signal)
filtered_noise_1 = sp.signal.lfilter(ir1, 1., signal)

signal_times = np.arange(len(signal))/44100.

figure()

subplot(211)
plot(ir_times, ir, 'b')
plot(ir_times, np.abs(sp.signal.hilbert(ir)), 'b--')

subplot(212)
plot(ir_times, ir, 'g')
plot(ir_times, np.abs(sp.signal.hilbert(ir)), 'g--')

figure()
subplot(311)
plot(signal_times, signal, 'k')
plot(signal_times, envelope, 'k')
subplot(312)
plot(signal_times, filtered_noise_0)
plot(signal_times, np.abs(sp.signal.hilbert(filtered_noise_0)))
subplot(313)
plot(signal_times, filtered_noise_1)
plot(signal_times, np.abs(sp.signal.hilbert(filtered_noise_1)))

figure()


subplot(313)
xcorr0 = fftxcorr(filtered_noise_0, noise)
xcorr1 = fftxcorr(filtered_noise_1, noise)

xcorr_times = (np.arange(len(xcorr0)) - len(filtered_noise_0) - 1)/samplerate

plot(xcorr_times, xcorr0, 'b')
plot(xcorr_times, xcorr1, 'g')

plot(xcorr_times, np.abs(sp.signal.hilbert(xcorr0)), 'b--')
plot(xcorr_times, np.abs(sp.signal.hilbert(xcorr1)), 'g--')

delay0 = xcorr_times[np.argmax(xcorr0)]
print delay0
np.argmax(xcorr1)
show()