import scipy as sp
from matplotlib.pyplot import *

from pynaural.signal.misc import *


samplerate = 44100.
cfs = my_logspace(350, 2000, 4)

fraction = 1/3.

left_signal = np.random.randn(44100)
right_signal = left_signal.copy()

butter_order = 3

filtered_left = octaveband_filterbank(left_signal, cfs, samplerate, fraction = 1./3, butter_order=butter_order)
filtered_right = octaveband_filterbank(right_signal, cfs, samplerate, fraction = 1./3, butter_order=butter_order)

tcut = 1e-3

res = np.zeros_like(cfs)
res_env = np.zeros_like(cfs)

for kcf in range(len(cfs)):
    left = filtered_left[:, kcf]
    right = filtered_right[:, kcf]
    times = (np.arange(len(left)+len(right)-1)+1-len(left))/samplerate
    xcorr = fftxcorr(left, right)/(rms(left)*rms(right)*len(right))
    res[kcf] = np.max(xcorr[np.abs(times) < tcut])

    left_env = np.abs(sp.signal.hilbert(left))
    right_env = np.abs(sp.signal.hilbert(right))
    left_p = np.sqrt(np.mean(left_env**2))
    right_p = np.sqrt(np.mean(right_env**2))
    xcorr_env = fftxcorr(left_env, right_env)/(left_p*right_p*len(right_env))

    res_env[kcf] = np.max(xcorr_env[np.abs(times) < tcut])

    x=plot(times, np.abs(sp.signal.hilbert(xcorr)))
    plot(times, xcorr_env, '--', color = x[0]._color)

    show()

figure()
plot(res, res_env)
xlim(-1,1)
ylim(-1,1)
show()