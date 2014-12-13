import scipy as sp
from pynaural.signal.misc import *
from matplotlib.pyplot import *
'''
the peak (amplitude and position) of the envelope of the XCORR is the same as the peak of the XCORR of the envelopes.
'''

samplerate = 44100.
cfs = my_logspace(350, 2000, 4)

fraction = 1/3.

left_signal = np.random.randn(44100)
right_signal = left_signal.copy() + np.random.randn(44100) * .5

butter_order = 3

filtered_left = octaveband_filterbank(left_signal, cfs, samplerate, fraction = 1./3, butter_order=butter_order)
filtered_right = octaveband_filterbank(right_signal, cfs, samplerate, fraction = 1./3, butter_order=butter_order)

tcut = 1e-3

res_env2 = np.zeros_like(cfs)
res_env = np.zeros_like(cfs)

for kcf in range(len(cfs)):
    left = filtered_left[:, kcf]
    right = filtered_right[:, kcf]
    times = (np.arange(len(left)+len(right)-1)+1-len(left))/samplerate

    left_env = np.abs(sp.signal.hilbert(left))
    right_env = np.abs(sp.signal.hilbert(right))
    left_p = np.sqrt(np.mean(left_env**2))
    right_p = np.sqrt(np.mean(right_env**2))
    xcorr_env = fftxcorr(left_env, right_env)/(left_p*right_p*len(right_env))

    res_env[kcf] = np.argmax(xcorr_env[np.abs(times) < tcut])

    env_xcorr = np.abs(sp.signal.hilbert(xcorr))
    res_env2[kcf] = np.argmax(xcorr_env[np.abs(times) < tcut])

figure()
plot(res_env2, res_env)
show()