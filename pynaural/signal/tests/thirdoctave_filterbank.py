import scipy
import numpy as np
from matplotlib.pyplot import *
from pynaural.signal.misc import my_logspace

samplerate = 44100.
cfs = my_logspace(100, 10000, 30)
fraction = 1/6.

signal = np.random.randn(44100)

butter_order = 3

res = np.zeros((len(signal), len(cfs)))

for kcf, cf in enumerate(cfs):
    fU = 2**((fraction/2.))*cf
    fL = .5**((fraction/2.))*cf
    print fL,cf, fU
    fU_norm = fU / (samplerate/2)
    fL_norm = fL / (samplerate/2)
    b, a = scipy.signal.butter(butter_order, (fL_norm, fU_norm), 'band')
    print b.shape
    res[:,kcf] = scipy.signal.filtfilt(b,a,signal)

figure()
for kcf in range(len(cfs)):
    plot(res[:,kcf]+kcf)