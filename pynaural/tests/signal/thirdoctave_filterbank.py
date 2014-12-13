import scipy
import numpy as np
from matplotlib.pyplot import *
from pynaural.signal.misc import my_logspace, octaveband_filterbank

samplerate = 44100.
cfs = my_logspace(350, 2000, 16)


fraction = 1/3.

signal = np.random.randn(44100)

butter_order = 3

res0 = octaveband_filterbank(signal,cfs,samplerate,fraction = 1./3,butter_order=butter_order)

res = np.zeros((len(signal), len(cfs)))

for kcf, cf in enumerate(cfs):
    fU = 2**((fraction/2.))*cf
    fL = .5**((fraction/2.))*cf
    fU_norm = fU / (samplerate/2)
    fL_norm = fL / (samplerate/2)
    b, a = scipy.signal.butter(butter_order, (fL_norm, fU_norm), 'band')
    res[:,kcf] = scipy.signal.lfilter(b,a,signal)

print (res0==res).all()

print np.max(np.abs(res0-res))
figure()
for kcf in range(len(cfs)):
    subplot(121)
    plot(res[:,kcf]+kcf)
    subplot(122)
    plot(res0[:,kcf]+kcf)

show()