import numpy as np

__all__ = ['atlevel', 'powerlawnoise', 'get_rms_dB']

def atlevel(sound, db):
    sound_rms_value = np.sqrt(np.mean((np.asarray(sound)-np.mean(np.asarray(sound)))**2))
    sound_rms_dB = 20.0*np.log10(sound_rms_value/2e-5)
    gain = 10**(((db-sound_rms_dB)/20.))
    leveled = gain*sound.copy()
    return leveled

def powerlawnoise(nsamples, alpha, samplerate, nchannels=1,normalise=False):
    n = int(2**np.ceil(np.log(nsamples)/np.log(2)))
    n2=np.floor(n/2)

    f=np.array(np.fft.fftfreq(n,d=1.0/samplerate), dtype=complex)
    f.shape=(len(f),1)
    f=np.tile(f,(1,nchannels))

    if n%2==1:
        z=(np.random.randn(n2,nchannels)+1j*np.random.randn(n2,nchannels))
        a2=1.0/( f[1:(n2+1),:]**(alpha/2.0))
    else:
        z=(np.random.randn(n2-1,nchannels)+1j*np.random.randn(n2-1,nchannels))
        a2=1.0/(f[1:n2,:]**(alpha/2.0))

    a2*=z

    if n%2==1:
        d=np.vstack((np.ones((1,nchannels)),a2,
                     np.flipud(np.conj(a2))))
    else:
        d=np.vstack((np.ones((1,nchannels)),a2,
                  1.0/( np.abs(f[n2])**(alpha/2.0) )*
                     np.random.randn(1,nchannels),
                     np.flipud(np.conj(a2))))


    x = np.real(np.fft.ifft(d.flatten()))
    x.shape=(n, nchannels)

    if normalise:
        for i in range(nchannels):
            x[:,i] = ((x[:,i] - np.amin(x[:,i]))/(np.amax(x[:,i]) - np.amin(x[:,i])) - 0.5) * 2;

    return x[:nsamples,:]


def get_rms_dB(signal):
    rms_value = np.sqrt(np.mean((np.asarray(signal)-np.mean(np.asarray(signal)))**2))
    rms_dB = 20.0*np.log10(rms_value/2e-5)
    return rms_dB

