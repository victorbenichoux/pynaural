import numpy as np

__all__ = ['atlevel', 'get_rms_dB']

def atlevel(sound, db):
    sound_rms_value = np.sqrt(np.mean((np.asarray(sound)-np.mean(np.asarray(sound)))**2))
    sound_rms_dB = 20.0*np.log10(sound_rms_value/2e-5)
    gain = 10**(((db-sound_rms_dB)/20.))
    leveled = gain*sound.copy()
    return leveled

def get_rms_dB(signal):
    rms_value = np.sqrt(np.mean((np.asarray(signal)-np.mean(np.asarray(signal)))**2))
    rms_dB = 20.0*np.log10(rms_value/2e-5)
    return rms_dB

