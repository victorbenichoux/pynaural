import numpy as np

try:
    import pyaudio
    has_pyaudio = True
except ImportError:
    has_pyaudio = False

try:
    import jack
    has_jack = True
except ImportError:
    has_jack = False

class AudioServer(object):
    def __init__(self, n_input_channels, n_output_channels, samplerate, buffer_size):
        pass
    
    def close(self):
        pass

    def process(self):
        pass

    def open(self):
        pass

class JackServer(AudioServer):
    def __init__(self):
        pass
    
    def open(self):
        self.client = jack.attach('PythonClient')
        myname = jack.get_client_name()

        jack.register_port("in_1", jack.IsInput)
        jack.register_port("in_2", jack.IsInput)
        jack.register_port("out", jack.IsOutput)
        client.activate()

        print "Jack ports (before):", jack.get_ports()
        jack.connect("system:capture_1", myname+":in_1")


    def print_client():
        print "Client:", myname

    def print_ports():
        print "Jack ports (before):", jack.get_ports()

    
    def process(self, output_buffer, input_buffer):
        try:
            jack.process(output_buffer, input_buffer)
        except jack.InputSyncError:
            print("InputSyncError")

    def close(self):
        jack.detach()
        jack.deactivate()


def get_rms_dB(signal):
    rms_value = np.sqrt(np.mean((np.asarray(signal)-np.mean(np.asarray(signal)))**2))
    rms_dB = 20.0*np.log10(rms_value/2e-5)
    return rms_dB

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


def load_word(wordnumber):
    wordlist = ['aunt', 'badge', 'freeze', 'quest', 'wine', 'wound']
    x,_= loadwave('./stimuli/%s.wav' % wordlist[int(wordnumber)])
    return x-np.mean(x)

def play_sound(p, sound, fs = 44100, channels = 1):
    stream = p.open(format=8, #p.get_format_from_width(wf.getsampwidth()),
                    channels=channels, #wf.getnchannels(),
                    rate=fs, #wf.getframerate(),
                    output=True)
        
    x = np.array((2 ** 15 - 1) * np.clip(sound, -1, 1), dtype=np.int16)
    stream.write(x, len(x))
    stream.stop_stream()
    stream.close()

def atlevel(sound, db):
    sound_rms_value = np.sqrt(np.mean((np.asarray(sound)-np.mean(np.asarray(sound)))**2))
    sound_rms_dB = 20.0*np.log10(sound_rms_value/2e-5)
    gain = 10**(((db-sound_rms_dB)/20.))
    leveled = gain*sound.copy()
    return leveled
