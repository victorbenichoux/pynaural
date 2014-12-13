import numpy as np
from pynaural.utils.spatprefs import get_pref

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
    def __init__(self):
        pass
    
    def close(self):
        pass

    def process(self):
        pass

    def open(self):
        pass

    def play(self, array):
        pass

class JackServer(AudioServer):
    def __init__(self):
        self.open()
    
    def open(self):
        self.client = jack.attach('PythonClient')
        myname = jack.get_client_name()

        jack.register_port("in_1", jack.IsInput)
        jack.register_port("in_2", jack.IsInput)
        jack.register_port("out", jack.IsOutput)
        self.client.activate()

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

    def play(self, sound):
        in_buf = np.zeros((sound.shape[0], 2))
        self.process(sound, in_buf)

    def close(self):
        jack.detach()
        jack.deactivate()

class PyAudioServer(object):
    def __init__(self):
        self.server = pyaudio.PyAudio()
        self.fs = get_pref('DEFAULT_SAMPLERATE')

    def play(self, sound, fs = None):
        if sound.shape[1] == 1:
            np.tile(np.asarray(sound), (1, 2))

        channels = sound.shape[1]

        fs = fs or self.fs

        stream = self.server.open(format=8,
                    channels=channels,
                    rate=int(fs),
                    output=True)

        x = np.array((2 ** 15 - 1) * np.clip(sound, -1, 1), dtype=np.int16)
        stream.write(x, len(x))
        stream.stop_stream()
        stream.close()


# A function to get the default audio server
if has_jack or has_pyaudio:
    if has_pyaudio:
        get_default_audioserver = lambda: PyAudioServer()
    else:
        get_default_audioserver = lambda: JackServer()
else:
    get_default_audioserver = lambda: None

def play_sound(sound):
    server = get_default_audioserver()
    server.play(sound)
