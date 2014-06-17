'''
Contains additional stuff to load sounds, mostly for the zoom, with support for the markers function
original script, doc and such can be found in the Stuff/zoom marquage folder
'''
import wave 
import struct
import numpy as np
from matplotlib.pyplot import *
from brian.hears import Sound, dB
from brian.stdunits import Hz
import time
from scikits.audiolab import Sndfile
import os
import numpy as np

def load_nist(fn):
    '''
    Load a NIST sphere WAV file, returns a Sound object
	(uses scikits.audiolab)

    fn : sound file name
    '''
    f1 = Sndfile(fn, 'r')
    samplerate = f1.samplerate
    sound = f1.read_frames(f1.nframes)
    return Sound(sound, samplerate = samplerate*Hz)


def load_wav24(fn):
    '''
    loads a 2 channel, 24 bit wav file

    fn : sound file name
    '''
    wav = wave.open(fn, "r")
    nchannels, sampwidth, framerate, nframes, comptype, compname = wav.getparams()
    frames = wav.readframes(nframes * nchannels)

    ZERO = struct.pack('B', 0)
    res = ''.join([ZERO+frames[3*k]+frames[3*k+1]+frames[3*k+2] for k in xrange(nframes * nchannels)])
    data = np.fromstring(res, dtype = np.dtype("i"))
    data = np.array(data, dtype = 'double').reshape((nframes, nchannels))
    data -= 2**12
    data /= 2**23
    return Sound(data, framerate * Hz)

############### ZOOM Marker data ######################
# stuff to load marker position recorded using a ZOOM handheld recorder (and usually found in wav24 format
def readnumber(f):
    c = f.read(4)
    if len(c)<4:
        error("Sorry, no cue information found.")
    return sum(ord(c[i])*256**i for i in range(4))

def get_labels(fn):
    f = open(fn,"r")
    if f.read(4) != "RIFF":
        print "Unknown file format (not RIFF)"
        return None
    f.read(4)
    if f.read(4) != "WAVE":
        print "Unknown file format (not WAVE)"
        return None
    name = f.read(4)
    while name != "cue ":
        leng= readnumber(f)
        f.seek(leng,1) # relative skip
        name = f.read(4)
    leng= readnumber(f)
    num = readnumber(f) # number of found markers
    if leng != 4+24*num:
        error("Inconsistent length of cue chunk")

    positions = []
    oldmarker = 0.0
    for i in range(1,num+1):
#        cuename = readnumber(f)
#        cuepos = readnumber(f)
#        cuechunk = f.read(4)
#        cuechunkstart = readnumber(f)
#        cueblockstart = readnumber(f)
        f.read(20) # remove if you remove previous lines
        cueoffset = readnumber(f)
        positions.append(cueoffset)
    return positions

    
    
