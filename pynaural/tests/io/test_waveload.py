from pynaural.io.sndfile_loaders_old import *

path = './wavefiles/'

success = []

for k in range(len(os.listdir(path))-1):
    fn = path + str(k) + '.wav'
    print fn
    try:
        print 'o'
        s = Sound.load(fn)
        success.append(1)
        print 'succes direct'
    except:
        try:
            #load_nist(fn)
            #load_wav24(fn)
            success.append(1)
        except:
            success.append(0)

print success