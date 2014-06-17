'''
This script contains functions to load animals' HRTFs.
'''
from spatializer import *
import pickle, re
import numpy as np
import os, inspect
from brian import *
_, name, _, _, _ = os.uname()

BOfolder = ''
CATFOLDER = ''
TCfolder = ''
SPHfolder = ''
KRdata = ''

# if you're using this script, don't hesitate to put your computer, just figure out the output of os.uname in a console and put another elif down there.
if name == 'victor-desktop':
    BOfolder = '/home/victor/Work/Data/HRTF/deconvolved Owl HRTFs/'
    KRdata = '/home/victor/Work/Data/HRTF/Rabbit (from Shig Kuwada)/'
    CATFOLDER = '/home/victor/Work/Data/HRTF/Synthesized cat/'
    SPHfolder = '/home/victor/workspace/hrtf_analysis/data/ircam_spherical'
    TCfolder = '/home/victor/Work/Data/HRTF/Cat (from Daniel Tollin)/Raw HRTFs/'
elif name == 'victor-laptop':
    TCfolder = '/home/victor/Work/Data/HRTF/Cat (from Daniel Tollin)/Raw HRTFs/'
    BOfolder = '/Users/victorbenichoux/data/HRTF/'
elif name == 'Victors-MacBook-Air-2.local':
    print 'Sur le macbook air'
    BOfolder = '/Users/victorbenichoux/data/HRTF/'
    TCfolder = '/Users/victorbenichoux/data/HRTF/Cat (from Daniel Tollin)/Raw HRTFs/'
    CATFOLDER = '/Users/victorbenichoux/data/HRTF/Synthesized cat/'
elif name == 'Victor-Benichouxs-iMac.local':
    print 'Sur le mac maison'
    TCfolder = '/Users/victorbenichoux/Workspace/hrtf_analysis/new/Cat (from Daniel Tollin)/Raw HRTFs/'
    CATFOLDER = '/Users/victorbenichoux/Workspace/Data/HRTF/synthcat/'
else:
    CATFOLDER = '/Users/victorbenichoux/data/HRTF/Synthesized cat/'
    print "Unrecognized computer %s" % name


HUMANS = {    
    'claire': 2024,
    'cyrille': 2026,
    'trevor': 2027,
    'romain': 2028,
    'boris': 2029,
    'leny': 2030,
    'victor': 2031,
    'jonathan': 2032
    }

ANIMALS = {
    'chinchilla441': 2033,
    'rat441': 2034,
# @ 192*kHz
    'rat': 3010,
    'chinchilla': 3012,
    'cobaye': 3012,
    'guineapig': 3012,
    'lapin': 3013,
    'rabbit': 3013,
    'chat': 3014,
    'cat':3014,
    'macaque': 3016,
    'chouette': 3017,
    'owl': 3017
    }



######################### Barn owls ##############################


owls = ['Zhaki', 'Pavarotti', 'Hurinfine', 'Hurin', 'Goldi', 'Scottie']

def wagnerHRIR(name = 'Goldi', folder = BOfolder):
    '''
    This function loads the deconvolved HRTFs of the barn owls measured by Herman Wagner.
    '''
    print 'Loading owl '+name
    f = open(folder+'owl_'+name+'_meta.pkl', 'rb')
    meta = pickle.load(f)
    f.close()
        
    f = open(folder+'owl_'+name+'_data.npy', 'rb')
    data = np.load(f)
    f.close()
        
    hrir = ImpulseResponse(data, coordinates = meta['coordinates'],
                           samplerate = meta['samplerate'], 
                           binaural = True)

    hrir.name = meta['subjectname']

    return hrir

######################### Rabbit ##############################


def load_rabbit(folder = KRdata, 
                distance = 160.*cm, 
                dataset = 'direct', 
                duration = 50*ms):
    '''
    This function loads the Rabbit's HRTFs measured by Shigh Kuwada.
    
    
    All of those HRIRs are band-pass filtered from 200 Hz - 20 kHz

    ``folder'' The folder containing the mat files
   
    ``dataset = 'direct' '' : 
    'full' contains the HRIR obtained from our raw data.
    'Direct' has the same fields but the data has been time windowed about the peak of the impulse response. The time window is a 20 millisecond blackman window.
    ``duration'': trim the IR at the given duration (otherwise is quite long!)

    '''
    azs = []
    els = []
    dists = []
    
    p = re.compile("(\d+)_([-]*\d+)_(\d+)")    
    
    dataR = []
    dataL = []
    
    for file_str in os.listdir(folder):
        fn, ext = file_str.split('.')
        if ext == 'mat':
            match = p.match(file_str)
            if match != None:
                dist, azim, elev = match.group(1, 2, 3)
                if float(int(dist)*cm) == float(distance):
                    data = loadmat(folder+file_str)['brir']
                    samplerate = data['Fs']
                    azs.append(data['azimuth'])
                    els.append(data['elevation'])
                    dataR.append(data[dataset][0,0]['hR'][0,0].flatten())
                    dataL.append(data[dataset][0,0]['hL'][0,0].flatten())
   
    azs = array(azs)
    _, pos = unique(mod(azs+180, 360)-180, return_index = True)#get rid of +180 and -180

    samplerate = samplerate[0][0][0][0]
    nsamples = dur2sample(duration, samplerate)
    
    Npairs = len(dataR)
    data = zeros((nsamples, Npairs*2))
    dtype_coords = [('azim','f8'), ('elev','f8')]
    coordinates = np.zeros(Npairs, dtype = dtype_coords)

    for k,i in enumerate(pos):

        data[:,k] = dataL[i][:nsamples]
        data[:,k+Npairs] = dataR[i][:nsamples]
        coordinates['azim'][k] = -float(azs[i][0][0][0][0])
        coordinates['elev'][k] = float(els[i][0][0][0][0])


    hrir = ImpulseResponse(data, coordinates = coordinates, binaural = True, 
                           samplerate = samplerate*Hz)
    hrir.distance = distance
    hrir.dataset = dataset
    return hrir

######################### Synthetic cat ##############################

import array as pyarray

#CATFOLDER = '/home/victor/Work/Data/HRTF/Synthesized cat/'


def load_syntheticcat(folder = CATFOLDER):
    folder += 'hrir_cat/horiz'
    N = 72
    Nsamples = 1024
    data = zeros((Nsamples, 2*N))

    dtype_coords = [('azim','f8'), ('elev','f8')]
    coords = zeros(N, dtype = dtype_coords)


    for i, azimuth in enumerate(xrange(0, 360, 5)):
        fname_l = folder+'L'+str(azimuth)+'.dat'
        fname_r = folder+'R'+str(azimuth)+'.dat'
        L = open(fname_l, 'rb').read()
        L = pyarray.array('d', L)
        data[:, i] = array(L)
        R = open(fname_r, 'rb').read()
        R = pyarray.array('d', R)
        R = array(R)
        data[:, i + N] = array(R)
        
        coords['azim'][i] = az_ircam2spat(azimuth)
        coords['elev'][i] = 0

    return ImpulseResponse(data, 
                           coordinates = coords, 
                           binaural = True, samplerate = 44.1*kHz)

def load_syntheticcat_new(folder = CATFOLDER, coordsfilter = lambda azim, elev: True):
    '''
    Synthetic Cat Loader
        
    Loads the new hrtf data from Makoto Otami.
    
    Usage:
    
    hrir = load_syntheticcat_new(folder = 'path/to/files', coordsfilter = lambda azim, elev: azim == 40)

    Note:
    The folder keyword argument must be the path to the folder containing the 'hrir_cat_new_clean' folder.
    '''
    

    folder = os.path.join(folder, 'hrir_cat_new_clean/')
    coordsfile = os.path.join(folder, 'coords.txt')

    Nsamples = 1024
    
    f = open(coordsfile, 'r')
    lines = f.readlines()
    Ncoords = len(lines)
    coords = np.zeros(Ncoords, dtype = [('azim','f8'), ('elev','f8')])
    to_be_loaded = np.zeros(Ncoords, dtype = bool)
    for k, line in enumerate(lines):
        az, el = line[1:-1].strip('()').split(',')
        coords['azim'][k] = float(az)
        coords['elev'][k] = float(el)
        if coordsfilter(coords['azim'][k], coords['elev'][k]):
            to_be_loaded[k] = True
    f.close()    

    tbl_index = np.nonzero(to_be_loaded)[0]
    N = len(tbl_index)
    data = np.zeros((Nsamples, N*2))
    
    for i, k in enumerate(tbl_index):
        fname_l = folder+'L'+str(k+1)+'.dat'
        fname_r = folder+'R'+str(k+1)+'.dat'
        L = open(fname_l, 'rb').read()
        L = pyarray.array('d', L)
        data[:, i] = array(L)
        R = open(fname_r, 'rb').read()
        R = pyarray.array('d', R)
        R = array(R)
        data[:, i + N] = array(R)

    coords = coords[to_be_loaded]

    return ImpulseResponse(data,
                           coordinates = coords, 
                           binaural = True, samplerate = 44.1*kHz)

# Last version synthcat loading


def get_dataset_info(dataset):
    folders = {
        'uniform' : 'catHRIRsdata441_uniform/',
        'centered' : 'catHRIRsdata441_body_centered/',
        'aligned' : 'catHRIRsdata441_aligned/'
        }
    
    coordsfiles = {
        'uniform' : 'hrtfpositions_uniform.txt',
        'centered' : 'coords.txt',
        'aligned' : 'hrtfpositions.txt'}
    
    distances_present = {
        'uniform' : True,
        'centered' : False,
        'aligned' : True
        }
    
    return {
        'folder' : folders[dataset],
        'coordsfile' : coordsfiles[dataset],
        'distances_present' : distances_present[dataset]
        }
        
    

def get_synthcat_coords(folder = CATFOLDER, 
                        dataset = 'aligned'):
    '''
    Returns the coordinates of the synthcat
    '''
    
    info = get_dataset_info(dataset)
    
    folder = os.path.join(folder, info['folder'])
    coordsfile = os.path.join(folder, info['coordsfile'])

    Nsamples = 1024
    
    f = open(coordsfile, 'r')
    lines = f.readlines()
    Ncoords = len(lines)
    
    if info['distances_present']:
        dtype = [('azim','f8'), ('elev','f8'), ('dist', 'f8')]
    else:
        dtype = [('azim','f8'), ('elev','f8')]
    
    coords = np.zeros(Ncoords, dtype = dtype)
    koffset = 0
    for k, line in enumerate(lines):
        if line[0] == '#':
            koffset  -= 1
#            if el [-2] == ')': # Hack: buggy behavior with the macbook air and the centered cat dataset
#                el = el[:-2]
        else:
            k += koffset
            if not info['distances_present']:
                az, el = line[:-1].strip('()').split(',')
            else:
                d, az, el = line[:-1].strip('()').split(',')
            coords['azim'][k] = float(az)
            coords['elev'][k] = float(el)
            if info['distances_present']:
                coords['dist'][k] = float(d)
                
            if k < 10 and False:
                print line, d
    f.close()
    return coords

def get_aligned_sources():
    '''
    Returns the Source positions for all the hrtf positions computed by makoto
    '''
    # basically a copy/paste of what is contained in the hrtf_location_final.py script
    cat_height = 0.2
    # under step2, (because step1 was only double the IRCAM positions)
    distances = [.25, .50, 1, 2, 5]
    heights = [.02, .2, 2]

    # for each distance, 3 heights
    ds = []
    hs = []
    for d in distances:
        ds += [d]*len(heights)
        hs += heights
    ds = list(np.sqrt(np.array(hs)**2 + np.array(ds)**2))
        
    elevations = list(np.arctan(np.array(hs)/np.array(ds)))
    
    azimuths = np.arange(-np.pi, np.pi, step = np.pi/72.)
    
    final_distances = []
    final_elevations = []
    final_azimuths = []
    
    for az in azimuths:
        final_azimuths += [az]*len(ds)
        final_elevations += elevations
        final_distances += ds

    
    positions = zip(rad2deg(final_azimuths), rad2deg(final_elevations), final_distances)
    
    # under step3
    distances = np.hstack((np.arange(.2, .5, step = .01),
                           np.arange(.5, 1, step = .02),
                           np.arange(1, 2, step = .1),
                           2))
    azimuths = list(np.arange(-np.pi, np.pi, step = np.pi/4))
    
    final_distances = []
    for d in distances:
        final_distances += [d] * len(azimuths)
        final_azimuths += azimuths
        
    final_elevations = np.zeros_like(final_azimuths)

    cat = OrientedReceiver(cat_height*UP, FRONT)
    positions+= zip(rad2deg(final_azimuths), rad2deg(final_elevations), final_distances)
    
    return positions

def load_syntheticcat_v3(folder = CATFOLDER, coordsfilter = lambda azim, elev: True,
                         dataset = "uniform"):
    '''
    Synthetic Cat Loader
        
    Loads the new new hrtf data from Makoto Otami.
    
    Usage:
    
    hrir = load_syntheticcat_new(folder = 'path/to/files', coordsfilter = lambda azim, elev: azim == 40)
    
    Note:
    the kwd argument 'dataset' controls which data set to load, loading files in the different folders as follows:
    'uniform' : 'catHRIRsdata441_uniform/',
    'centered' : 'catHRIRsdata441_body_centered/',
    'aligned' : 'catHRIRsdata441_aligned/'
    
    
    - uniform : aligned origin (with body), straight head, uniform positions
    - centered: aligned origin (with body), head to the side, uniform positions
    - aligned: aligned origin (with body), straight head, positions specified for reflection simulation. For the source positions, refer to get_aligned_source_positions. For the initial script (that generated the hrtfpositions.txt), look at hrtf_location_final.py
    
    Note: 'aligned' dataset also contains the IRCAM doubled positions
    
    Note:
    The folder keyword argument must be the path to the folder containing the 'hrir_cat_new_clean' folder.
    '''
    
    info = get_dataset_info(dataset)
    print "looking into ", folder
    folder = os.path.join(folder, info['folder'])
    coordsfile = os.path.join(folder, info['coordsfile'])

    Nsamples = 1024
    try:
        f = open(coordsfile, 'r')
    except:
        print 'Couldnt find the files, was looking in '+coordsfile
        raise
    lines = f.readlines()
    Ncoords = len(lines)

    if info['distances_present']:
        dtype = [('azim','f8'), ('elev','f8'), ('dist', 'f8')]
    else:
        dtype = [('azim','f8'), ('elev','f8')]


    coords = np.zeros(Ncoords, dtype = dtype)
        
    to_be_loaded = np.zeros(Ncoords, dtype = bool)
    koffset = 0
    for k, line in enumerate(lines):
        if not (line[0] == '#'):
            k += koffset
            
            if info['distances_present']:
                d, az, el = line[1:-1].strip('()').split(',')
                coords['dist'][k] = float(d)
            else:
                az, el = line[1:-1].strip('()').split(',')
            
            coords['azim'][k] = float(az)
            
            try:
                coords['elev'][k] = float(el)
            except:
                coords['elev'][k] = float(el[:-2])
                
            if callable(coordsfilter):
                if len(inspect.getargspec(coordsfilter).args) == 2:
                    if coordsfilter(coords['azim'][k], coords['elev'][k]):
                        to_be_loaded[k] = True
                else:
                    if coordsfilter(coords['azim'][k], coords['elev'][k], coords['dist'][k]):
                        to_be_loaded[k] = True
            elif isinstance(coordsfilter, int):
                if coordsfilter == k:
                    to_be_loaded[k] = True
        else:
            koffset -= 1

    f.close()    

    tbl_index = np.nonzero(to_be_loaded)[0]
    N = len(tbl_index)
    data = np.zeros((Nsamples, N*2))
    
    for i, k in enumerate(tbl_index):
        fname_l = folder+'L'+str(k+1)+'.dat'
        fname_r = folder+'R'+str(k+1)+'.dat'
        L = open(fname_l, 'rb').read()
        L = pyarray.array('d', L)
        data[:, i] = array(L)
        R = open(fname_r, 'rb').read()
        R = pyarray.array('d', R)
        R = array(R)
        data[:, i + N] = array(R)

    coords = coords[to_be_loaded]

    return ImpulseResponse(data, 
                           coordinates = coords, 
                           binaural = True, samplerate = 44.1*kHz)

############################### Tollin cat ##########################



def load_Tollin_cat(path_to =  TCfolder,
                    coordsfilter = lambda azim, elev: True):
    '''
    loads Tollin's cat HRTFs
    they are simply responses to clicks (so IRs directly)
    '''
    
    Nsamples = 1999
    samplerate = 97656.25
    Nazs = 36

    coords = zeros(Nazs, dtype = [('azim','f8'), ('elev','f8')])

    coords['elev'] = zeros(Nazs)

    data = zeros((Nsamples, Nazs*2))
    
    prefix = 'nov24_'
    postfix = '_7.txt'
    
    for k in range(Nazs):
        az = k * 10
        az_str = str(az)
        coords['azim'][k] = mod(-az + 90 + 180 , 360) - 180
        try:
            f = open(path_to+prefix+az_str+postfix, 'rb')
        except IOError:
            print 'Couldn\'t find the HRTF files, was looking into '+path_to+prefix+az_str+postfix
            return
        l = f.readline()
        i = 0
        while l != '':
            tmp = (l.strip('\r\n')).split(' ')
            t, data[i,k], data[i, k+ Nazs] = map(float, tmp)
            l = f.readline()
            i += 1
        
    f.close()
    
    res = ImpulseResponse(data, samplerate = samplerate * Hz,
                  binaural = True,
                  coordinates = coords)
    return res.forcoordinates(coordsfilter)

if __name__ == '__main__':
    hrir = load_syntheticcat_new(coordsfilter = lambda azim, elev: elev == 0)
