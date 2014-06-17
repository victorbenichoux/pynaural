'''
This script contains functions to load animals' HRTFs.
'''
from spatializer import *
import pickle, re
import numpy as np
import os, inspect
from brian import *
_, name, _, _, _ = os.uname()


#################### QU database HRTFs ##################################


# Set this to your default folder
QUfolder = "/home/victor/Work/Data/HRTF/Qu-HRTF"

# Helper functions
def get_qu_coordinates(folder = QUfolder):
    Qu_folder = os.path.join(folder, 'Qu-HRTF database')
    
    #p = re.compile("a*zi(\d+)_elev([-]*\d+)_dist(\d+).dat")    
    p = re.compile("azi(\d+)_elev([-]*\d+)_dist(\d+).dat")    

    azs, els, dists = [], [], []

    
    for dist_folder in os.listdir(Qu_folder):
        cur_dist_folder = os.path.join(Qu_folder, dist_folder)
        for elev_folder in os.listdir(cur_dist_folder):
            cur_elev_folder = os.path.join(cur_dist_folder, elev_folder)
            for filename in os.listdir(cur_elev_folder):
                match = p.match(filename)
                if not (match is None):
                    #print filename, match.group(1, 2, 3)
                    # then the file is valid
                    az, el, dist = match.group(1, 2, 3)
                    azs.append(float(az))
                    els.append(float(el))
                    dists.append(float(dist)/100)
                    
                    
                else:
                    #print 'unrecognized file ', filename
                    pass

    dtype_coords = [('azim','f8'), ('elev','f8'), ('dist', 'f8')]        
    coordinates = np.zeros(len(azs), dtype = dtype_coords)
    coordinates['azim'] = azs
    coordinates['elev'] = els
    coordinates['dist'] = dists
    return coordinates
    
def azeldist_to_filename(azim, elev, dist):
    '''
    Given an azim, elev and distance, returns the proper folder hierarchy and filename pointing to the '.dat' file.
    '''
    str_dist_cm = str(int(dist*100))
    str_elev = str(int(elev))
    str_azim = str(int(azim))
    
    dist_folder = 'dist' + str_dist_cm
    elev_folder = 'elev' + str_elev
    
    fn = 'azi'+str_azim+'_elev'+str_elev+'_'+dist_folder+'.dat'
    
    return os.path.join(dist_folder, elev_folder, fn)

# Main function

def load_qu_hrtf(coordsfilter = lambda azim, elev, distance: True, folder = QUfolder):
    '''
    Loads the HRTFs with a condition on the coordinates. The condition is expressed with a lambda function with three arguments (az, el, dist), it should return a boolean value. Have a look at the example below.

    The kwd argument folder should be set to the folder _containing_ the 'Qu-HRTF database' folder from the archive.
    
    The return format is a simple ImpulseResponse object.

    Example use: 
    
    hrir = load_qu_hrtf(coordsfilter = lambda azim, elev, distance: (distance == 1.3) * (azim >= 0), folder = 'path-to-the folder')
    '''
    coordinates = get_qu_coordinates(folder)

    valid_positions = coordsfilter(coordinates['azim'], coordinates['elev'], coordinates['dist'])
    if isinstance(valid_positions, bool):
        valid_coordinates = coordinates
    else:
        valid_coordinates = coordinates[valid_positions]

    print 'Loading %d HRTF positions' % (len(valid_coordinates))
    
    Ncoordinates = len(valid_coordinates)
    
    data = np.zeros((1024, Ncoordinates*2))
    
    for k in range(Ncoordinates):
        azim, elev, dist = valid_coordinates[k]['azim'], valid_coordinates[k]['elev'], valid_coordinates[k]['dist']
        
        filename = azeldist_to_filename(azim, elev, dist)
        
        total_filename = os.path.join(folder, 'Qu-HRTF database',  filename)

        fread = open(total_filename, 'rb').read()
        tmparray = pyarray.array('d', fread)
        res = np.array(tmparray)
        
        left = res[:1024]
        right = res[1024:]
        
        data[:,k] = left
        data[:,k+Ncoordinates] = right
        
    return ImpulseResponse(data, coordinates = coordinates, binaural = True, samplerate = 65536.*Hz)

