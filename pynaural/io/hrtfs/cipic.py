from spatializer.dsp import ImpulseResponse
from spatializer.geometry.base import cartesian2spherical, rad2deg
from brian.stdunits import kHz
from scipy.io import loadmat
import numpy as np
import os

CIPICfolder = 'D:/HRTF/CIPIC/CIPIC_hrtf_database/standard_hrir_database'

def load_CIPIC_subject(subject, folder = CIPICfolder, 
                       coordsfilter = lambda azim, elev: True):
    '''
    Loads a CIPIC subject.
    
    (website: http://interface.cipic.ucdavis.edu/sound/hrtf.html)
    
    * Arguments * 

    ``subject'' is the integer number of the subject
    
    ** Keywords **
    
    ``folder'' controls where to look for the hrtfs (the folder
    where the subject_xxx files are)

    ``coordsfilter'' a usual "lambda azim, elev:" type of filter
    
    '''
    # Apply the right syntax to find the file
    subjectstr = 'subject_%3d' % subject
    subjectstr = subjectstr.replace(' ', '0')
    filename = 'hrir_final.mat'
    path = os.path.join(folder, subjectstr, filename)
    # load the file
    matfile = loadmat(path)
    
    # global parameters
    # found on: http://interface.cipic.ucdavis.edu/data/doc/hrir_data_documentation.pdf
    
    # samplerate
    samplerate = 44.1*kHz
    
    # coordinates system
    elevation = lambda nel : -45 + 5.625*nel
    
    azimuthvals = np.hstack((np.array([-80,-65,-55,]),
                             np.arange(-45, 45, 5),
                             np.array([45, 55, 65, 80])))
    azimuth = lambda naz : azimuthvals[naz]

    # Note for the conversion:
    # It seems that for them elevation of 90+ means source is in the
    # back (assholes), so I have to figure out a way to change that.

    # fetch the HRIRs
    hrir_r = matfile['hrir_r']
    hrir_l = matfile['hrir_l']

    # prepare data structures    
    overall_shape = hrir_l.shape[0] * hrir_l.shape[1]
    print overall_shape
    dtype_coords = [('azim','f8'), ('elev','f8')]
    coordinates = np.zeros(overall_shape, dtype = dtype_coords)
    data = np.zeros((hrir_l.shape[2], 2 * overall_shape))
    
    onR = matfile['OnR']
    onL = matfile['OnL']
    onRfin = np.zeros(overall_shape)
    onLfin = np.zeros(overall_shape)


    # go!
    for naz in range(hrir_l.shape[0]):
        for nel in range(hrir_l.shape[1]):
            az_theirs = azimuth(naz)
            el_theirs = elevation(nel)
            az, el = convert_azel_cipic2spat(az_theirs, el_theirs)
            idx = naz * 50 + nel
            if coordsfilter(az, el):
                data[:, idx] = hrir_l[naz, nel, :]
                data[:, overall_shape+idx] = hrir_r[naz, nel, :]
                onRfin[idx] = onR[naz, nel]
                onLfin[idx] = onL[naz, nel]
                coordinates['azim'][idx] = az
                coordinates['elev'][idx] = el
            else:
                data[:,idx] = np.nan
                data[:,idx + overall_shape] = np.nan
                coordinates['azim'][idx] = np.nan
                coordinates['elev'][idx] = np.nan

# The problem is somewhere here.... You drop out half of the data by doing so!
#
#    datapoints  = -np.isnan(coordinates['azim'])
#    print datapoints
#    azims = coordinates['azim'][datapoints]
#    elevs = coordinates['elev'][datapoints]
#    data = data[:, datapoints]
#
#    coordinates = np.zeros(len(azims), dtype = dtype_coords)
#    coordinates['azim'] = azims
#    coordinates['elev'] = elevs
                           
    
    return ImpulseResponse(data, 
                           binaural = True,
                           coordinates = coordinates), onRfin, onLfin,data


def cipic2cartesian(az, el):
    x = np.cos(az) * np.cos(el)
    y = np.sin(az)
    z = np.cos(az) * np.sin(el)
    return x, y, z


def convert_azel_cipic2spat(az, el):
    x, y, z = cipic2cartesian(az, el)
    d, az, el = cartesian2spherical(x,y,z)
    return rad2deg(az), rad2deg(el)
    

