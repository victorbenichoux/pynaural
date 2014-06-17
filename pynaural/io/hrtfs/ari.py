'''
This script contains functions to load animals' HRTFs.
'''
from spatializer import *
import pickle, re
import numpy as np
import os, inspect
from brian import *
_, name, _, _, _ = os.uname()


#################### ARI database HRTFs ##################################


def load_ARI_subject(subject, folder = ARIfolder):
    
    # Apply the right syntax to find the file
    subject_filename = folder+'hrtf_'+str(subject)+'_hrtf.mat'
    
    # load the file
    matfile = loadmat(subject_filename)

    # samplerate
    samplerate = matfile['stimPar']['SamplingRate'][0][0][0][0]
    
    # coordinates system
    azimuth   = real(matfile['meta']['pos'][0][0][:,0])
    azimuth[azimuth>180.0] = azimuth[azimuth>180.0] -360.0
    elevation = real(matfile['meta']['pos'][0][0][:,1])
    
    # HRIRs
    hrir_raw = matfile['hM']

    # prepare data structures    
    overall_shape = hrir_raw.shape[1]
    dtype_coords = [('azim','f8'), ('elev','f8')]
    coordinates = np.zeros(overall_shape*2 , dtype = dtype_coords)
    data = np.zeros((hrir_raw.shape[0], 2*overall_shape))
    
    # go!
    for n in range(overall_shape):
        data[:, n]                 = hrir_raw[:,n,0]
        data[:, n + overall_shape] = hrir_raw[:,n,1]
        coordinates['azim'][n]               = azimuth[n]
        coordinates['azim'][n+overall_shape] = azimuth[n]
        coordinates['elev'][n]               = elevation[n]
        coordinates['elev'][n+overall_shape] = elevation[n]
           
    
    return ImpulseResponse(data, 
                           binaural = True,
                           coordinates = coordinates)
    
