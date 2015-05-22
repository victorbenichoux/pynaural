from tables import openFile
import numpy as np
import pynaural.signal.impulseresponse

fn = "/home/victor/Downloads/mit_kemar_normal_pinna.sofa"
h5f = openFile(fn)

def make_coordinates_array(x):
    if x.shape[1] != 3:
        return ValueError("Input array should be of shape (n, 3)")
    out = np.zeros(x.shape[0], dtype = [('azim','f8'), ('elev','f8'), ('dist', 'f8')])
    out['azim'][:] = x[:,0]
    out['elev'][:] = x[:,1]
    out['dist'][:] = x[:,2]
    return out

def az_sofa2pyn(az):
    '''
    Converts azimuth positions between sofa and pynaural (same as ircam->pynaural)
    '''
    if az < 0 or az > 360:
        raise ValueError('Wrong azimuth for conversion IRCAM->spatializer')
    if az <= 180:
        return az
    else:
        return -(360-az)

def sofaHRIR(fn, return_info = False):
    '''
    This static method loads SOFA HRIRs.

    Usage:
    hrir = sofaHRIR(filename)

    '''
    h5f = openFile(fn)

    ir = h5f.root._v_children['Data.IR'][:]
    samplerate = h5f.root._v_children['Data.SamplingRate'][:][0]

    ir_data = np.hstack((ir[:,0,:], ir[:,1,:]))

    tmp_coords = h5f.root._v_children['SourcePosition'][:].squeeze()
    tmp_coords[:,0] = az_sofa2pyn(tmp_coords[:,0])

    out = pynaural.signal.impulseresponse.ImpulseResponse(ir_data, coordinates = make_coordinates_array(tmp_coords), binaural = True, samplerate = samplerate)

    if return_info:
        info_dict = dict(pre_delay = h5f.root._v_children['Data.Delay'][:][0],
            emitter_position = make_coordinates_array(h5f.root._v_children['EmitterPosition'][:]),
            listener_position = make_coordinates_array(h5f.root._v_children['ListenerPosition'][:]),
            receiver_position = make_coordinates_array(h5f.root._v_children['ReceiverPosition'][:][:,:,0]),
            listener_up = h5f.root._v_children['ListenerUp'][:].squeeze(),
            listener_view = h5f.root._v_children['ListenerView'][:].squeeze(),
            C = h5f.root._v_children['C'][:],
            E = h5f.root._v_children['E'][:][0],
            I = h5f.root._v_children['I'][:][0],
            M = h5f.root._v_children['M'][:].squeeze(),
            N = h5f.root._v_children['N'][:].squeeze(),
            R = h5f.root._v_children['R'][:].squeeze())
        return out, info_dict
    else:
        return out







