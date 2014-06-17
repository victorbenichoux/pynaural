from spatializer.dsp import ImpulseResponse
from spatializer.geometry.base import cartesian2spherical, rad2deg
from brian.stdunits import kHz
from scipy.io import loadmat
import numpy as np
import os


def ircamHRIR(subject, coordsfilter = None, path = None, compensated = False):
        '''
        This static method enables one to load the IRCAM HRIRs.
        
        Usage:
        hrir = ircamHRIR(subject, coordsfilter = None, path = None)
        
        Keyword arguments:
        
        ``coordsfilter'' : filters the coordinates to be loaded

        ``path'' : specify this and the IRC_xxxx file will be looked for in this folder, otherwise uses the IRCAMpath default preference.
        '''
        if path == None:
            path = prefs.get_pref('IRCAMpath', default = '')

        if not isinstance(path, (list, tuple)):
            path = [path]
        
        names = []
        for curpath in path:
            names += glob(os.path.join(curpath, 'IRC_*'))
        splitnames = [os.path.split(name) for name in names]
        
        p = re.compile('IRC_\d{4,4}')
        subjects = [int(name[4:8]) for base, name in splitnames 
                    if not (p.match(name[-8:]) is None)]


        subject = str(subject)
        if subject[0] == '3':
            # this is the case only for stuffed animals recordings
            # IRC_30..
            samplerate = 192*kHz
        else:
            samplerate = 44.1*kHz
        
        ### Beginning of code from the Brian ircam.py by Dan
        ok = False
        k = 0

        while k < len(path) and not ok:
            try:
                filename = os.path.join(path[k], 'IRC_' + subject)
                if compensated:
                    filename = os.path.join(filename, 'COMPENSATED/MAT/HRIR/IRC_' + subject + '_C_HRIR.mat')
                else:
                    filename = os.path.join(filename, 'RAW/MAT/HRIR/IRC_' + subject + '_R_HRIR.mat')
                m = loadmat(filename, struct_as_record=True)
                ok = True
            except IOError:
                ok = False
            k += 1
        if not ok:
            print('''Couldn't find the HRTF files for subject %d''' % int(subject))
            print('Was looking in: %s'%path)
            print('Found subjects %s'%str(subjects))
            raise IOError

        if 'l_hrir_S' in m.keys(): # RAW DATA
            affix = '_hrir_S'
        else:                      # COMPENSATED DATA
            affix = '_eq_hrir_S'
        l, r = m['l' + affix], m['r' + affix]

        azim = l['azim_v'][0][0][:, 0]
        elev = l['elev_v'][0][0][:, 0]

        if len(azim) == len(elev) and len(azim) == 1:
            azim = l['azim_v'][0][0][0, :]
            elev = l['elev_v'][0][0][0, :]
        
        l = l['content_m'][0][0]
        r = r['content_m'][0][0]

        data = vstack((reshape(l, (1,) + l.shape), reshape(r, (1,) + r.shape)))
        
        # not sure I need to do that:
        dtype_coords = [('azim','f8'), ('elev','f8')]
        newcoords = np.zeros(len(azim), dtype = dtype_coords)
        for i in range(len(newcoords)):
            newcoords['azim'][i] = az_ircam2spat(azim[i])
        newcoords['elev'] = elev
        
        newdata = np.zeros((data.shape[2], data.shape[1] * 2))

        newdata[:, :data.shape[1]] = data[0,:,:].transpose()
        newdata[:, data.shape[1]:] = data[1,:,:].transpose()
        
        hrir = ImpulseResponse(newdata, 
                               coordinates = newcoords, 
                               binaural = True, 
                               samplerate = samplerate)
        
        if not coordsfilter == None:
            hrir = hrir.forcoordinates(coordsfilter)
        
        return hrir
