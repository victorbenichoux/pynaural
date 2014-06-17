from spatializer.dsp.impulseresponse import *
from spatializer.dsp.transferfunction import *
import numpy as np
from scipy import *

__all__ = ['AzimuthPhaseInterpolator']

class AzimuthPhaseInterpolator(object):
    def __init__(self, hrtf):
        if isinstance(hrtf, ImpulseResponse):
            hrtf = TransferFunction(hrtf)
        azs = hrtf.coordinates['azim']
        azorder = np.argsort(azs)
        self.ipds = np.angle(hrtf.left/hrtf.right)[:,azorder]
        self.azs = azs[azorder]
# class NewNewInterpolatingHRTFSet(HRTFSet):
#     '''
#     This class is (yet another) attempt at doing cool interpolation of HRTFs.
#     It basically computes the distance from the considered point to all the other points of the sphere (in possibly 2 different ways, the orthodromic one is really slow though, I should vertorize it).
#     It then applies a weighing kernel to the distance, and interpolates linearly with those weight values.
#     Sample usage:
#     hrtfset = NewNewInterpolatingHRTFSet(IRCAM_LISTEN(IRCAMpath, compensated = True).load_subject(subject))
#     hrtf = hrtfset.get_hrir(az,el, method = 'interpolate')
    
#     this returns an ImpulseResponse object that you can hear using the .listen() method.
    
#     NB: The sound is somewhat colored. I should look into that in more detail. Maybe implement that IPTF method thingy that I once saw on the internet, or do more complex cue based interpolation
    
#     Notable attribute:
#     ``compensatedcoordinates`` : real spatializer coordinates (not the other bullshitty IRCAM ones)
#     '''
#     def __init__(self, hrtfset, kernel = None, interpolation = 'linear'):
#         super(NewNewInterpolatingHRTFSet, self).__init__(hrtfset.data, hrtfset.samplerate, 
#                                                          hrtfset.coordinates)
#         if kernel is None:
#             # somewhat arbitrary value so that the point 2.5 degrees far is weighed with .5 value
#             dref = -0.02/np.log(.5)
#             self.kernel = lambda d: np.exp(-d/2)
#         self._makecoordinates(hrtfset)
#         self.interpolation = interpolation

    
#     def distances(self, az, el, method = 'dummy'):
#         '''
#         kwdargs are ``dummy`` or ``orthodromic``. The latter is really slow so just use dummy for now
#         '''
#         if method == 'orthodromic':
#             matrix = np.zeros(self.nhrtfs)
#             for i in range(self.nhrtfs):
#                 matrix[i] = orthodromic_distance(self.compensatedcoordinates[i], 
#                                                  (az,el))
#             return matrix
#         else:
#             return self.dummy_distance(az,el)
        
#     @property
#     def nhrtfs(self):
#         return len(self.coordinates)

#     def _makecoordinates(self, hrtfset):
#         coordinates = hrtfset.coordinates
#         elevations, _  = zip(*hrtfset.coordinates)
#         elevations = list(set(elevations))
#         elevations.sort()
#         self.elevations = np.array(elevations)
#         azperel = [[]]*len(self.elevations)
#         for i,el in enumerate(self.elevations):
#             tmp = [az_ircam2spat(azim) for (elev, azim) in coordinates if elev == el]
#             tmp.sort()
#             azperel[i] = np.array(tmp, dtype = int)
#         self.azperel = azperel

#     @property
#     def compensatedcoordinates(self):
#         '''
#         spits out coordinates for use with the rest of the
#         spatializer, that is arranged in (az,el) value pairs with the
#         correct (mine) azimuth convention
#         '''
#         res = []
#         for (el, az) in self.coordinates:
#             res.append((az_ircam2spat(az), el))
#         return res
    
#     def dummy_distance(self, az, el):
#         rot = az*degAz + el*degEl
#         ref = FRONT
#         point = rot*FRONT
#         point = np.tile((rot*FRONT).array().reshape((3,1)), (1,len(self)))
#         if not hasattr(self, 'points'):
#             self.computepoints()
#         distances = np.sum((point - self.points)**2, axis = 0)
#         return distances
    
#     def computepoints(self):
#         self.points = np.zeros( (3, len(self)) )
#         for i,(az,el) in enumerate(self.compensatedcoordinates):
#             rot = az*degAz + el*degEl
#             ref = FRONT
#             point = rot*FRONT
#             self.points[:, i] = point.array().flatten()

#     def get_hrir(self, az, el, method = None):
#         '''
#         methods: 
#         closest (self explanatory)
#         linear: weights all the hrtfs by a function (kernel) of the distance
#         exponential: this is a weird method of mine that i think might be inspired by this paper I read once.
#         '''
        
#         if method == 'interpolate':
#             print 'Replace the call of interpolate by the real method (i.e linear)'
#             method = 'linear'
#         if method is None:
#             method = self.interpolation

#         if method == 'linear':
#             weights = self.kernel(self.distances(az,el))
#             weights /= np.sum(weights)
#             res = np.zeros((len(self[0].left), 2))
#             for i in range(len(self)):
#                 res[:, 0] += weights[i]*self[i].left.flatten()
#                 res[:, 1] += weights[i]*self[i].right.flatten()
#             out = binauralIR(res , 
#                              samplerate = self.samplerate,
#                              coordinates = (az, el))
#             return out
#         elif method == 'closest':
#             distances = self.distances(az, el)
#             closest = np.argmin(distances)
#             data = np.hstack((self[closest].left, self[closest].right))
#             return binauralIR(data,
#                               samplerate = self.samplerate,
#                               coordinates = (az,el))
#         ## The two following methods are not quite working
#         elif method == 'exponential':
#             # frequency domain, nonlinear interpolation.
#             # say we pick the N closest hrtfs
#             # doesn't quite sound right, except by trimming the end of it
#             # this is possibly due to a sub-sample tf like pb
#             N = 2
#             closests_id = np.argsort(self.distances(az, el))[:N]
#             distances = self.distances(az, el)[closests_id]
#             weights = 1/distances #self.kernel(distances)
#             weights /= np.sum(weights)
#             res = np.ones((len(self[0].left), 2))
#             for i in range(N):
#                 hrtf = self[closests_id[i]]
#                 left_ft = fft(hrtf.left.flatten()) 
#                 right_ft = fft(hrtf.right.flatten()) 
#                 res[:, 0] *= left_ft ** weights[i]
#                 res[:, 1] *= right_ft ** weights[i]
#             data = ifft(res, axis = 0)[:len(res)/2]
#             out = binauralIR(data, 
#                              samplerate = self.samplerate,
#                              coordinates = (az,el)
#                              )
                             
#             return out
#         elif method == 'exponential2':
#             # slightly different method because it also takes care of good ILD interpolation
#             N = 4
#             closests_id = np.argsort(self.distances(az, el))[:N]
#             distances = self.distances(az, el)[closests_id]
#             weights = 1/distances
#             weights /= np.sum(weights)
            
#             amplitudes_right = np.zeros((len(self[0].left), 1))
#             amplitudes_left = np.zeros((len(self[0].left), 1))
#             for i in range(N):
#                 hrtf = self[closests_id[i]]
#                 left_ft = fft(hrtf.left.flatten()) 
#                 right_ft = fft(hrtf.right.flatten()) 
#                 amplitudes_left[:,0] += weights[i]*np.abs(left_ft)
#                 amplitudes_right[:,0] += weights[i]*np.abs(right_ft)
#             res = np.hstack((amplitudes_left, amplitudes_right))
#             for i in range(N):
#                 hrtf = self[closests_id[i]]
#                 left_ft = fft(hrtf.left.flatten()) 
#                 right_ft = fft(hrtf.right.flatten()) 
#                 res[:, 0] *= (left_ft/np.abs(left_ft)) ** weights[i]
#                 res[:, 1] *= (right_ft/np.abs(right_ft)) ** weights[i]

#             data = ifft(res, axis = 0)[:len(res)/2]
#             out = binauralIR(data, 
#                              samplerate = self.samplerate,
#                              coordinates = (az, el))
#             return out


# ####################################################            
# ########## DEPRECATED STUFF ########################
# ####################################################

# class InterpolatingHRTFSet(HRTFSet):
#     '''
#     InterpolatingHRTFSet specialized to the IRCAM (old) db
#     Simple bilinear interpolation.
#     Expects to get all azimuths in (my good) spatializer format, not (their bad) IRCAM
    
#     DEPRECATED, it only works on the horizontal plane with the IRCAM Old DB.
#     '''
#     def __init__(self, hrtfset):
#         super(InterpolatingHRTFSet, self).__init__(hrtfset.data, hrtfset.samplerate, 
#                                                    hrtfset.coordinates)
#         self.dAz = 15.
#         self.dEl = 15.
        
#     def get_hrir(self, az, el, method = 'interpolate'):
#         neighbours = self.closestneighbours(az, el)
#         if len(neighbours)!=4:
#             log_debug('Interpolation problem')
#             log_debug('Was trying to interpolate position'+str((az,el))+', with method '+method)
        
#         left = np.zeros_like(neighbours[0].left)
#         right = np.zeros_like(neighbours[0].right)
#         dAz = self.dAz
#         dEl = self.dEl
#         mAz = np.floor(az/dAz)*dAz
#         cAz = np.mod(az,dAz)/dAz
#         mEl = np.floor(el/dEl)*dEl
#         cEl = np.mod(el,dEl)/dEl
#         coefs = np.array([(1-cAz)*(1-cEl) , cAz*(1-cEl), cAz*cEl, (1-cAz)*cEl])
#         if method == 'closest' or (coefs>.9).any():
#             closest = np.argmax(coefs)
#             return HRTF(neighbours[closest].left,hrir_r = neighbours[closest].right)
            
#         left = interpolate(neighbours[0].left,
#                            neighbours[1].left,
#                            neighbours[2].left,
#                            neighbours[3].left,
#                            cAz, cEl)
        
#         right = interpolate(neighbours[0].right,
#                             neighbours[1].right,
#                             neighbours[2].right,
#                             neighbours[3].right,
#                             cAz,cEl)
        
#         return HRTF(left,hrir_r=right)
    
#     def closestneighbours(self, az, el):
#         az = az_spat2ircam(az)
        
#         if el>=45 or el<-45:
#             print 'WARNING: not good'
        
#         dAz = self.dAz
#         dEl = self.dEl
        
#         mAz=np.floor(az/dAz)*dAz
#         cAz=np.mod(az, dAz)/dAz
#         mEl=np.floor(el/dEl)*dEl
#         cEl=np.mod(el, dEl)/dEl
        
#         neighbours = self.subset(lambda azim,elev: 
#                                  (azim,elev) in [(mAz,mEl),
#                                                  (np.mod(mAz+dAz,360),mEl),
#                                                  (np.mod(mAz+dAz,360),mEl+dEl),
#                                                  (mAz,mEl+dEl)])
#         return neighbours
    
#     @property
#     def compensatedcoordinates(self):
#         '''
#         spits out coordinates for use with the rest of the
#         spatializer, that is arranged in (az,el) value pairs with the
#         correct (mine) azimuth convention
#         '''
#         res = []
#         for (el, az) in self.coordinates:
#             res.append((az_ircam2spat(az), el))
#         return res


# class NewInterpolatingHRTFSet(HRTFSet):
#     '''
#     This is an abandoned attempt at doing interpolation
#     just don't use that method anymore. use distances.
#     '''
#     def __init__(self, hrtfset):
#         super(NewInterpolatingHRTFSet, self).__init__(hrtfset.data, hrtfset.samplerate, 
#                                                       hrtfset.coordinates)
#         self._makecoordinates(hrtfset)
                           
#     def _makecoordinates(self, hrtfset):
#         coordinates = hrtfset.coordinates
#         elevations, _  = zip(*hrtfset.coordinates)
#         elevations = list(set(elevations))
#         elevations.sort()
#         self.elevations = np.array(elevations)
#         azperel = [[]]*len(self.elevations)
#         for i,el in enumerate(self.elevations):
#             tmp = [az_ircam2spat(azim) for (elev, azim) in coordinates if elev == el]
#             tmp.sort()
#             azperel[i] = np.array(tmp, dtype = int)
#         self.azperel = azperel
        
#     def closestneighbours(self, az, el, method = 'bilinear', debug = False):
#         lower_el_id = np.max(np.nonzero(el >= self.elevations))
#         lower_el = self.elevations[lower_el_id]
#         if lower_el == 90:
#             upper_el_id = lower_el_id
#             upper_el = 90
#         else:
#             upper_el_id = np.mod(lower_el_id+1, len(self.elevations))
#             upper_el = self.elevations[upper_el_id]
            

#         if len(self.azperel[upper_el_id]) == len(self.azperel[lower_el_id]):
#             bl_az_id = np.max(np.nonzero(az >= self.azperel[lower_el_id]))
#             left_az = self.azperel[lower_el_id][bl_az_id]

#             right_az = self.azperel[lower_el_id][np.mod(bl_az_id+1, len(self.azperel[lower_el_id]))]
        
#             bl_az, bl_el = left_az, lower_el
#             br_az, br_el = right_az, lower_el
#             tl_az, tl_el = left_az, upper_el
#             tr_az, tr_el = right_az, upper_el
#         elif len(self.azperel[upper_el_id]) > len(self.azperel[lower_el_id]):
#             raise NotImplementedError
#             # cas hemisphere bas
#             np.mod(lower_el_id+1, len(self.elevations))

#             tl_az_id = np.max(np.nonzero(az >= self.azperel[upper_el_id]))
#             tr_az_id = np.mod(tl_az_id + 1, len(self.azperel[upper_el_id]))

#             tl_az = self.azperel[upper_el_id][tl_az_id]
#             tr_az = self.azperel[upper_el_id][tr_az_id]

#             bl_az_id = np.max(np.nonzero(az >= self.azperel[lower_el_id]))
#             bl_az = self.azperel[lower_el_id][bl_az_id]

#             bl_el = lower_el

#             br_az = bl_az
#             br_el = bl_el
#         else:
#             raise NotImplementedError
            
#         out = [(bl_az, bl_el),
#                (tl_az, tl_el),
#                (br_az, br_el),
#                (tr_az, tr_el)]

#         if debug:
#             s = 'Position '+str((az, el))+'\n'
#             s += 'Will be interpolated between '+str(out)
        
#         return out

#     @staticmethod
#     def _coefsfromneighbours(neighbours, az, el):
#         [(bl_az, bl_el),
#          (tl_az, tl_el),
#          (br_az, br_el),
#          (tr_az, tr_el)] = neighbours

#         dAz_lower = abs(br_az - bl_az)
#         dAz_upper = abs(tr_az - tl_az)
#         dAz = float(max((dAz_lower, dAz_upper)))

#         cAz_lower = (az - bl_az)/dAz
#         cAz_upper = (az - tl_az)/dAz

#         dEl_left = abs(tl_el - bl_el)
#         dEl_right = abs(tr_el - br_el)
#         dEl = float(max((dEl_left, dEl_right)))

#         cEl_left = (el - bl_el)/dEl
#         cEl_right = (el - br_el)/dEl
        
#         # coefs are in the order
#         # bl, tl, br, tr
#         c_bl = (1 - cEl_left)*(1 - cAz_lower)
#         c_tl = cEl_left*(1 - cAz_upper)
#         c_br = (1 - cEl_right)*cAz_lower
#         c_tr = cEl_right * cAz_upper
        
#         out = np.array([c_bl, c_tl, c_br, c_tr])
# #        out /= np.sum(out)
#         return out

#     @property
#     def compensatedcoordinates(self):
#         '''
#         spits out coordinates for use with the rest of the
#         spatializer, that is arranged in (az,el) value pairs with the
#         correct (mine) azimuth convention
#         '''
#         res = []
#         for (el, az) in self.coordinates:
#             res.append((az_ircam2spat(az), el))
#         return res

    
#     def exactsubset(self, azel_list):
#         out = []
#         for az, el in azel_list:
#             az = az_spat2ircam(az)
#             tmp = (self.subset(lambda azim,elev: (azim, elev) in [(az,el)]))
#             if len(tmp) == 0:
#                 raise ValueError('(az, el) = ' + str((az,el)) + ' not in HRTFSet')
#             out += tmp
#         return out

#     def get_hrir(self, az, el, method = 'interpolate'):
#         """
#         returns a spatializer ImpulseResponse object
#         """
#         neighbours = self.closestneighbours(az, el)
        
#         hrtfs = self.exactsubset(neighbours)
#         coeficients = self._coefsfromneighbours(neighbours, az, el)
        
#         left = np.zeros_like(hrtfs[0].left)
#         right = np.zeros_like(hrtfs[0].right)
        
#         if method == 'closest' or (coeficients > .9).any():
#             closest = np.argmax(coeficients)
#             coeficients = np.zeros(4)
#             coeficients[closest] = 1

#         out = self._interpolate(hrtfs, coeficients)
#         return binauralIR(out, samplerate = hrtfs[0].samplerate, coordinates = (az,el))
    
#     def get_HRTF(self, az, el, method = 'interpolate'):
#         """
#         returns a Brian HRTF object
#         """
#         ir = self.get_hrir(az, el, method = method)
#         return HRTF(ir[:,0], hrir_r = ir[:,1])
    
#     @staticmethod
#     def _interpolate(hs, cs):
#         left = (cs[0] * hs[0].left + cs[1] * hs[1].left 
#                 + cs[2] * hs[2].left + cs[3] * hs[3].left).reshape((len(hs[0].left),1))
#         right = (cs[0] * hs[0].right + cs[1] * hs[1].right 
#                 + cs[2] * hs[2].right + cs[3] * hs[3].right).reshape((len(hs[0].right),1))
#         return np.hstack((left,right))


# ############# Proximity ############################

# def closestneighbors(az,el,coords,n=4):
#     ref=vectfromspherical(deg2rad(az),deg2rad(el))
#     distances=zeros(len(coords))

#     for (i,(el,az)) in enumerate(coords):
#         v=vectfromspherical(deg2rad(az_spat2ircam(az)),deg2rad(el))
#         distances[i]=(v-ref).norm()
    
#     neighbors=[]
#     neighborsdist=[]
#     for i in xrange(n):
#         id=np.argmin(distances)
#         mi=distances[id]
#         neighbors.append(np.argmin(distances))
#         neighborsdist.append(mi)
#         distances[neighbors[-1]] = np.inf
#     return neighbors,neighborsdist

# def closestneighbors_IRCAM(az, el, hrtfset):
#     az=az_spat2ircam(az)
#     if el>=45 or el<-45:
#         print 'WARNING: not good'
#     dAz=15.
#     dEl=15.
#     mAz=np.floor(az/dAz)*dAz
#     cAz=np.mod(az,dAz)/dAz
#     mEl=np.floor(el/dEl)*dEl
#     cEl=np.mod(el,dEl)/dEl
#     hrtfs=hrtfset.subset(lambda azim,elev: (azim,elev) in [(mAz,mEl),
#                                                            (np.mod(mAz+dAz,360),mEl),
#                                                            (np.mod(mAz+dAz,360),mEl+dEl),
#                                                            (mAz,mEl+dEl)])
#     return hrtfs

# # def interpolate(h0,c0,h1,c1,h2,c2,h3,c3):
# #     return h0*c0+h1*c1+h2*c2+h3*c3

# ########### Interpolation

# def interpolate(h0,h1,h2,h3,cAz,cEl):
#     return (1-cAz)*(1-cEl)*h0 + cAz*(1-cEl)*h1 + cAz*cEl*h2 + (1-cAz)*cEl*h3

# def interpolate_IRCAM(az,el,hrtfset,n=4,channel='left'):
#     az=az_spat2ircam(az)
#     if el>=45 or el<-45:
#         print 'WARNING: not good'
#     dAz=15.
#     dEl=15.
#     mAz=np.floor(az/dAz)*dAz
#     cAz=np.mod(az,dAz)/dAz
#     mEl=np.floor(el/dEl)*dEl
#     cEl=np.mod(el,dEl)/dEl
#     #bilinear method for HRTF interpolation
#     hrtfs=closestneighbors_IRCAM(az,el,hrtfset)
#     if len(hrtfs)!=4:
#         'PROBLEM, everyday is problem'
#         print 'was trying to interpolate position ',(az,el)
#     res=np.zeros_like(hrtfs[0].left)
#     if channel=='left':

#         return interpolate(hrtfs[0].left,
#                            hrtfs[1].left,
#                            hrtfs[2].left,
#                            hrtfs[3].left,cAz,cEl)
#     return res

# def coeficients_from_neighborspos(az,el,neighbors,hrtfset):
#     positions=zeros((4,2))
#     for i in range(4):
#         azim,elev=hrtfset.coordinates[neighbors[i]]
#         positions[i,0]=azim
#         positions[i,1]=elev
#     maxazim=np.nonzero(positions[:,0]==np.amax(positions[:,0]))
#     minazim=np.argmin(positions[:,0])
#     maxelev=np.argmax(positions[:,1])
#     minelev=np.argmin(positions[:,1])
#     azimstep=positions[maxazim,0]-positions[minazim,0]
#     elevstep=positions[maxelev,1]-positions[minelev,1]
#     cazim,celev=np.mod(az,azimstep)/azimstep,np.mod(el,elevstep)/elevstep
    

# def coeficients_from_distance(distances):
#     coefs=zeros(4)
#     for i in range(4):
#         coefs[i]=exp(-distances[i])
#     coefs/=sum(coefs)

# def hrtfinterpolate(hrtfset,az,el,method='bilinear'):
#     if method=='bilinear':
#         neighbors,neighborsdist=closestneighbors(az,el,hrtfset.coordinates)
#         for n in neighbors:
#             print hrtfset.coordinates[n]
#         ctheta,cphi=coeficients_from_neighborspos(az,el,neighbors,hrtfset)

# def correct_coords(coordinates):
#     res = []
#     for (az,el) in coordinates:
#         res.append((az_spat2ircam(az), el))
#     return res


# if __name__=='__main__':
#     db='IRCAM'
#     subject=1029
#     db_pth='/home/victor/Work/Data/HRTF/IRCAM'

#     hrtfdb=IRCAM_LISTEN(db_pth)
    
#     hrtfset=hrtfdb.load_subject(subject)
    
#     hset=NewNewInterpolatingHRTFSet(hrtfset)
    
#     sound=pinknoise(500*ms)

#     el=30
#     for az in linspace(-90, 90, 10):
#         h=hset.get_hrtf(az,el)
#         h.listen(sound = sound, sleep = True)
