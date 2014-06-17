
class NaturalGroundScene(SimpleScene):
    '''
    A specialized SimpleScene that handles the Miki model (because it
    doesn't fit in usual model architecture). 
    
    ** Initialization ** 
    
    NaturalGroundScene(sigma = 2e6)
    Or, 
    NaturalGroundScene(model = NaturalGroundModel(2e6))
    
    ** Keywords ** 
    
    with_delays : should be set to false when using an HRTFReceiver with HRTFs that depend on distance, in which case the additional path length is already in the data. WARNING: It's not the case with Makoto's HRIRs apparently, but the attenuation is there already though.
    

        
    '''
    def __init__(self, sigma = 2e6, model = None, with_delays = True):
        if (model is None):
            self.sigma = sigma
            model = NaturalGroundModel(sigma)
        else:
            self.sigma = model.sigma
        
        self.with_delays = with_delays
            
        super(NaturalGroundScene, self).__init__(model = model)

    def _computeTFs(self, beam, nfft, **kwdargs):
        if 'samplerate' in kwdargs:
            samplerate = kwdargs['samplerate']
        else:
            samplerate = get_pref('DEFAULT_SAMPLERATE', default = 44100*Hz)
            log_debug('Samplerate automatically detected, '+str(samplerate))
        samplerate = float(samplerate)
        
        #This is subclassed here because the Miki model depends on the ratio of total dist and direct dist
        if not isinstance(self.ground.model, NaturalGroundModel):
            raise AttributeError('This NaturalGroundScene doesn\'t have the right model')
        if not (beam is None):
            TFs = np.ones((nfft, beam.nrays), dtype = complex)
            alltotaldists = beam.get_totaldists()

            #sourcestotaldists = alltotaldists[::2] + alltotaldists[1::2]# why would one do that?
            sourcestotaldists = alltotaldists[1::2]
            
            self.ground.model.prepare(samplerate, int(nfft))

            sirs = self.ground.model.compute(sourcestotaldists, deg2rad(beam.incidences[1::2]))
            
            TFs[:,1::2] = sirs.reshape((nfft, beam.nrays/2)) # reflected rays
            TFs[:,0::2] = 1. # direct rays

            if self.with_delays:            
                def make_delay(delays, nfft, attenuations, samplerate):
                    nchannels = len(delays)
                    freqs = fftfreq(nfft) * samplerate
                    freqs = np.tile(freqs.reshape((nfft, 1)), (1, nchannels))
                    delays = np.tile(delays.reshape((1, nchannels)), (nfft, 1))
                    attenuations = np.tile(attenuations.reshape((1, nchannels)), (nfft, 1))
                    delayf = attenuations*exp(-2j*np.pi*freqs*delays)
                    return delayf


                delays = make_delay(alltotaldists/c,
                                    TFs.shape[0],
                                    1/alltotaldists,
                                    self.ground.model.samplerate)

                if False:
                    print 'are dists nan?', np.isnan(alltotaldists).any()
                    print 'are dists zero?', (alltotaldists==0).any()
                    print 'are delays nan?', np.isnan(delays).any()

                TFs *= delays
            
            if not np.isfinite(TFs).all():
                log_debug('Warning, problem, model res are not finite')

            return TFs
        else:
            return 1

    def volume(self):
        raise AttributeError('Volume is infinite for a Natural Ground Scene')

class AcousticScene(GeometricScene):
    '''
    This class is a specialized GeometricScene that handles the
    acoustical computations.


    ** Initialization **
    
    ``model = None`` Defines a model for all the surfaces added at
    initialization.
    

    ** Model Handling ** 

    .. automethod :: set_globalmodel
    
    ** Acoustical Computations **

    .. automethod :: computeTF
    .. automethod :: computeTFs

    .. automethod :: computeIR
    .. automethod :: computeIRs
    
    '''
    def __init__(self, *args, **kwdargs):
        super(AcousticScene, self).__init__(*args, **kwdargs)

        if 'model' in kwdargs and not isinstance(self, VoidScene):
            self.set_globalmodel(kwdargs['model'])
            

    ###################### MODEL HANDLING ######################
    def set_globalmodel(self, model, include_sources = True):
        '''
        Method to set a global model for all the surfaces of the scene.
        the Keyword argument ``include_sources`` is set to True by default, if switched to false then only the non-sources surfaces will see their model changed.
        Returns the number of changed models
        '''
        if not (model == None):
            i = 0
            for obj in self.surfaces:
                if include_sources or (not (obj in self.sources)): 
                    obj.model = model
                    i += 1
            return i

    def preparemodels(self, samplerate, nfft):
        # prepares all the models of all the surfaces
        for surf in self.surfaces:
            if not surf.model is None:
                surf.model.prepare(samplerate, nfft)
    
    ###################### ACOUSTIC COMPUTATIONS ######################
    def _computeTFs(self, beam, nfft, debug = False):
        # This does all the actual computation on (possibly chunks of) beams.
        # It walks into the beam and applies the functions of the models to fetch transfer functions
        
        if not (beam is None):
            TFs = np.ones((nfft, beam.nrays), dtype = complex)
            for i,surface in enumerate(self.surfaces):
                subbeam_idx = np.nonzero((beam.surfacehit == i) * -beam.sourcereached)[0]
                if len(subbeam_idx) != 0:
                    if not (surface.model is None):
                        nargs = surface.model.nargs
                        if nargs == 1:
                            modelres = surface.model.compute(beam.distances[subbeam_idx])
                        elif nargs == 2:
                            modelres = surface.model.compute(beam.distances[subbeam_idx],
                                                             beam.incidences[subbeam_idx])
                        if False:
                            print 'surface ', i
                            print 'ray', subbeam_idx
                            print 'dist', beam.distances[subbeam_idx]
                            print 'delay', beam.distances[subbeam_idx]/342.*1000
                        
                        if modelres.ndim == 1:
                            modelres.shape = (len(modelres),1)
                        if not np.isfinite(modelres).all():
                            log_debug('Warning, problem, model res are not finite')
                            modelres = 1
                    else:
                        if not nomodel:
                            log_debug('Surface has no model')
                        nomodel = True
                        modelres = 1
                    TFs[:,subbeam_idx] = modelres
            sir = TFs * self._computeTFs(beam.next, nfft) # recursive call
            return sir
        else:
            return 1
        
    def computeTFs(self, obj, **kwdargs):
        '''
        Computes the transfer function of all the sources of the
        AcousticScene separately. 
        
        Input argument can be either a Beam object, result of the
        geometric rendering in GeometricScene.render, or a Receiver
        object, in which case GeometricScene.render is called on it first.
    
        ``samplerate = DEFAULT`` Defines the samplerate for which to do the
        computations. If it is not set, then the default samplerate is picked 
        
        ``nfft = None`` Defines for how many points to do the
        computations. If it is not set, then the nfft is automatically
        detected by looking at the longest delays in the rays.
        
        ``split = 1`` Whether or not to split the computations. Since
        frequency domain computations require the storage of lots of
        data, it often reaches the memory bound of the computer. To
        take care of that one can choose to split the acoustic
        computations in different steps, and aggregate the result for
        each source. This implies collapse = True, and it is turned on
        if the number of rays is bigger than 500.
        
        ``collapse = True`` If this flag is set to false, then the
        TransferFunction will contain the transfer functions computed
        separately for each ray.
        '''
        # This methods takes care of 
        # - the indexing in the
        # ImpulseResponse (sources, coordinates),  
        # - gathering rays that come from different
        # sources
        # - equalizing different rays
        # - (almost more importantly) takes care of
        # splitting the data to avoid memory limitations
        
        # KWDARGS parsing
        
        binaural = kwdargs.get('binaural', False)        
        collapse = kwdargs.get('collapse', False)

        split = kwdargs.get('split', 1)
        
        if 'samplerate' in kwdargs:
            samplerate = kwdargs['samplerate']
        else:
            samplerate = get_pref('DEFAULT_SAMPLERATE', default = 44100*Hz)
            log_debug('Samplerate automatically detected, '+str(samplerate))
        samplerate = float(samplerate)
        
        # ARGS 
        if isinstance(obj, Receiver):
            # if it is a receiver, we render a beam...
            receiver = obj
            beam = self.render(receiver)
            if binaural != obj.binaural:
                log_debug("""Conflicting ``binaural`` flags, taking
        the objects""")
            binaural = obj.binaural
        elif isinstance(obj, Beam):
            receiver = None
            beam = obj
        else:
            raise ValueError('Bad argument for computeIR, type '+str(type(obj)))
        
        #### Computation
        ## We gather the rays that reached their source
        reached_source = beam[beam.get_reachedsource_index()]
        nrays = reached_source.nrays
        
        if nrays == 0:
            raise ValueError('No rays reached the source')

        ## At this point we are able to guess the nfft from the
        ## longest path
        if 'nfft' in kwdargs:
            nfft = kwdargs['nfft']
        else:
            nfft = beam.detect_nfft(samplerate)
            log_debug('Nfft automatically detected, '+str(nfft))

        ### Model application preparation
        # preparing models
            
        print 'scene', samplerate
        self.preparemodels(samplerate, nfft)

        ### Resulting IR preparation:
        # target source,
        target_source = reached_source.target_source
        # spherical coords
        incoming_direction = np.zeros(target_source.shape, 
                                      dtype = [('azim','f8'),('elev','f8')])
        d, az, el = cartesian2spherical(reached_source.directions, unit = 'deg')
        # compensate for receiver orientation if it is oriented
        if not receiver is None and isinstance(receiver, OrientedReceiver):
            (_, azref, elref) = receiver.orientation.toSphericalCoords(unit = 'deg')
            az -= azref
            el -= elref 
        # ... and instantiate
        incoming_direction['azim'] = az
        incoming_direction['elev'] = el


        ### Equalization
        ## Here we prepare the weight vector, containing one weight
        ## per ray (even if the receiver is not equalized)
        weights = np.ones(nrays)
        for k in range(self.nsources):
            # constructing the array for each source
            source = self.sources[k]
            if isinstance(source, EqualizedSource):
                # if the source is equalized, fetch the weights
                finalbeam = beam.get_finalbeam()
                curweights = self.sources[k].get_equalization(reached_source.forsource(source.id))
                weights[target_source == source.id] = curweights
            else:
                # if it is not the weights are just 1/nrays
                curweights = 1./nrays
                weights[target_source == source.id] = curweights

        if not receiver is None:
            # we don't need it anymore
            del beam
            
        ### Data management remarks:
        # Computations are dense in frequency so it imposes a huge
        # load on the memory, to cope with that, I split the
        # computations in chunks that are processed sequentially.
        # It is only possible when collapse == True, otherwise the
        # individual irs for rays must be kept and yielded to the 
        # user. I may implement some writing to file thing in the future to handle this

        # So... 
        # impose splitting if there are too many rays*nfft. 
        # I think 5*10**6 is good enough, it allows for 5000 channels
        # to be processed at the same time @nfft = 1024, let's see if it holds!
        if (split != 1 or nrays*nfft > MAX_ARRAY_SIZE) and collapse:
            # compute the number of chunks, called split
            if split ==1:
                split = np.ceil(nrays*nfft / float(MAX_ARRAY_SIZE))

            chunksize = nrays/split
            chunks = np.floor(np.arange(nrays)/chunksize)
        else:
            chunks = np.zeros(nrays)
            
        #### Data array instantiation
        if not collapse:
            # in this case there is one tf per ray
            TFs = onesTF((nfft, nrays), 
                         samplerate = samplerate,
                         target_source = target_source, 
                         binaural = binaural)
        else:
            try:
                TFs = onesTF((nfft, self.nsources), 
                             samplerate = samplerate,
                             target_source = np.unique(target_source), 
                             binaural = binaural)
            except ValueError:
                raise ValueError("""There are too many rays to compute all the transfer functions. 
Consider only computing one, or splitting the computations""")

        ### Computation per se
        log_debug('Starting acoustical computations on '+str(split)+' chunks')
        try:
            step = nrays/split
            for i in range(int(split)):
                log_debug('Computing chunk '+str(i))
                # for all rays concerned by this chunk
                cur_chunk = (chunks == i)
                # get the weights
                cur_weights = weights[cur_chunk]
                # apply the model, and the source equalization
                
                ########## HERE BE A UGLY HACK
                if not isinstance(self, NaturalGroundScene):
                    # special case, requires samplerate, todo: remove that!
                    cur_TF = self._computeTFs(reached_source[cur_chunk], 
                                              nfft, samplerate) * cur_weights
                else:
                    cur_TF = self._computeTFs(reached_source[cur_chunk], nfft) * .5
                # make this nd array a tf (simpler for later)
                cur_TF = TransferFunction(cur_TF, 
                                          samplerate = samplerate,
                                          target_source = reached_source[cur_chunk].target_source)
                if (cur_TF == 0).any():
                    print 'some zeros in cur_TF'
                    
                # collapsing
                if not collapse:
                    # just place the chunk of TF where it belongs
                    TFs[:, cur_chunk] = cur_TF
                else:
                    # here, take care of the collapsing, sourcewise
                    if True:
                        # UGLY HACK
                        IR = ImpulseResponse(cur_TF)
                        ir = IR.collapse()
                        collapsed = TransferFunction(ir)
                    else:
                        collapsed = cur_TF.collapse()
                        
                    if (collapsed == 0).all():
                        log_debug('Collapsed result is zero')
                    for source in np.unique(cur_TF.target_source):
                        TFs[:, TFs.target_source == source] *= collapsed.forsource(source)
        
        except MemoryError:
            log_debug('MemoryError caught, you should split the computations or compute a single TF')
        
        print 'outscene', TFs.samplerate
        return TFs
    
    def computeTF(self, *args, **kwdargs):
        '''
        Computes a single transfer function for each source in the
        scene.
        '''
        kwdargs['collapse'] = True
        return self.computeTFs(*args, **kwdargs)

    # Time domain counterparts
    def computeIRs(self, *args, **kwdargs):
        '''
        Time-domain counterpart of computeTFs
        '''

        irs = ImpulseResponse(self.computeTFs(*args, **kwdargs))
        return irs
        
    def computeIR(self, *args, **kwdargs):
        '''
        Time-domain counterpart of computeTF
        '''
        ir = ImpulseResponse(self.computeTF(*args, **kwdargs))
        return ir

