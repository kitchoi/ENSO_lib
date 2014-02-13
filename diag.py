def findENSO_percentile(index,percentile):
    ''' percentile < 50.
    warm,cold = findENSO_percentile(nino34,15.)
    warm and cold are dictionaries containing the locations and peak values of the events
    '''
    import numpy
    if percentile > 50.: percentile = 100. - percentile
    warm = {}
    cold = {}
    if not isinstance(index,numpy.ma.core.MaskedArray):
        index = numpy.ma.array(index)
    warm['locs'],warm['peaks'] = findEvents(index,'>',numpy.percentile(index[index.mask==0],100-percentile))
    cold['locs'],cold['peaks'] = findEvents(index,'<',numpy.percentile(index[index.mask==0],percentile))
    return warm,cold


def findEvents(index,operator,threshold,per=5,window=[-3,3]):
    ''' Return the locations of ENSO warm/cold event peaks for a given index.
    Inputs:
    index       - numpy.ndarray (e.g. Nino3.4 index)
    operator    - string ('>' or '<')
    threshold   - threshold for event definition
    per         - minimum persistence for index >/< threshold (default=5, unit is consistent with the array grid)
    window      - requires the peak to be a global minima/maxima within the window around the peak (default=[-3,3])
    
    Outputs:
    pklocs      - location in the input array
    pks         - values of extrema
    '''
    import numpy
    import util.stat as stat
    
    if operator == '>':
        argpeak_op = numpy.argmax
        comp_op = numpy.greater
        peak_op = numpy.max
    elif operator == '<':
        argpeak_op = numpy.argmin
        comp_op = numpy.less
        peak_op = numpy.min
    else:
        raise Exception('operator has to be either > or <')
    
    locs = numpy.where(comp_op(index,threshold))[0]
    jumps = numpy.where(numpy.diff(locs)>1)[0]
    starts = numpy.insert(locs[jumps+1],0,locs[0])
    ends = numpy.append(locs[jumps],locs[-1])
    
    # Ignore the chunks that starts from the beginning or ends at the end of the index
    if starts[0] == 0:
        starts  = starts[1:]
        ends    = ends[1:]
    if ends[-1] == len(index)-1:
        starts  = starts[:-1]
        ends    = ends[:-1]
    
    # Chunks of the index that exceed the threshold
    subsets = [ index[starts[i]:ends[i]] for i in range(len(starts)) ]
    
    # Persistence check
    pklocs = [ starts[i]+argpeak_op(subsets[i]) for i in range(len(subsets)) if len(subsets[i]) >= per ]
    
    # Check for being global extrema within the window
    pklocs = [ loc for loc in pklocs if index[loc] == peak_op(index[numpy.max([0,loc+window[0]]):numpy.min([len(index)-1,loc+window[1]])]) ]
    
    pklocs = [ int(loc) for loc in pklocs if loc != False ]
    pks = numpy.array([ index[loc].squeeze() for loc in pklocs ])
    
    return pklocs,pks


def find_EN_pattern(field,nino34,nino34_mid=0.8,nino34_tole=0.4,verbose=False):
    ''' Given a field and Nino3.4 index time series, extract
    the time at which nino34_mid-nino34_tole < nino34 < nino34_mid+nino34_tole
    Then compute the climatology for these snap shots
    Input:
    field  - util.nc.Variable
    nino34 - util.nc.Variable
    nino34_mid - mid point
    nino34_tole - half bin size
    '''
    import numpy
    import util
    import warnings
    warm,cold = findENSO_percentile(nino34.data,49.)
    locs = warm['locs'] + cold['locs']
    peaks = numpy.append(warm['peaks'],cold['peaks'])
    locs = numpy.array(locs)[ numpy.abs(peaks - nino34_mid)
                                < nino34_tole]
    if len(locs) == 0: 
        result = util.nc.Variable(data=numpy.ma.ones(field[0].data.shape),
                                  parent=field[0])
        result.data[:] = numpy.ma.masked
        warnings.warn("No event found!")
        return result
    pattern = util.nc.climatology(field[locs])
    pattern.setattr('event_loc',locs.squeeze())
    if verbose: print 'Nino 3.4: '+ str(nino34[locs].time_ave().squeeze().data)
    return pattern

def ENSO_duration(nino34,percentile,thres_std_fraction):
    import numpy
    warm,cold = findENSO_percentile(nino34,percentile)
    warm_end = nino34.std()*thres_std_fraction
    cold_end = warm_end*-1
    duration = {}
    
    def compute_duration(operator,locs,evt_end):
        lengths = []
        for iloc in xrange(len(locs)):
            loc = locs[iloc]
            after_end = operator(nino34[loc:],evt_end)
            if after_end.any():
                length =  numpy.where(after_end)[0][0]
                if iloc < len(locs)-1:
                    # Avoid double counting events that reintensify
                    if locs[iloc+1]-locs[iloc] > length:
                        lengths.append(length)
        return lengths
    
    duration['warm'] = compute_duration(numpy.less,warm['locs'],warm_end)
    duration['cold'] = compute_duration(numpy.greater,cold['locs'],cold_end)
    return duration

def ENSO_transition(nino34,percentile,wait_window,per=5,window=[-3,3]):
    import numpy
    warm,cold = findENSO_percentile(nino34,percentile)
    events = ['warm',]*len(warm['locs'])+['cold',]*len(cold['locs'])
    events_locs = sorted(zip(events,warm['locs']+cold['locs']),key=lambda v:v[1])
    events = [ evt for (evt,loc) in events_locs ]
    locs = [ loc for (evt,loc) in events_locs ]
    transition = {}
    for iloc in xrange(len(events)-1):
        evt_pair = '_'.join(events[iloc:iloc+2])
        if locs[iloc+1]-locs[iloc] <= wait_window:
            transition[evt_pair] = transition.setdefault(evt_pair,0) + 1
    transition['warm'] = len(warm['locs'])
    transition['cold'] = len(cold['locs'])
    return transition

def ENSO_skewness(nino34):
    import util.stat
    return util.stat.skewness(nino34)

def Threshold2Composite(field,nino34,warm_thres,cold_thres):
    '''Threshold only '''
    import numpy
    import util
    data = field[numpy.logical_or(nino34.data>warm_thres,nino34.data<cold_thres)]
    time = field.getTime()[numpy.logical_or(nino34.data>warm_thres,nino34.data<cold_thres)]
    itime = data.getCAxes().index('T')
    data.dims[itime].data = time
    return data
