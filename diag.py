def findENSO_percentile(index,percentile):
    ''' percentile < 50.
    warm,cold = findENSO_percentile(nino34,15.)
    warm and cold are dictionaries containing the locations and peak values of the events
    '''
    import numpy
    if percentile > 50.: percentile = 100. - percentile
    warm = {}
    cold = {}
    warm['locs'],warm['peaks'] = findEvents(index,'>',numpy.percentile(index,100-percentile))
    cold['locs'],cold['peaks'] = findEvents(index,'<',numpy.percentile(index,percentile))
    return warm,cold


def findEvents(index,operator,threshold,per=5,window=[-3,3]):
    ''' Return the locations of ENSO warm/cold event peaks for a given index.
    Inputs:
    index       - numpy.ndarray (e.g. Nino3.4 index)
    operator    - string ('>' or '<')
    threshold   - threshold for event definition
    per         - minimum persistence for index >/< threshold (default=5)
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

    # Find pattern for weak El Nino
def find_EN_pattern(field,nino34,nino34_mid=0.8,nino34_tole=0.4):
    ''' Given a field and Nino3.4 index time series, extract
    the time at which nino34_mid-nino34_tole < nino34 < nino34_mid+nino34_tole
    Then compute the climatology for these snap shots
    '''
    import numpy
    import util
    warm,cold = findENSO_percentile(nino34.data,49.)
    locs = warm['locs'] + cold['locs']
    peaks = numpy.append(warm['peaks'],cold['peaks'])
    locs = numpy.array(locs)[ numpy.abs(peaks - nino34_mid)
                                < nino34_tole]
    if len(locs) == 0: 
        result = util.nc.Variable(data=numpy.ma.ones(field[0].data.shape),
                                  parent=field[0])
        result.data[:] = numpy.ma.masked
        return result
    pattern = util.nc.climatology(field[locs])
    print 'Nino 3.4: '+ str(nino34[locs].time_ave().squeeze().data)
    return pattern
