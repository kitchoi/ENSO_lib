import numpy
import warnings
import itertools

import util.stat as stat
import util


def findENSO_percentile(index,percentile,*args,**kwargs):
    ''' percentile < 50.
    warm,cold = findENSO_percentile(nino34,15.)
    warm and cold are dictionaries containing the locations and peak values of the events
    '''
    if percentile > 50.: percentile = 100. - percentile
    if percentile < 0.:
        raise ValueError("percentile cannot be smaller than zero.")
    if percentile > 100.:
        raise ValueError("percentile cannot be bigger than 100.")
    
    warm = {}
    cold = {}
    if numpy.ma.isMaskedArray(index):
        if index.mask.any():
            warm['locs'],warm['peaks'] = findEvents(index,'>',numpy.percentile(index[~index.mask],100-percentile),*args,**kwargs)
            cold['locs'],cold['peaks'] = findEvents(index,'<',numpy.percentile(index[~index.mask],percentile),*args,**kwargs)
            return warm,cold
    warm['locs'],warm['peaks'] = findEvents(index,'>',numpy.percentile(index,100-percentile),*args,**kwargs)
    cold['locs'],cold['peaks'] = findEvents(index,'<',numpy.percentile(index,percentile),*args,**kwargs)
    return warm,cold


def findENSO_threshold(index,warm_threshold,cold_threshold):
    ''' 
    Using threshold to find ENSO events
    Return: (warm,cold)
    warm and cold are dictionaries containing the locations and peak values of the events
    '''
    warm = {}
    cold = {}
    if not isinstance(index,numpy.ma.core.MaskedArray):
        index = numpy.ma.array(index)
    warm['locs'],warm['peaks'] = findEvents(index,'>',warm_threshold)
    cold['locs'],cold['peaks'] = findEvents(index,'<',cold_threshold)
    return warm,cold


def Persistence_check(is_selected,per):
    ''' Return a numpy boolean array of the same shape of is_selected
    where True indicating an active, persistent event
    is_selected- a numpy boolean array
    per     - persistence required
    '''
    assert len(is_selected) > 0
    assert is_selected.dtype == numpy.bool
    if not numpy.any(is_selected):
        return is_selected
    is_filtered = numpy.zeros_like(is_selected,dtype=numpy.bool)
    in_event = 0
    n_select = 0
    i_start = 0
    for i,selected in enumerate(is_selected):
        if selected:
            n_select += 1
            if not in_event:
                # going from normal to selected
                i_start = i
            in_event = 1
        else:
            if in_event and n_select >= per:
                is_filtered[i_start:i-1] = True
            i_start = i
            n_select = 0
            in_event = 0
    return is_filtered


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
    if len(locs) <= 1:
        return ([],numpy.array([]))
    
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


def find_EN_pattern(field,nino34,nino34_mid=0.8,nino34_tole=0.4,
                    do_climo=True,verbose=False):
    ''' Given a field and Nino3.4 index time series, extract
    the time at which nino34_mid-nino34_tole < peak nino34 < nino34_mid+nino34_tole
    Then compute the climatology for these snap shots
    Input:
    field  - util.nc.Variable
    nino34 - util.nc.Variable
    nino34_mid - mid point
    nino34_tole - half bin size
    
    Output:
    pattern - util.nc.Variable (a climatology)
    '''
    if numpy.any(field.dims[0].data != nino34.dims[0].data):
        raise Exception("Expect the time record of nino3.4 and the field to be the same")
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
    pattern = field[locs]
    if do_climo:
        pattern = util.nc.climatology(pattern)
    pattern.setattr('event_loc',locs.squeeze())
    if verbose: print 'Nino 3.4: '+ str(nino34[locs].time_ave().squeeze().data)
    return pattern


def compute_duration(nino34,operator,locs,evt_end,remove_merge_event=True):
        lengths = []
        for iloc in xrange(len(locs)):
            loc = locs[iloc]
            after_end = operator(nino34[loc:],evt_end)
            if after_end.any():
                length =  numpy.where(after_end)[0][0]
                if remove_merge_event:
                    if iloc < len(locs)-1:  # if this is not the last event peak
                        if locs[iloc+1]-locs[iloc] > length:
                             lengths.append(length) # Only count the duration 
                             # if the next event occurs after
                             # the termination
                else:
                    lengths.append(length)
        return lengths
    


def ENSO_duration(nino34,percentile,thres_std_fraction,per=5,window=[-3,3]):
    ''' compute the duration of the warm and cold events
    Input:
    nino34             - numpy 1D array
    percentile         - 0. < percentile < 100.
    thres_std_fraction - the fraction times the nino34 standard deviation 
                         is used for defining the termination
    
    Return : A dictionary with keys "warm" and "cold"
             each contains a list of integers which
             are the duration of the events
    '''
    warm,cold = findENSO_percentile(nino34,percentile,per,window)
    warm_end = nino34.std()*thres_std_fraction
    cold_end = warm_end*-1
    duration = {}
    duration['warm'] = compute_duration(nino34,numpy.less,
                                        warm['locs'],warm_end)
    duration['cold'] = compute_duration(nino34,numpy.greater,
                                        cold['locs'],cold_end)
    duration['warm_locs'] = warm['locs']
    duration['cold_locs'] = cold['locs']
    # Length of duration['warm'] may not match the length of 
    # duration['warm_locs'] as double counting is taken care of
    return duration


def ENSO_transition2(nino34,percentile,wait_window,thres_std_fraction,
                     per=5,window=[-3,3]):
    ''' Word for word copy from the Matlab code
    Compute the transition probabilities
    TODO: should be made more elegant
    '''
    ENLNind = []
    LNENind = []
    ENENind = []
    LNLNind = []
    warm,cold = findENSO_percentile(nino34,percentile,per,window)
    warm['locs'] = numpy.array(sorted(warm['locs']))
    cold['locs'] = numpy.array(sorted(cold['locs']))
    warm_end = nino34.std()*thres_std_fraction
    cold_end = warm_end*-1
    for i,iwarm in enumerate(warm['locs']):
        ended = numpy.where(nino34[iwarm:] < warm_end)[0]
        if len(ended):
            # The index where it terminates
            iterm = iwarm + ended[0]
        nextcold = numpy.where(cold['locs']>iwarm)[0]
        nextwarm = numpy.where(warm['locs']>iwarm)[0]
        if len(nextcold) and len(nextwarm):
            inextcold = cold['locs'][nextcold[0]]
            inextwarm = warm['locs'][nextwarm[0]]
            if (inextcold - iterm) <= wait_window:
                ENLNind.append(iwarm)
            elif (inextwarm - iterm) <= wait_window:
                ENENind.append(iwarm)
        elif len(nextcold):
            if (cold['locs'][nextcold[0]] - iterm) <= wait_window:
                ENLNind.append(iwarm)
        elif len(nextwarm):
            if (warm['locs'][nextwarm[0]] - iterm) <= wait_window:
                ENENind.append(iwarm)
    for i,icold in enumerate(cold['locs']):
        ended = numpy.where(nino34[icold:] > cold_end)[0]
        if len(ended):
            # The index where it terminates
            iterm = icold + ended[0]
        nextwarm = numpy.where(warm['locs']>icold)[0]
        nextcold = numpy.where(cold['locs']>icold)[0]
        if len(nextwarm) and len(nextcold):
            inextwarm = warm['locs'][nextwarm[0]]
            inextcold = cold['locs'][nextcold[0]]
            if (inextwarm - iterm) <= wait_window:
                LNENind.append(icold)
            elif (inextcold - iterm) <= wait_window:
                LNLNind.append(icold)
        elif len(nextwarm):
            if (warm['locs'][nextwarm[0]] - iterm) <= wait_window:
                LNENind.append(icold)
        elif len(nextcold):
            if (cold['locs'][nextcold[0]] - iterm) <= wait_window:
                LNLNind.append(icold)
    nEN = len(warm['locs'])
    nLN = len(cold['locs'])
    if nEN and nLN:
        if warm['locs'][-1] > cold['locs'][-1]:
            # warm event the last
            nEN -= 1
        else:
            nLN -= 1
    elif nEN:
        nEN -= 1
    elif nLN:
        nLN -= 1
    transition = {}
    if nEN:
        transition['warm_cold'] = float(len(ENLNind))/nEN
        transition['warm_warm'] = float(len(ENENind))/nEN
    if nLN:
        transition['cold_warm'] = float(len(LNENind))/nLN
        transition['cold_cold'] = float(len(LNLNind))/nLN
    return transition,dict(ENLN=ENLNind,ENEN=ENENind,
                           LNEN=LNENind,LNLN=LNLNind)



def ENSO_transition(nino34,percentile,wait_window,thres_std_fraction,
                    per=5,window=[-3,3]):
    warm,cold = findENSO_percentile(nino34,percentile,per,window)
    warm_end = nino34.std()*thres_std_fraction
    cold_end = warm_end*-1
    durations = {'warm': compute_duration(nino34,numpy.less,
                                          warm['locs'],warm_end,False),
                 'cold': compute_duration(nino34,numpy.greater,
                                          cold['locs'],cold_end,False),}
    events = ['warm',]*len(warm['locs'])+\
             ['cold',]*len(cold['locs'])
    events,locs,termT = zip(*sorted(
        zip(events,
            warm['locs']+cold['locs'],
            durations['warm']+durations['cold']),
        key=lambda v:v[1]))
    transition = {}
    for iloc in xrange(len(events)-1):
        evt_pair = '_'.join(events[iloc:iloc+2])
        if (locs[iloc+1] - locs[iloc] - termT[iloc]) <= wait_window:
            transition[evt_pair] = transition.setdefault(evt_pair,0) + 1
    # warm,cold = findENSO_percentile(nino34,percentile,per,window)
    # events = ['warm',]*len(warm['locs'])+['cold',]*len(cold['locs'])
    # events,locs = zip(*sorted(
    #     zip(events,warm['locs']+cold['locs']),key=lambda v:v[1]))
    # transition = {}
    # for iloc in xrange(len(events)-1):
    #     evt_pair = '_'.join(events[iloc:iloc+2])
    #     if locs[iloc+1]-locs[iloc] <= wait_window:
    #         transition[evt_pair] = transition.setdefault(evt_pair,0) + 1
    transition['warm'] = len(warm['locs'])
    transition['cold'] = len(cold['locs'])
    transition['warm_locs'] = warm['locs']
    transition['cold_locs'] = warm['locs']
    return transition


def ENSO_transition_prob(*args,**kwargs):
    tra = ENSO_transition(*args,**kwargs)
    nEN,nLN = tra['warm'],tra['cold']
    Prob = {}
    if tra['warm_locs'] and tra['cold_locs']:
        if tra['warm_locs'][-1] > tra['cold_locs'][-1]:
            # warm comes last
            nEN -= 1
        else:
            # cold comes last
            nLN -= 1
    elif tra['warm_locs']:
        # warm comes last
        nEN -= 1
    elif tra['cold_locs']:
        # cold comes last
        nLN -= 1
    for phase1,phase2 in itertools.product(['warm','cold'],['warm','cold']):
        tra_phase = phase1+"_"+phase2
        if phase1 == 'warm': 
            n = nEN
        else:
            n = nLN
        Prob[tra_phase] = float(tra.get(tra_phase,0.))/n
    return Prob


def ENSO_skewness(nino34):
    return stat.skewness(nino34)


def Threshold2Composite(field,nino34,warm_thres,cold_thres):
    '''Threshold only '''
    data = field[numpy.logical_or(nino34.data>warm_thres,nino34.data<cold_thres)]
    time = field.getTime()[numpy.logical_or(nino34.data>warm_thres,nino34.data<cold_thres)]
    itime = data.getCAxes().index('T')
    data.dims[itime].data = time
    return data


def Seasonal_Locking(pklocs,months):
    ''' Given the indices of the peak
    and a list of months, return the count of events in 
    a particular month (Jan-Dec).
    The indices of the events should refer to the same 
    time axis that the months are referring to
    '''
    event_months = months[pklocs]
    count = numpy.zeros(12)
    for month in event_months:
        count[month-1]+=1
    return count


def Seasonal_Locking_from_nino34(nino34,months,
                                 findEvents_func=lambda index: findENSO_threshold(index,0.8,-0.8),
                                 count_warm=True,count_cold=True):
    assert nino34.shape[0] == months.shape[0]
    warm,cold = findEvents_func(nino34)
    total_count = 0
    if count_warm:
        total_count += Seasonal_Locking(warm['locs'],months)
    if count_cold:
        total_count += Seasonal_Locking(cold['locs'],months)
    return total_count
