import warnings
import itertools

import numpy
import scipy

try:
    import geodat
    _GEODAT_INSTALLED = True
    _IMPORT_GEODAT_ERROR = None
except ImportError:
    _GEODAT_INSTALLED = False
    _IMPORT_GEODAT_ERROR = ImportError("GeoDAT is not installed. "+\
                                "http://kychoi.org/geodat_doc")


def find_enso_percentile(index, percentile, *args, **kwargs):
    ''' Find ENSO events using percentiles (i.e. insensitive to time mean)

    Args:
        index (numpy.ndarray): ENSO SST anomaly monthly index.  Masked array
           is supported
        percentile (numeric): percentile beyond which El Nino and La Nina
           events are identified

    args and kwargs are fed to :py:func:`find_events`

    Returns:
       (dict, dict) each has keys "locs" and "peaks".  "locs" contains the index
         where an event peaks.  "peaks" contains the corresponding peak values

    Example::

        >>> warm,cold = find_enso_percentile(nino34,15.)
    '''
    if percentile > 50.:
        percentile = 100. - percentile
    if percentile < 0.:
        raise ValueError("percentile cannot be smaller than zero.")
    if percentile > 100.:
        raise ValueError("percentile cannot be bigger than 100.")

    warm = {}
    cold = {}
    if numpy.ma.isMaskedArray(index):
        if index.mask.any():
            # Ignore masked values
            warm['locs'], warm['peaks'] = find_events(index, '>',
                                                      numpy.percentile(
                                                          index[~index.mask],
                                                          100-percentile),
                                                      *args, **kwargs)
            cold['locs'], cold['peaks'] = find_events(index, '<',
                                                      numpy.percentile(
                                                          index[~index.mask],
                                                          percentile),
                                                      *args, **kwargs)
            return warm, cold
    # No masked values:
    warm['locs'], warm['peaks'] = find_events(index, '>',
                                              numpy.percentile(index,
                                                               100-percentile),
                                              *args, **kwargs)
    cold['locs'], cold['peaks'] = find_events(index, '<',
                                              numpy.percentile(index,
                                                               percentile),
                                              *args, **kwargs)
    return warm, cold


def find_enso_threshold(index, warm_threshold, cold_threshold, *args, **kwargs):
    ''' Similar to find_enso_percentile but uses threshold to find ENSO events
    Args:
        index (numpy.ndarray): ENSO SST anomaly monthly index.  Masked array
           is supported
        warm_threshold (numeric): Above which El Nino SST anomalies are
        cold_threshold (numeric): Below which La Nina SST anomalies are

    args and kwargs are fed to :py:func:`find_events`

    Returns:
       (dict, dict) each has keys "locs" and "peaks".  "locs" contains the index
         where an event peaks.  "peaks" contains the corresponding peak values
    '''
    warm = {}
    cold = {}
    if not isinstance(index, numpy.ma.core.MaskedArray):
        index = numpy.ma.array(index)
    warm['locs'], warm['peaks'] = find_events(index, '>', warm_threshold,
                                              *args, **kwargs)
    cold['locs'], cold['peaks'] = find_events(index, '<', cold_threshold,
                                              *args, **kwargs)
    return warm, cold


def persistence_check(is_selected, per):
    ''' Filter is_selected where True indicating an active, persistent event

    Args:
        is_selected (numpy boolean array)
        per (int): length persistence required

    Returns:
        numpy boolean array same shape as is_selected
    '''
    assert len(is_selected) > 0
    assert is_selected.dtype == numpy.bool
    if not numpy.any(is_selected):
        return is_selected
    is_filtered = numpy.zeros_like(is_selected, dtype=numpy.bool)
    in_event = 0
    n_select = 0
    i_start = 0
    for i, selected in enumerate(is_selected):
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


def find_events(index, operator, threshold, per=5, window=[-3, 3]):
    ''' Return the locations of ENSO warm/cold event peaks for a given index.

    Args:
        index (numpy 1d array): ENSO SST anomaly
        operator (str): ">" (index greater than threshold) or "<" (index smaller
            then threshold)
        threshold (numeric): threshold for event definition
        per (int): minimum persistence for index >/< threshold
                (default=5, unit is consistent with the array grid)
        window  (iterable): range around the event peak within which the peak
            has to be a global minima/maxima; length = 2 (default=[-3,3])

    Returns:
       (pklocs, pks) = (location in the input array, values of extrema)
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

    if len(window) != 2:
        raise ValueError("window must have length=2")

    locs = numpy.where(comp_op(index, threshold))[0]
    if len(locs) <= 1:
        return ([], numpy.array([]))

    # Find the beginning (starts) and the end (ends) of events
    jumps = numpy.where(numpy.diff(locs) > 1)[0]
    starts = numpy.insert(locs[jumps+1], 0, locs[0])
    ends = numpy.append(locs[jumps], locs[-1])

    # Ignore the chunks that starts from the beginning or ends at the end of the
    # index
    if starts[0] == 0:
        starts = starts[1:]
        ends = ends[1:]
    if ends[-1] == len(index)-1:
        starts = starts[:-1]
        ends = ends[:-1]

    # Chunks of the index that exceed the threshold
    subsets = [index[starts[i]:ends[i]] for i in range(len(starts))]

    # Find the location of peaks and apply persistence check
    pklocs = [starts[i]+argpeak_op(subsets[i])
              for i in range(len(subsets))
              if len(subsets[i]) >= per]

    # Check for being global extrema within the window
    pklocs_new = []
    local_append = pklocs_new.append
    for loc in pklocs:
        window_start = numpy.max([0, loc+window[0]])
        window_end = numpy.min([len(index)-1, loc+window[1]])
        if index[loc] == peak_op(index[window_start:window_end]):
            local_append(loc)

    # I don't think this does anything more than copying pklocs_new to pklocs
    pklocs = [int(loc) for loc in pklocs_new if loc != False]

    pks = numpy.array([index[loc].squeeze() for loc in pklocs])

    return pklocs, pks


def find_en_pattern(field, nino34, nino34_mid=0.8, nino34_tole=0.4,
                    do_climo=True, verbose=False):
    ''' Given a field and Nino3.4 index monthly time series, extract the time at
    which nino34_mid-nino34_tole < peak nino34 < nino34_mid+nino34_tole
    Then compute the climatology for these snap shots

    Args:
        field (geodat.nc.Variable)
        nino34 (geodat.nc.Variable): Nino3.4 SST anomaly index
        nino34_mid (numeric): mid point
        nino34_tole (numeric): half bin size
        do_climo (bool): whether a climatology is computed for the composite,
           default True
        verbose (bool)

    Returns:
        pattern (geodat.nc.Variable)

    This function requires the library `GeoDAT <http://kychoi.org/geodat_doc>`_
    '''
    if not _GEODAT_INSTALLED:
        raise _IMPORT_GEODAT_ERROR

    if not numpy.allclose(field.getTime(), nino34.getTime()):
        raise Exception("Expect the time record of nino3.4 and "+\
                        "the field to be the same")

    warm, cold = find_enso_percentile(nino34.data, 49.)
    locs = warm['locs'] + cold['locs']
    peaks = numpy.append(warm['peaks'], cold['peaks'])
    locs = numpy.array(locs)[numpy.abs(peaks - nino34_mid) < nino34_tole]
    if len(locs) == 0:
        field_sliced_time = field.getRegion(time=slice(0, 1))
        result = geodat.nc.Variable(data=numpy.ma.ones(
            field_sliced_time.data.shape),
                                    parent=field_sliced_time)
        result.data[:] = numpy.ma.masked
        warnings.warn("No event found!")
        return result
    pattern = field[locs]
    if do_climo:
        pattern = geodat.nc.climatology(pattern)
    pattern.setattr('event_loc', locs.squeeze())
    if verbose: print 'Nino 3.4: '+ str(nino34[locs].time_ave().squeeze().data)
    return pattern


def compute_duration(nino34, operator, locs, evt_end,
                     remove_merge_event=True):
    ''' Compute the duration of events counting from the event peak (locations
    given by locs) until the termination of events (given by the first
    occurrence of operator(nino34,evt_end)).

    See `Choi et al (2013) <http://dx.doi.org/10.1175/JCLI-D-13-00045.1>`_

    Args:
        nino34 (numpy array): ENSO SST anomaly index
        operator (numpy operator): e.g. numpy.less or numpy.greater
        locs (list of int or numpy array of int): indices of event peaks
        evt_end (scalar number): value of nino34 when an event is considered as
           terminated
        remove_merge_event (bool): remove events that are joined on together
           due to reintensification

    Returns:
        list of int (same length as locs)
    '''
    lengths = []
    for iloc in xrange(len(locs)):
        loc = locs[iloc]
        after_end = operator(nino34[loc:], evt_end)
        if after_end.any():
            length = numpy.where(after_end)[0][0]
            if remove_merge_event:
                 # if this is not the last event peak
                if iloc < len(locs)-1:
                    # Only count the duration if the next event occurs after
                    # the termination
                    if locs[iloc+1]-locs[iloc] > length:
                        lengths.append(length)
            else:
                lengths.append(length)

    return lengths



def enso_duration(nino34, percentile, thres_std_fraction,
                  per=5, window=[-3, 3]):
    '''Compute the duration of the warm and cold events

    Args:
        nino34 (numpy 1D array): ENSO SST anomalies
        percentile (numeric): greater than 0 and less than 100
        thres_std_fraction (numeric): the fraction times the nino34 standard
              deviation is used for defining the termination

    Returns:
        dict: with keys "warm" and "cold" each contains a list of integers which
             are the duration of the events
    '''
    warm, cold = find_enso_percentile(nino34, percentile, per, window)
    warm_end = nino34.std()*thres_std_fraction
    cold_end = warm_end*-1
    duration = {}
    duration['warm'] = compute_duration(nino34, numpy.less, warm['locs'],
                                        warm_end)
    duration['cold'] = compute_duration(nino34, numpy.greater, cold['locs'],
                                        cold_end)
    duration['warm_locs'] = warm['locs']
    duration['cold_locs'] = cold['locs']
    # Length of duration['warm'] may not match the length of
    # duration['warm_locs'] as double counting is taken care of
    return duration


def enso_transition2(nino34, percentile, wait_window, thres_std_fraction,
                     per=5, window=[-3, 3]):
    ''' Compute the transition probabilities.  Word for word copy from the
    Matlab code, which was used in.
    See `Choi et al (2013) <http://dx.doi.org/10.1175/JCLI-D-13-00045.1>`_

    Args:
        nino34 (numpy 1d array): ENSO SST anomaly index
        percentile (numeric): greater than 0. and less than 100.
        wait_window (int): time lapse between an event termination and the next
          event peak
        thres_std_fraction (numeric): the fraction times the nino34 standard
              deviation is used for defining the termination, similar to
              :py:func:`enso_duration`
        per (int): see :py:func:`find_events`
        windows (iterable): see :py:func:`find_events`

    Returns:
       transition (dict), indices(dict), warm_indices(list), cold_indices(list)
         transition= {"warm_cold": probability of warm-to-cold transition,
                            "warm_warm": probability of warm-to-warm transition,
                            "cold_warm":..., "cold_cold":...,}
         indices= {"ENLN": indices of the peaks of warm events that are
                             followed by a cold event,
                   "ENEN": indices of the peaks of warm events that are
                             followed by a warm event,
                   "LNLN": indices of the peaks of cold events that are
                             followed by a cold event,
                   "LNEN": indices of the peaks of cold events that are
                             followed by a warm event}
         warm_indices (list) is the list of warm event peak locations;
         cold_indices (list) is the list of cold event peak locations
    '''
    ENLNind = []
    LNENind = []
    ENENind = []
    LNLNind = []
    warm, cold = find_enso_percentile(nino34, percentile, per, window)
    warm['locs'] = numpy.array(sorted(warm['locs']))
    cold['locs'] = numpy.array(sorted(cold['locs']))
    warm_end = nino34.std()*thres_std_fraction
    cold_end = warm_end*-1
    for i, iwarm in enumerate(warm['locs']):
        ended = numpy.where(nino34[iwarm:] < warm_end)[0]
        if len(ended):
            # The index where it terminates
            iterm = iwarm + ended[0]
        nextcold = numpy.where(cold['locs'] > iwarm)[0]
        nextwarm = numpy.where(warm['locs'] > iwarm)[0]
        if len(nextcold) and len(nextwarm):
            inextcold = cold['locs'][nextcold[0]]
            inextwarm = warm['locs'][nextwarm[0]]
            if (inextcold - iterm) <= wait_window:
                ENLNind.append(iwarm)
            elif (inextwarm - iterm) <= wait_window:
                ENENind.append(iwarm)
        elif len(nextcold):
            if (cold['locs'][nextcold[0]]-iterm) <= wait_window:
                ENLNind.append(iwarm)
        elif len(nextwarm):
            if (warm['locs'][nextwarm[0]]-iterm) <= wait_window:
                ENENind.append(iwarm)
    for i, icold in enumerate(cold['locs']):
        ended = numpy.where(nino34[icold:] > cold_end)[0]
        if len(ended):
            # The index where it terminates
            iterm = icold + ended[0]
        nextwarm = numpy.where(warm['locs'] > icold)[0]
        nextcold = numpy.where(cold['locs'] > icold)[0]
        if len(nextwarm) and len(nextcold):
            inextwarm = warm['locs'][nextwarm[0]]
            inextcold = cold['locs'][nextcold[0]]
            if (inextwarm - iterm) <= wait_window:
                LNENind.append(icold)
            elif (inextcold - iterm) <= wait_window:
                LNLNind.append(icold)
        elif len(nextwarm):
            if (warm['locs'][nextwarm[0]]-iterm) <= wait_window:
                LNENind.append(icold)
        elif len(nextcold):
            if (cold['locs'][nextcold[0]]-iterm) <= wait_window:
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
    return transition, dict(ENLN=ENLNind, ENEN=ENENind,
                            LNEN=LNENind, LNLN=LNLNind),\
           warm["locs"], cold["locs"]



def enso_transition(nino34, percentile, wait_window, thres_std_fraction,
                    per=5, window=[-3, 3]):
    ''' Compute the number of transitions
    However, it does not eliminate double counting due to reintensification.
    Therefore the results are different from those of
    :py:func:`enso_transition2`

    Args:
        nino34 (numpy 1d array): ENSO SST anomaly index
        percentile (numeric): 0. < percentile < 100.
        wait_window (int): time lapse between an event termination and the next
          event peak
        thres_std_fraction (numeric): the fraction times the nino34 standard
              deviation is used for defining the termination, similar to
              :py:func:`enso_duration`
        per (int): see :py:func:`find_events`
        windows (iterable): see :py:func:`find_events`

    Returns:
       transition (dict) = {"warm_cold": number of warm-to-cold transition,
                            "warm_warm": number of warm-to-warm transition,
                            "cold_warm":..., "cold_cold":...,}
    '''
    warm, cold = find_enso_percentile(nino34, percentile, per, window)
    warm_end = nino34.std()*thres_std_fraction
    cold_end = warm_end*-1
    durations = {'warm': compute_duration(nino34, numpy.less,
                                          warm['locs'], warm_end, False),
                 'cold': compute_duration(nino34, numpy.greater,
                                          cold['locs'], cold_end, False),}
    events = ['warm',]*len(warm['locs'])+\
             ['cold',]*len(cold['locs'])
    events, locs, termT = zip(*sorted(
        zip(events,
            warm['locs']+cold['locs'],
            durations['warm']+durations['cold']),
        key=lambda v: v[1]))
    transition = {}
    for iloc in xrange(len(events)-1):
        evt_pair = '_'.join(events[iloc:iloc+2])
        if (locs[iloc+1] - locs[iloc] - termT[iloc]) <= wait_window:
            transition[evt_pair] = transition.setdefault(evt_pair, 0) + 1
    # warm,cold = find_enso_percentile(nino34,percentile,per,window)
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


def enso_transition_prob(*args, **kwargs):
    ''' Compute the transition probability from the results given by
    :py:func:`enso_transition`

    All arguments and keyword arguments are fed to :py:func:`enso_transition`'''
    tra = enso_transition(*args, **kwargs)
    nEN, nLN = tra['warm'], tra['cold']
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
    for phase1, phase2 in itertools.product(['warm', 'cold'],
                                            ['warm', 'cold']):
        tra_phase = phase1+"_"+phase2
        if phase1 == 'warm':
            n = nEN
        else:
            n = nLN
        Prob[tra_phase] = float(tra.get(tra_phase, 0.))/n
    return Prob


def enso_skewness(nino34):
    ''' Shortcut for calculating skewness using scipy.stats.
    Choose scipy.stats.mstats.skew if nino34 is a masked array (as it often is)

    Args:
        nino34 (numpy 1d array): ENSO SST anomaly index

    Returns:
        scalar
    '''
    if _GEODAT_INSTALLED:
        return geodat.stat.skewness(nino34)
    else:
        if isinstance(nino34, numpy.ma.coreMaskedArray):
            return scipy.stats.mstats.skew(nino34)
        else:
            return scipy.stats.skew(nino34)


## This is assuming geodat.nc.Variable inputs
## The current version of geodat.nc.Variable can achieve the following easily
# def threshold2composite(field,nino34,warm_thres,cold_thres):
#     '''Threshold only '''
#     # Extract the times when the thresholds are passed
#     data = field[numpy.logical_or(nino34.data > warm_thres,
#                                   nino34.data < cold_thres)]
#     # Get the time stamps
#     time = field.getTime()[numpy.logical_or(nino34.data > warm_thres,
#                                             nino34.data < cold_thres)]
#     itime = data.getCAxes().index('T')
#     # Update time axis
#     data.dims[itime].data = time
#     return data


def seasonal_locking(pklocs, months):
    ''' Given the indices of the peak and a list of months, return the count of
    events in a particular month (Jan-Dec).

    Args:
        pklocs (list of int): indices of event peaks with reference to some time
            axis
        months (list of int): calendar months of that time axis

    Returns:
        list of int = [number of events peaked in Jan,
                       number of events peaked in Feb,...]
    '''
    event_months = months[pklocs]
    count = numpy.zeros(12)
    for month in event_months:
        count[month-1] += 1
    return count


def seasonal_locking_from_nino34(nino34, months,
                                 find_events_func=None,
                                 count_warm=True, count_cold=True):
    ''' Shortcut for applying :py:func:`seasonal_locking` manually for warm and
    cold events.

    Args:
        nino34 (numpy 1d array): ENSO SST monthly anomaly
        months (list of int or numpy 1d array of int): calendar months for the
          time axis of nino34
        find_events_func (function): a function that accepts nino34 as the only
          arguments for finding warm and cold events.
          If None, default is find_enso_threshold(nino34, 0.8, -0.8)
        count_warm (bool): whether warm events are counted
        count_cold (bool): whether cold events are counted

    Returns:
        list of int = [number of events peaked in Jan,
                       number of events peaked in Feb,...]
    '''
    if find_events_func is None:
        find_events_func = lambda index: find_enso_threshold(index, 0.8, -0.8)
    assert nino34.shape[0] == months.shape[0]
    warm, cold = find_events_func(nino34)
    total_count = 0
    if count_warm:
        total_count += seasonal_locking(warm['locs'], months)
    if count_cold:
        total_count += seasonal_locking(cold['locs'], months)
    return total_count


def __keep_old_function_name__(f):
    ''' For preserving old function names used in previous projects '''
    def new_func(*args, **kwargs):
        return f(*args, **kwargs)
    new_func.__doc__ = "Same as "+f.__name__
    return new_func


# Functions are renamed in order to compile with coding style convention
# For compatibility with previous projects, old names are kept here
findENSO_percentile = __keep_old_function_name__(find_enso_percentile)
findENSO_threshold = __keep_old_function_name__(find_enso_threshold)
Persistence_check = __keep_old_function_name__(persistence_check)
findEvents = __keep_old_function_name__(find_events)
find_EN_pattern = __keep_old_function_name__(find_en_pattern)
ENSO_duration = __keep_old_function_name__(enso_duration)
ENSO_transition2 = __keep_old_function_name__(enso_transition2)
ENSO_transition = __keep_old_function_name__(enso_transition)
ENSO_transition_prob = __keep_old_function_name__(enso_transition_prob)
ENSO_skewness = __keep_old_function_name__(enso_skewness)
Seasonal_Locking = __keep_old_function_name__(seasonal_locking)
Seasonal_Locking_from_nino34 = __keep_old_function_name__(
    seasonal_locking_from_nino34)
