import functools

import numpy
import pylab

import geodat


def max_anom_index(u, lon_width=40., lat=(-2.,2.), region=None, option='value',
                   sign=1.):
    ''' Return the index where U is maximum after applying
    a lon_width degree running mean along the longitude.

    Assumed uniform grid

    Args:
         u (geodat.nc.Variable): any field with latitude and longitude
         lon_width (number): width along the longitude for running average
         lat (iterable): latitudinal region to be averaged over
         region (dict): region to be averaged over (default None, overwrites
             lat if defined, see `geodat.nc.getRegion </geodat_doc/nc.html#geodat.nc.Variable.getRegion>`_
         option (str): 'lon'/'lat'/'value' for finding longitude of maximum,
                  latitude of maximum and the maximum value respectively
         sign (scalar): if positive, the function looks for maximum values; if
                  negative, the function looks for minimum values (useful for
                  analysing La Nina)
    '''
    if region is None:
        utmp = u.getRegion(lat=lat)
    else:
        utmp = u.getRegion(**region)

    runaveu = utmp.runave(lon_width, 'X')
    runaveu.data *= numpy.sign(sign)

    def max_lon(runaveu):
        ''' Option = "lon", find longitude of the maximum '''
        runaveu = runaveu.wgt_ave('Y').squeeze()
        index = numpy.ma.apply_along_axis(numpy.ma.argmax,
                                          runaveu.getCAxes().index('X'),
                                          runaveu.data)
        return runaveu.getLongitude()[index]

    def max_lat(runaveu):
        ''' Option = "lat", find latitude of the maximum '''
        iyaxis = runaveu.getCAxes().index('Y')
        indices = numpy.unravel_index(numpy.ma.argmax(runaveu.data),
                                      runaveu.data.shape)
        return runaveu.getLatitude()[indices[iyaxis]]

    def max_value(runaveu):
        ''' Option = "value", find the maximum value'''
        runaveu = runaveu.wgt_ave('Y').squeeze()
        ixaxis = runaveu.getCAxes().index('X')
        return geodat.nc.Variable(data=numpy.array(
            numpy.ma.max(runaveu.data, ixaxis)*sign),
                                  dims=[d for d in runaveu.dims
                                        if d.getCAxis() not in 'XY'],
                                  parent=runaveu)
    funcs = {'lon': max_lon,
             'lat': max_lat,
             'value': max_value}

    return funcs[option](runaveu)


def max_anom_lat(u,*args,**kwargs):
    ''' Same as :py:func:`max_anom_index` but with option="lat" enforced
    '''
    kwargs['option'] = 'lat'
    return max_anom_index(u,*args,**kwargs)


def max_anom_lon(u,*args,**kwargs):
    ''' Same as :py:func:`max_anom_index` but with option="lon" enforced
    '''
    kwargs['option'] = 'lon'
    return max_anom_index(u,*args,**kwargs)


def max_anom(u,*args,**kwargs):
    ''' Same as :py:func:`max_anom_index` but with option="value" enforced
    '''
    kwargs['option'] = 'value'
    return max_anom_index(u,*args,**kwargs)


def find_max_u_nino3_pairs(u_ref,nino3,lon_width=40.,lat=(-2.,2.)):
    ''' Locate the maximum u_ref within a region of fixed size (lon_width * lat
    range) at each nino3.  Return the results as two pairs of (u_ref, nino3) for
    warm and cold conditions respectively

    Args:
        u_ref (geodat.nc.Variable): zonal wind anomaly
        nino3 (geodat.nc.Variable): ENSO SST anomaly index
        lon_width (number): width along the longitude for running average
        lat (iterable): latitudinal region to be averaged over
    
    The time axes of u_ref and nino3 should match

    Returns:
        ((pos_u,pos_t), (neg_u,neg_t)) contains numpy 1d arrays
    '''
    if not numpy.allclose(nino3.getTime(),u_ref.getTime()):
        raise ValueError("The time axes of u_ref and nino3 should match")

    # Positive nino3
    pos_slice = nino3.data > 0.
    pos_regress = geodat.nc.regress(u_ref.getRegion(time=pos_slice),
                                    nino3.getRegion(time=pos_slice))
    pos_lon = max_anom_lon(pos_regress)
    # Negative nino3
    neg_slice = nino3.data < 0.
    neg_regress = geodat.nc.regress(u_ref.getRegion(time=neg_slice),
                                  nino3.getRegion(time=neg_slice))
    neg_lon = max_anom_lon(neg_regress,lon_width=lon_width,lat=lat)
    u_regional = u_ref.getRegion(lon=(pos_lon-lon_width/2.,pos_lon+lon_width/2.),lat=lat).area_ave().data.squeeze()
    pos_u = u_regional[pos_slice]
    neg_u = u_regional[neg_slice]
    pos_t = nino3.data[pos_slice]
    neg_t = nino3.data[neg_slice]
    return (pos_u,pos_t),(neg_u,neg_t)


def plot_r(y,x,doPlot=True,*args,**kwargs):
    ''' Plot (if doPlot is True) the results and compute the value of r using
    linear regression.

    Args:
        y (numpy 1d array)
        x (numpy 1d array): x axis (default = [-len(y)/2,...,1,0,-1,...-len(y)/2]

    Returns:
        r (number): (slope_positive_x - slope_negative_x)/(slope_positive_x + slope_negative_x)
    '''
    #maxu = numpy.array([max_anom(u) for u in u_list])

    if len(y) != len(x):
        raise ValueError("Length of y should match that of x")
    #yx = numpy.array(zip(y,x),dtype=[('y','f4'),('x','f4')])
    #yx = numpy.sort(yx,order='x')
    s_neg = numpy.polyfit(x[x<=0],y[x<=0],1)[0]
    s_pos = numpy.polyfit(x[x>=0],y[x>=0],1)[0]
    #y = [ a[0] for a in yx ]
    #x = [ a[1] for a in yx ]
    if doPlot:
        #pylab.plot(x,y,*args,**kwargs)
        pylab.scatter(x,y)
    return (s_pos - s_neg)/(s_pos + s_neg)
