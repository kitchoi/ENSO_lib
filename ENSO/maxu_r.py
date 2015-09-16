import numpy
import geodat
import pylab
import functools
import geodat.keepdims as keepdims


def max_anom_index(u,lon_width=40.,lat=(-2.,2.),region=None,option='value',sign=1.):
    ''' Return the index where U is maximum after applying
    a lon_width degree running mean along the longitude.
    
    Assumed uniform grid
    
    Input:
    u          - geodat.nc.Variable
    lon_width  - width along the longitude for running average
    lat        - latitudinal band to be averaged over
    region     - region to be averaged over (default None, overwrites lat if defined)
    option     - 'lon','lat','value' for finding longitude of maximum, 
                  latitude of maximum and the maximum value respectively
    sign       - multiply the data by the sign of this argument
    '''
    if region is None:
        utmp = u.getRegion(lat=lat)
    else:
        utmp = u.getRegion(**region)
    
    runaveu = utmp.runave(lon_width,'X')
    runaveu.data *= numpy.sign(sign)
    
    def max_lon(runaveu):
        runaveu = runaveu.wgt_ave('Y').squeeze()
        index = numpy.ma.apply_along_axis(numpy.ma.argmax,runaveu.getCAxes().index('X'),runaveu.data)
        return runaveu.getLongitude()[index]
    
    def max_lat(runaveu):
        iyaxis = runaveu.getCAxes().index('Y')
        indices = numpy.unravel_index(numpy.ma.argmax(runaveu.data),
                                      runaveu.data.shape)
        return runaveu.getLatitude()[indices[iyaxis]]
    
    def max_value(runaveu):
        runaveu = runaveu.wgt_ave('Y').squeeze()
        ixaxis = runaveu.getCAxes().index('X')
        return geodat.nc.Variable(data=numpy.array(numpy.ma.max(runaveu.data,ixaxis)*sign),
                                dims=[d for d in runaveu.dims
                                      if d.getCAxis() not in 'XY'],
                                parent=runaveu)
    funcs = {'lon': max_lon,
             'lat': max_lat,
             'value': max_value}
    
    return funcs[option](runaveu)


def max_anom_lat(u,*args,**kwargs):
    ''' Return the latitude where U is maximum after applying
    a lat_width degree running mean along the longitude.
    Assumed uniform grid
    '''
    kwargs['option'] = 'lat'
    return max_anom_index(u,*args,**kwargs)


def max_anom_lon(u,*args,**kwargs):
    ''' Return the longitude where U is maximum after applying
    a lon_width degree running mean along the longitude.
    Assumed uniform grid
    '''
    kwargs['option'] = 'lon'
    return max_anom_index(u,*args,**kwargs)


def max_anom(u,*args,**kwargs):
    ''' Return the maximum U after applying a lon_width degree 
    running mean along the longitude.  Assumed uniform grid.
    Input:
    u - geodat.nc.Variable
    lon_width - scalar in degree (default = 40.)
    lat - region in the latitude (default = (-2.,2.))
    '''
    kwargs['option'] = 'value'
    return max_anom_index(u,*args,**kwargs)


def find_max_u_nino3_pairs(u_ref,nino3,lon_width=40.,lat=(-2.,2.)):
    ''' Find the pair of (u_ref, nino3) for warm and cold conditions
    after locating the longitude where the regression is largest for 
    each condition 
    
    Return:
    ( pos_u, pos_t ), ( neg_u, neg_t)
    pos_u, pos_t, neg_u and neg_t are all numpy 1d array
    '''
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
    ''' Plot (if doPlot is True) the results and compute 
    the value of r using
    linear regression.
    Input:
    y      - a numpy 1d array
    x      - x axis (default = [ -len(u_list)/2,...,1,0,-1,...-len(u_list)/2 ]
    Return: 
    r = 
      (slope_positive_x - slope_negative_x)/(slope_positive_x + slope_negative_x)
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


    
