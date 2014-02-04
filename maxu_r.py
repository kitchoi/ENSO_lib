import numpy
import util
import pylab

def max_anom_index(u,lon_width=40.,lat=(-2.,2.),region=None,option='value',sign=None):
    ''' Return the index where U is maximum after applying
    a lon_width degree running mean along the longitude.
    Assumed uniform grid
    '''
    if region is None:
        utmp = u.getRegion(lat=lat)
    else:
        utmp = u.getRegion(**region)
    
    if sign is None: 
        sign = numpy.sign(utmp.data)
    else:
        assert sign.ndim == 1 
        sign_shape = [ 1 if utmp.data.shape[i] != len(sign) else utmp.data.shape[i] 
                       for i in range(utmp.data.ndim)]
        sign = numpy.sign(sign.reshape(sign_shape))
    
    runaveu = (utmp*sign).wgt_ave('Y').runave(lon_width,'X').squeeze()
    ixaxis = runaveu.getCAxes().index('X')
    if option == 'loc':
        index = numpy.ma.apply_along_axis(numpy.ma.argmax,ixaxis,runaveu.data)
        return u.getLongitude()[index]
    elif option == 'value':
        return util.nc.Variable(data=\
                           numpy.ma.apply_along_axis(numpy.ma.max,ixaxis,runaveu.data)*sign.squeeze(),
                                parent=runaveu)

def max_anom_loc(u,*args,**kwargs):
    ''' Return the longitude where U is maximum after applying
    a lon_width degree running mean along the longitude.
    Assumed uniform grid
    '''
    kwargs['option'] = 'loc'
    return max_anom_index(u,*args,**kwargs)

def max_anom(u,*args,**kwargs):
    ''' Return the maximum U after applying a lon_width degree 
    running mean along the longitude.  Assumed uniform grid.
    Input:
    u - util.nc.Variable
    lon_width - scalar in degree (default = 40.)
    lat - region in the latitude (default = (-2.,2.))
    '''
    kwargs['option'] = 'value'
    return max_anom_index(u,*args,**kwargs)

def plot_r(y,x,doPlot=True,*args,**kwargs):
    ''' Read a list of variable, apply maxu_anom to each of them
    to find the maximum.  Plot (if doPlot is True) the results and compute 
    the value of r using
    linear regression.
    Input:
    y      - a numpy array
    x      - x axis (default = [ -len(u_list)/2,...,1,0,-1,...-len(u_list)/2 ]
    Return: 
    r = 
      (slope_positive_x - slope_negative_x)/(slope_positive_x + slope_negative_x)
    '''
    #maxu = numpy.array([max_anom(u) for u in u_list])

    y = numpy.append(y,0.)
    x = numpy.append(x,0.)
    
    assert len(y) == len(x)
    yx = numpy.array(zip(y,x),dtype=[('y','f4'),('x','f4')])
    yx = numpy.sort(yx,order='x')
    s_neg = numpy.polyfit(x[x<=0],y[x<=0],1)[0]
    s_pos = numpy.polyfit(x[x>=0],y[x>=0],1)[0]
    y = [ a[0] for a in yx ]
    x = [ a[1] for a in yx ]
    if doPlot:
        #pylab.plot(x,y,*args,**kwargs)
        pylab.scatter(x,y)
    return (s_pos - s_neg)/(s_pos + s_neg)


    
