import numpy
import util

def lines(precip,x,y,thres):
    '''
    Given precip (numpy.array), x (lon in degree),
    y (lat in degree), locate the ITCZ and SPCZ central
    axis.

    thres - threshold for large precipitation,
            has to be a function
    
    Return {'NP': [slope,intercept],'SP':[slope,intercept]}
    '''
    x,y = numpy.meshgrid(x,y)
    if not isinstance(precip,numpy.ma.core.MaskedArray):
        precip = numpy.ma.array(precip)
    assert x.shape == y.shape == precip.data.shape

    xselected = x[precip.data > thres(precip.data)]
    yselected = y[precip.data > thres(precip.data)]

    # Northern Hemisphere x and y
    NPind = yselected > 0.
    xnp = xselected[NPind]
    ynp = yselected[NPind]

    # Southern Hemisphere x and y
    SPind = yselected < 0.
    xsp = xselected[SPind]
    ysp = yselected[SPind]

    results = {'NP': numpy.polyfit(xnp,ynp,1),
               'SP': numpy.polyfit(xsp,ysp,1)}
    return results


def precipfunc(var):
    ''' Select the region lat=(-20.,20.), lon=(150.,260.)
    var - util.nc.Variable
    '''
    return var.time_ave().squeeze().getRegion(lat=(-20.,20.),
                                                lon=(150.,260.))

def lines_var(var,thres):
    return lines(var.data,var.getLongitude(),var.getLatitude(),
                           thres)

def y0(var,lon,lat):
    ''' Given the var (numpy.ndarray), lon (numpy.array),
    lat (numpy.array), find the latitude of the ITCZ/SPCZ
    '''
    assert var.ndim == 2
    varxave = var.mean(axis=-1)
    y0 = {}
    # Northern
    y0['NP'] = lat[lat>0][varxave[lat>0].argmax()]
    # Southern
    y0['SP'] = lat[lat<0][varxave[lat<0].argmax()]
    return y0

def x0(var,lon,lat):
    ''' Given the var (numpy.ndarray), lon (numpy.array),
    lat (numpy.array), find the longitude of the ITCZ/SPCZ
    '''
    import util.stat
    assert var.ndim == 2
    assert var.shape[0] == len(lat)
    assert var.shape[1] == len(lon)
    varave = util.stat.runave(var,10,axis=-1)
    x0 = {}
    # Northern
    varaveNP = varave[lat>0,:].mean(axis=0)
    x0['NP'] = lon[varaveNP.argmax()]
    # Southern
    varaveSP = varave[lat<0,:].mean(axis=0)
    x0['SP'] = lon[varaveSP.argmax()]
    return x0
    
def width(var,lon,lat,thres):
    ''' Given the var (numpy.ndarray), lon (numpy.array),
    lat (numpy.array), find the slopes of the central axes of the ITCZ and
    rotate the field and find the widths of the ITCZ/SPCZ
    thres - threshold, has to be a function
    Return : {'NP': width in degree, 'SP': with in degree }
    
    '''
    # Get the slopes and intercepts for the ITCZ and SPCZ
    import scipy.ndimage.interpolation
    central_axes = lines(var,lon,lat,thres)
    widths = {}
    for key,value in central_axes.items():
        slope = value[0]
        theta = numpy.arctan(slope)*-1
        matrix = numpy.array([[numpy.cos(theta),-1*numpy.sin(theta)],
                              [numpy.sin(theta),numpy.cos(theta)]])
        newvar = scipy.ndimage.interpolation.affine_transform(
                                                           var,matrix)
        newvar_xave = newvar.mean(axis=-1)
        if slope > 0.: # NP
            regime = lat > 0.
            newvar_xave[lat<0.] = numpy.ma.masked
            ind = 0
        else:
            regime = lat < 0.
            newvar_xave[lat>0.] = numpy.ma.masked
            ind = -1
        # Half-width-half-maximum
        hwhm = numpy.abs(
                          numpy.extract(
                              numpy.logical_and(
                                       newvar_xave < newvar_xave.max()/2.,
                                       regime),
                              lat)[ind] - lat[newvar_xave.argmax()]
                         )
        # width parameter for gaussian function
        widths[key] = hwhm/numpy.sqrt(2.*numpy.log(2.))
        
    return widths

def analysis(var,thres):
    ''' Given var (util.nc.variable), perform an analysis on the ITCZ pattern
    Return:
    { 'NP': dict(key=['y0','width','slope','y-intercept'])
      'SP' : same as above }

    where y0 is the latitude of the max zonal mean
          width is the gaussian width of the ITCZ/SPCZ across itself
          slope,y-intercept gives the location of the ITCZ/SPCZ as
          lat = y-intercept + slope * lon
    Input:
    var - util.nc.Variable
    thres - a function that takes var data and return the threshold
    '''
    y0_ans = y0(var.data,var.getLongitude(),var.getLatitude())
    width_ans = width(var.data,var.getLongitude(),var.getLatitude(),thres)
    slopes_ans = lines(var.data,var.getLongitude(),var.getLatitude(),thres)
    x0_ans = x0(var.data,var.getLongitude(),var.getLatitude())
    results = { key: dict(y0=y0_ans[key],width=width_ans[key],
                          slope=slopes_ans[key][0],
                          y_intercept=slopes_ans[key][1],
                          x0=x0_ans[key]) for key in y0_ans.keys() }
    return results

def idealized(amp,x,y,x_loc,y_loc,slope,x_width,y_width):
    theta = numpy.arctan(slope)
    xP = (x - x_loc)*numpy.cos(theta) + (y - y_loc)*numpy.sin(theta)
    yP = -1.*(x - x_loc)*numpy.sin(theta) + (y - y_loc)*numpy.cos(theta)
    q = amp*numpy.exp(-0.5*(xP**2.)/(x_width**2.))*numpy.exp(-0.5*(yP**2.)/(y_width**2.))
    return q

