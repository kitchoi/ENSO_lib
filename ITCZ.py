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
    And perform time average
    var - util.nc.Variable
    Return a util.nc.Variable
    '''
    return var.getRegion(lat=(-20.,20.),
                         lon=(150.,260.)).time_ave().squeeze()

def lines_var(var,thres):
    return lines(var.data,var.getLongitude(),var.getLatitude(),
                           thres)

def ITCZ_lat_x(var,lon,lat,axis=-2,weighted=True):
    ''' Latitude of the ITCZ as a function of longitude
    weighted by the intensity of the ITCZ
    var - numpy.ndarray
    lon - numpy.array
    lat - numpy.array
    '''
    import keepdims
    assert var.shape[-1] == len(lon)
    assert var.shape[-2] == len(lat)
    assert type(axis) is int
    lonND,latND = numpy.meshgrid(lon,lat)  # (y,x)
    lonND = lonND[(numpy.newaxis,)*(var.ndim-2)]
    latND = latND[(numpy.newaxis,)*(var.ndim-2)]
    
    axis = axis % var.ndim 
    y0 = {}
    if weighted:
        # Northern
        y0['NP'] = keepdims.sum(var*latND*(latND>0),axis=axis)/\
                   keepdims.sum(var*(latND>0),axis=axis)
        # Southern
        y0['SP'] = keepdims.sum(var*latND*(latND<0),axis=axis)/\
                   keepdims.sum(var*(latND<0),axis=axis)
    else:
        # Northern
        y0['NP'] = lat[(var*(latND>0)).argmax(axis=axis)]
        # Southern
        y0['SP'] = lat[(var*(latND<0)).argmax(axis=axis)]
    
    return y0



def y0(var,lon,lat):
    ''' Given the var (numpy.ndarray), lon (numpy.array),
    lat (numpy.array), find the latitude of the ITCZ/SPCZ
    '''
    assert var.ndim == 2
    assert var.shape[0] == len(lat)
    assert var.shape[1] == len(lon)
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

def xave_max(var,lon,lat):
    ''' Given the var (numpy.ndarray),
    lon (numpy.array), lat(numpy.array),
    compute the zonal average and then 
    return the max for the ITCZ(y>0)/SPCZ(y<0)
    '''
    assert var.ndim == 2
    assert var.shape[0] == len(lat)
    assert var.shape[1] == len(lon)
    varxave = var.mean(axis=-1)
    amp = {}
    # Northern
    amp['NP'] = varxave[lat>0,:].max()
    # Southern
    amp['SP'] = varxave[lat<0,:].max()
    return amp
    

def amp_sum(var,lon,lat):
    ''' Given the var (numpy.ndarray),
    lon (numpy.array), lat(numpy.array),
    return the sum for the ITCZ(y>0)/SPCZ(y<0)
    '''
    amp = {}
    # Northern
    amp['NP'] = var[lat>0,:].sum()
    # Southern
    amp['SP'] = var[lat<0,:].sum()
    return amp
    
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
        if key == 'NP':
            newvar_xave[lat<0] = numpy.ma.masked
        else:
            newvar_xave[lat>0] = numpy.ma.masked
        
        imax = newvar_xave.argmax()
        iwidth = numpy.argmin(
            numpy.abs(newvar_xave/newvar_xave[imax]
                          - numpy.exp(-1.)))
        widths[key] = numpy.abs(lat[iwidth] - lat[imax])/numpy.sqrt(2.)
    
    return widths

def analysis(var,thres):
    ''' Given var (util.nc.variable), perform an analysis on the ITCZ pattern
    Return:
    { 'NP': dict(key=['y0','x0','width','slope','y-intercept','amp'])
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
    xave_max_ans = xave_max(var.data,var.getLongitude(),var.getLatitude())
    results = { key: dict(y0=y0_ans[key],width=width_ans[key],
                          slope=slopes_ans[key][0],
                          y_intercept=slopes_ans[key][1],
                          x0=x0_ans[key],
                          xave_max=xave_max_ans[key]) for key in y0_ans.keys() }
    return results

def idealized(amp,x,y,x_loc,y_loc,slope,x_width,y_width,return_Variable=False):
    theta = numpy.arctan(slope)
    if x.ndim ==2 and y.ndim ==2:
        assert x.shape == y.shape
    else:
        assert x.ndim == 1 and y.ndim == 1
        x,y = numpy.meshgrid(x,y)
    
    xP = (x - x_loc)*numpy.cos(theta) + (y - y_loc)*numpy.sin(theta)
    yP = -1.*(x - x_loc)*numpy.sin(theta) + (y - y_loc)*numpy.cos(theta)
    #if amp_sum is not None:
    #    print('amp_sum is given and is used to recalculate amp.')
    #    amp = amp_sum/2.*numpy.pi/y_width/x_width
    
    q = amp*numpy.exp(-0.5*(xP**2.)/(x_width**2.))*numpy.exp(-0.5*(yP**2.)/(y_width**2.))
    #if amp_sum is not None:
    #    print "q_sum = {:.2f} and amp_sum = {:.2f}".format(q.sum(),amp_sum)
    if return_Variable:
        newvar = util.nc.Variable(data=q,varname="Q",
                                  dims=[util.nc.Dimension(data=y[:,0],
                                                          dimname='LAT',units='degrees_N'),
                                        util.nc.Dimension(data=x[0,:],
                                                          dimname='LON',units='degrees_E')])
        return newvar
    else:
        return q


def y_max(precip,lat,yaxis=-2,ave_axis=-1):
    import keepdims
    var_ave = keepdims.mean(precip,axis=ave_axis)
    return lat[var_ave.argmax(axis=yaxis)].squeeze()
