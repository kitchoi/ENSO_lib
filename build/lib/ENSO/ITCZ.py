import numpy
import scipy.ndimage.interpolation

try:
    import geodat
    _GEODAT_INSTALLED = True
    _import_geodat_error = None
except ImportError:
    _GEODAT_INSTALLED = False
    _import_geodat_error = ImportError("GeoDAT is needed but not installed."+\
                                       "http://kychoi.org/geodata_doc")


def lines(precip, x1d, y1d, thres):
    '''
    Locate the ITCZ and SPCZ orientations (assumed straight line).

    Args:
        precip (numpy 2d array)
        x1d (numpy 1d array): longitude
        y1d (numpy 1d array): latitude
        thres (function): a function that takes precip
             as the only argument, the result is used as the threshold for
             precipitation. e.g. lambda precip: numpy.percentile(precip,70.)

    Returns:
        dict: {'NP': [slope,intercept],'SP':[slope,intercept]}
    '''
    x, y = numpy.meshgrid(x1d, y1d)
    if not isinstance(precip, numpy.ma.core.MaskedArray):
        precip = numpy.ma.array(precip)
    assert x.shape == y.shape == precip.data.shape

    xselected = x[precip > thres(precip)]
    yselected = y[precip > thres(precip)]

    # Northern Hemisphere x and y
    NPind = yselected > 0.
    xnp = xselected[NPind]
    ynp = yselected[NPind]

    # Southern Hemisphere x and y
    SPind = yselected < 0.
    xsp = xselected[SPind]
    ysp = yselected[SPind]

    results = {'NP': numpy.polyfit(xnp, ynp, 1),
               'SP': numpy.polyfit(xsp, ysp, 1)}
    return results


def precipfunc(var, region=dict(lat=(-20., 20.), lon=(150., 260.))):
    ''' Select the region and perform time average

    Args:
        var (geodat.nc.Variable)
        region (dict)

    Returns:
        geodat.nc.Variable

    This function requires the `GeoDAT </geodat_doc>`_ library
    '''
    if not _GEODAT_INSTALLED:
        raise _import_geodat_error

    return var.getRegion(lat=(-20., 20.),
                         lon=(150., 260.)).time_ave().squeeze()


def lines_var(var, thres):
    ''' Shortcut for using lines with geodat.nc.Variable

    Args:
        var (geodat.nc.Variable): precipitation
        thres (function): see :py:func:`ENSO.ITCZ.lines`

    Returns:
        dict: {'NP': [slope,intercept],'SP':[slope,intercept]}

    This function requires the `GeoDAT </geodat_doc>`_ library
    '''
    if not _GEODAT_INSTALLED:
        raise _import_geodat_error

    return lines(var.data, var.getLongitude(), var.getLatitude(), thres)


def ITCZ_lat_x(var, lon, lat, weighted=True):
    ''' Return the latitude of the ITCZ/SPCZ as a function of longitude with the
    option of being weighted by the intensity of the ITCZ/SPCZ

    Args:
        var (numpy.ndarray): precipitation (...,lat,lon)
        lon (numpy.array): longitudes
        lat (numpy.array): latitudes
        weighted (bool): default True

    Returns:
        dict: {'NP':latitudes (numpy array) of the ITCZ,
               'SP':latitudes (numpy array) of the SPCZ}
    '''
    if var.ndim < 2:
        raise ValueError("var should have the shape of (...,lat,lon)")
    if var.shape[-1] != len(lon):
        raise ValueError("Latitude must be the last dimension")
    if var.shape[-2] != len(lat):
        raise ValueError("Latitude must be the second last dimension")

    lonND,latND = numpy.meshgrid(lon,lat)
    lonND = lonND[(numpy.newaxis,)*(var.ndim-2)]
    latND = latND[(numpy.newaxis,)*(var.ndim-2)]
    # Latitude must be the second last dimension
    lat_axis = -2 % var.data.ndim
    y0 = {}
    if weighted:
        newaxis_slice = (slice(None),)*lat_axis + (numpy.newaxis,)
        # Northern
        y0['NP'] = (var*latND*(latND > 0)).sum(axis=lat_axis)[newaxis_slice]/\
                   (var*(latND > 0)).sum(axis=lat_axis)[newaxis_slice]
        # Southern
        y0['SP'] = (var*latND*(latND < 0)).sum(axis=lat_axis)[newaxis_slice]/\
                   (var*(latND < 0)).sum(axis=lat_axis)[newaxis_slice]
    else:
        # Northern
        y0['NP'] = lat[(var*(latND > 0)).argmax(axis=lat_axis)]
        # Southern
        y0['SP'] = lat[(var*(latND < 0)).argmax(axis=lat_axis)]

    return y0



def y0(var, lon, lat):
    ''' Find the latitude of the ITCZ/SPCZ where the rainfall is heaviest

    Args:
        var (numpy 2d array): precipitation (lat,lon)
        lon (numpy 1d array): longitudes
        lat (numpy 1d array): latitudes

    Returns:
        dict: {'NP':latitude (scalar) of the ITCZ,
               'SP':latitude (scalar) of the SPCZ}
    '''
    assert var.ndim == 2
    assert var.shape[0] == len(lat)
    assert var.shape[1] == len(lon)
    varxave = var.mean(axis=-1)
    y0 = {}
    # Northern
    y0['NP'] = lat[lat > 0][varxave[lat > 0].argmax()]
    # Southern
    y0['SP'] = lat[lat < 0][varxave[lat < 0].argmax()]
    return y0


def x0(var, lon, lat):
    ''' Find the longitude of the ITCZ/SPCZ where the rainfall is heaviest

    Args:
        var (numpy 2d array): precipitation (lat,lon)
        lon (numpy 1d array): longitudes
        lat (numpy 1d array): latitudes

    Returns:
        dict: {'NP':for northern hemisphere, 'SP':for southern hemisphere}
    '''
    assert var.ndim == 2
    assert var.shape[0] == len(lat)
    assert var.shape[1] == len(lon)
    varave = geodat.stat.runave(var, 10, axis=-1)
    x0 = {}
    # Northern
    varaveNP = varave[lat > 0, :].mean(axis=0)
    x0['NP'] = lon[varaveNP.argmax()]
    # Southern
    varaveSP = varave[lat < 0, :].mean(axis=0)
    x0['SP'] = lon[varaveSP.argmax()]
    return x0


def xave_max(var, lon, lat):
    ''' Compute the zonal average and then return the max for ITCZ and SPCZ

    Args:
        var (numpy 2d array): precipitation (lat,lon)
        lon (numpy 1d array): longitudes
        lat (numpy 1d array): latitudes

    Returns:
        dict: {'NP':for northern hemisphere, 'SP':for southern hemisphere}
    '''
    assert var.ndim == 2
    assert var.shape[0] == len(lat)
    assert var.shape[1] == len(lon)
    varxave = var.mean(axis=-1)
    amp = {}
    # Northern
    amp['NP'] = varxave[lat > 0, :].max()
    # Southern
    amp['SP'] = varxave[lat < 0, :].max()
    return amp


def amp_sum(var, lon, lat):
    ''' Return the sum for the ITCZ and SPCZ

    Args:
        var (numpy 2d array): precipitation (lat,lon)
        lon (numpy 1d array): longitudes
        lat (numpy 1d array): latitudes

    Returns:
        dict("NP"=for northern hemisphere, "SP"=for southern hemisphere)
    '''
    amp = {}
    # Northern
    amp['NP'] = var[lat > 0, :].sum()
    # Southern
    amp['SP'] = var[lat < 0, :].sum()
    return amp


def width(var, lon, lat, thres):
    ''' Find the slopes of the central axes of the ITCZ; rotate the field and
    then find the widths of the ITCZ/SPCZ

    Args:
       var (numpy 2d array): of shape (lat,lon)
       lon (numpy 1d array): longitudes
       lat (numpy 1d array): latitudes
       thres (function): a function that takes precip
             as the only argument, the result is used as the threshold for
             precipitation. e.g. lambda precip: numpy.percentile(precip,70.)

    Returns:
        dict: {'NP':width in degree for the northern hemisphere,
               'SP':same but for the southern hemisphere}

    '''
    # Get the slopes and intercepts for the ITCZ and SPCZ
    central_axes = lines(var, lon, lat, thres)
    widths = {}
    for key, value in central_axes.items():
        slope = value[0]
        theta = numpy.arctan(slope)*-1
        matrix = numpy.array([[numpy.cos(theta), -1*numpy.sin(theta)],
                              [numpy.sin(theta), numpy.cos(theta)]])
        newvar = scipy.ndimage.interpolation.affine_transform(var,matrix)
        newvar_xave = newvar.mean(axis=-1)
        if key == 'NP':
            newvar_xave[lat < 0] = numpy.ma.masked
        else:
            newvar_xave[lat > 0] = numpy.ma.masked

        imax = newvar_xave.argmax()
        iwidth = numpy.argmin(
            numpy.abs(newvar_xave/newvar_xave[imax]
                          - numpy.exp(-1.)))
        widths[key] = numpy.abs(lat[iwidth] - lat[imax])/numpy.sqrt(2.)

    return widths


def analysis(var,thres):
    ''' Perform an analysis on the ITCZ pattern

    Args:
       var (geodat.nc.Variable)
       thres (function): see :py:func:`ENSO.ITCZ.lines` or
          :py:func:`ENSO.ITCZ.width`

    Return:
       dict("NP"=dict(key=['y0','x0','width','slope','y-intercept','amp']),
            "SP" : same as above }
       where y0 is the latitude of the max zonal mean;
          width is the gaussian width of the ITCZ/SPCZ across itself slope;\n
          y-intercept gives the location of the ITCZ/SPCZ as
          lat = y-intercept + slope * lon

    This function requires the `GeoDAT </geodat_doc>`_ library
    '''
    if not _GEODAT_INSTALLED:
        raise _import_geodat_error

    y0_ans = y0(var.data, var.getLongitude(), var.getLatitude())
    width_ans = width(var.data, var.getLongitude(), var.getLatitude(), thres)
    slopes_ans = lines(var.data, var.getLongitude(), var.getLatitude(), thres)
    x0_ans = x0(var.data, var.getLongitude(), var.getLatitude())
    xave_max_ans = xave_max(var.data, var.getLongitude(), var.getLatitude())
    results = {key: dict(y0=y0_ans[key], width=width_ans[key],
                         slope=slopes_ans[key][0],
                         y_intercept=slopes_ans[key][1],
                         x0=x0_ans[key],
                         xave_max=xave_max_ans[key])
               for key in y0_ans.keys()}
    return results

def idealized(amp,x,y,x_loc,y_loc,slope,x_width,y_width,return_Variable=False):
    ''' Return a 2D numpy array or a geodat.nc.Variable with a gaussian profile

    Args:
        amp (numeric): amplitude of the profile
        x (numpy 1d or 2d array): the x axis
        y (numpy 1d or 2d array): the y axis
        x_loc (numeric): x coordinate for the maximum of the gaussian profile
        y_loc (numeric): y coordinate for the maximum of the gaussian profile
        slope (numeric): rotate the gaussian profile so that it makes an angle
                           arctan(slope) with the horizontal axis
        x_width (numeric): width, standard deviation of the profile in the x
                direction (before rotation)
        y_width (numeric): wifth, standard deviation of the profile in the y
                direction (before rotation)
        return_Variable (bool): whether or not a geodat.nc.Variable is returned
              (supported if `GeoDat </geodat_doc>`_ is installed)

    Returns:
        numpy.ndarray or geodat.nc.Variable if return_Variable is True
    '''

    theta = numpy.arctan(slope)
    if x.ndim == 2 and y.ndim == 2:
        assert x.shape == y.shape
    else:
        assert x.ndim == 1 and y.ndim == 1
        x,y = numpy.meshgrid(x,y)

    xP = (x - x_loc)*numpy.cos(theta) + (y - y_loc)*numpy.sin(theta)
    yP = -1.*(x - x_loc)*numpy.sin(theta) + (y - y_loc)*numpy.cos(theta)
    #if amp_sum is not None:
    #    print('amp_sum is given and is used to recalculate amp.')
    #    amp = amp_sum/2.*numpy.pi/y_width/x_width

    q = amp*numpy.exp(-0.5*(xP**2.)/(x_width**2.))*numpy.exp(-0.5*(yP**2.)/\
                                                             (y_width**2.))
    #if amp_sum is not None:
    #    print "q_sum = {:.2f} and amp_sum = {:.2f}".format(q.sum(),amp_sum)
    if return_Variable:
        if not _GEODAT_INSTALLED:
            raise _import_geodat_error

        newvar = geodat.nc.Variable(data=q,varname="Q",
                                    dims=[
                                        geodat.nc.Dimension(data=y[:,0],
                                                            dimname='LAT',
                                                            units='degrees_N'),
                                        geodat.nc.Dimension(data=x[0,:],
                                                            dimname='LON',
                                                            units='degrees_E')])
        return newvar
    else:
        return q


def y_max(precip,lat,yaxis=-2,ave_axis=-1):
    var_ave = geodat.keepdims.mean(precip,axis=ave_axis)
    return lat[var_ave.argmax(axis=yaxis)].squeeze()
