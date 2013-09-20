import numpy
import util
import pylab

def max_anom(u,lon_width=40.,lat=(-5.,5.)):
    ''' Return the maximum U after applying a lon_width degree 
    running mean along the longitude.  Assumed uniform grid.
    Input:
    u - util.nc.Variable
    lon_width - scalar in degree (default = 40.)
    lat - region in the latitude (default = (-5.,5.))
    '''
    nlon = numpy.round(lon_width/numpy.diff(u.getLongitude()[0:2])[0])
    runaveu = u.getRegion(lat=lat).wgt_ave('Y').runave(nlon,'X').data.squeeze()
    return runaveu[numpy.abs(runaveu) == numpy.abs(runaveu).max()][0]

def plot_r(y,x,*args,**kwargs):
    ''' Read a list of variable, apply maxu_anom to each of them
    to find the maximum.  Plot the results and compute the value of r using
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
    print x,y
    s_neg = numpy.polyfit(x[x<=0],y[x<=0],1)[0]
    s_pos = numpy.polyfit(x[x>=0],y[x>=0],1)[0]

    pylab.plot(numpy.sort(x),numpy.sort(y),*args,**kwargs)
    pylab.scatter(x,y)    
    return (s_pos - s_neg)/(s_pos + s_neg)


    
