import numpy
import pylab

import geodat
import geodat.plot

def NINO34_Pattern(field,nino34,factors,nino34_ref,nino34_tole):
    ''' Given a field and a nino3.4 index, group the snapshot of the
    field such that the instanteous nino3.4 index is within
    nino34_ref*factor - nino34_tole < nino34 < nino34_ref*factor
    + nino34_tole
    Input
    field  - a geodat.nc.Variable of shape: (TIME,...)
    nino34 - a geodat.nc.Variable of shape: (TIME)
    factors - a dictionary { casename : factor }
    nino34_ref - a scalar
    nino34_tole - a scalar
    Output
    a dictionary { casename : grouped_field(TIME,...) }
    Example: NINO34_Pattern(data['precip_a'],data['nino34'],factors,1.6,0.25)
    '''
    import ENSO.diag
    return { casename : ENSO.diag.find_EN_pattern(field,nino34,
                                                  nino34_ref*factor,nino34_tole)
             for casename,factor in factors.items() }


def plot_Patterns(pattern,name,prefix,root='../'):
    ''' Input
    pattern - a dictionary with casename as keys
      name    - how would you call the pattern, will appear in the file name
      prefix  - e.g. model
      Output
      Print to root/figures/prefix.name_casename.eps
      '''
    for casename in factors.keys():
        pylab.figure(); geodat.nc.contourf((pattern[casename]).time_ave());
        pylab.colorbar(orientation='horizontal')
        pylab.title(pattern[casename].getattr('long_name'))
        pylab.colorbar()
        pylab.savefig(root+'figures/'+prefix+'.'+name+'_'+casename+'.eps')

def Find_x_up_dn(div_u):
    nlon = numpy.round(10./numpy.diff(div_u.getLongitude()[0:2])[0])
    div_u_ave = div_u.runave(nlon,'X').getRegion(lat=(-5.,5.)).wgt_ave('Y').squeeze()  # (Time,Lon)
    tmp = div_u_ave.getRegion(lon=(120.,300.))
    xiax = tmp.getCAxes().index('X')
    x_up_loc = numpy.argmin(tmp.data,axis=xiax)
    x_dn_loc = numpy.argmax(tmp.data,axis=xiax)
    x_up = numpy.ma.array(tmp.getLongitude()[x_up_loc])
    x_dn = numpy.ma.array(tmp.getLongitude()[x_dn_loc])    
    x_up.mask = numpy.all(tmp.data.mask,axis=xiax)
    x_dn.mask = numpy.all(tmp.data.mask,axis=xiax)
    return x_up,x_dn


def Estimate_u_star(precip,sphum,div_u):
    x_up,x_dn = Find_x_up_dn(div_u)
    L_x = x_up - x_dn
    p_prime = (precip.getRegion(lon=x_up,lat=(-5.,5.)) - precip.getRegion(lon=x_dn,lat=(-5.,5.))).squeeze()
    if 'Z' in sphum.getCAxes():
        q_prime = geodat.nc.integrate(sphum.getRegion(lon=x_up,level=slice(-3,None)),sphum.getCAxes().index('Z')).squeeze() + geodat.nc.integrate(sphum.getRegion(lon=x_dn,level=slice(-3,None)),sphum.getCAxes().index('Z')).squeeze()
    else: 
        q_prime = (sphum.getRegion(lon=x_up) + sphum.getRegion(lon=x_dn)).squeeze()
    q_prime = q_prime*100./10.
    q_prime.ensureMasked()
    print (p_prime.data.shape,q_prime.data.shape)
    assert p_prime.data.shape == q_prime.data.shape
    u_star = p_prime*L_x/2./q_prime*110000.
    return u_star,L_x,p_prime,q_prime

def Estimate_u_star_Seasonal(precip,sphum,div_u):
    u_star = numpy.ma.zeros(precip.dims[0].data.shape)
    L_x = numpy.ma.zeros(precip.dims[0].data.shape)
    p_prime = numpy.ma.zeros(precip.dims[0].data.shape)
    q_prime = numpy.ma.zeros(precip.dims[0].data.shape)
    for itime in range(len(u_star)):
        #u_star[itime], L_x[itime], p_prime[itime], q_prime[itime]
        result = \
                 Estimate_u_star(
                     precip[itime].getRegion(lat=(-5.,5.)).wgt_ave('Y'),
                     sphum[itime].getRegion(lat=(-5.,5.)).wgt_ave('Y'),
                     div_u[itime])
        u_star[itime] = result[0].data
        L_x[itime] = result[1]
        p_prime[itime] = result[2].data
        q_prime[itime] = result[3].data
    
    return u_star,L_x,p_prime,q_prime

def plot_estimatedU_VS_true(U_estimate,U_pattern,title=''):
    import ENSO.maxu_r
    tmp = [ (factors[casename], U_estimate[casename]['u_star'].mean())
        for casename in factors.keys() ]
    y = [ a[1] for a in tmp ]
    x = [ a[0] for a in tmp ]
    tmp2 = [ (factors[casename], ENSO.maxu_r.max_anom(U_pattern[casename]))
        for casename in factors.keys() ]
    y_true = [ a[1] for a in tmp2 ]
    x_true = [ a[0] for a in tmp2 ]
    # pylab.figure()
    r_true = ENSO.maxu_r.plot_r(y_true,x_true)
    r = ENSO.maxu_r.plot_r(y,x)
    pylab.legend(['Max U anom, r = '+str(numpy.round(r_true,2)),
                    'Estimated U, r = '+str(numpy.round(r,2))],loc='best')
    pylab.title(title)
    pylab.xlabel('Factor')
    pylab.ylabel('U anomaly [m/s]')
    pylab.show()


def plot_dictionary(xaxis,var,sortedkeys=None,
                    yfunc=lambda v: v,xfunc=lambda v:v,
                    legend=True,**kwargs):
    '''
    Required:
    xaxis - a dictionary of numpy array (x)
    var - a dictionary of numpy array (y)
    
    Optional:
    sortedkeys  - a list used for sorting the key of var, if None, sortedkey = var.keys()
    yfunc      - a function to be applied to var while plotting
    xfunc      - a function to be applied to xaxis while plotting
    
    Keyword arguments to pylab.plot are applied here.
    If the value of a keyword argument is a dictionary, it is matched with the dictionary var.
    Otherwise it is assumed general for all lines.
    '''
    if sortedkeys is None: sortedkeys = var.keys()
    for key in sortedkeys:
        kw = { arg: kwargs[arg][key] if type(kwargs[arg]) is dict else kwargs[arg]
               for arg in kwargs.keys() }
        pylab.plot(xfunc(xaxis[key]),yfunc(var[key]),label=key,**kw)
    
    geodat.plot.template.default()
    if legend: l = geodat.plot.reorderlegend(sortedkeys)
    
    #l.set(prop={'size':10})

def plot_txave(var,factors,func=lambda v: v.wgt_ave('TX').squeeze(),
               xfunc=lambda v:v.getLatitude(),**kwargs):
    sortedkeys = sorted(factors.keys(),key=lambda t:factors[t])
    linesty = { key: '-' if numpy.sign(factors[key]) > 0 else '--' for key in sortedkeys }
    plot_dictionary(var,var,xfunc=lambda v: xfunc(func(v)),
                    yfunc=lambda v: func(v).data,sortedkeys=sortedkeys,
                    linestyle=linesty,**kwargs)
    pylab.xlabel('Latitude [degree]')

def plot_tyave(var,factors,func=lambda v: v.wgt_ave('TY').squeeze(),
               xfunc=lambda v:v.getLongitude(),**kwargs):
    sortedkeys = sorted(factors.keys(),key=lambda t:factors[t])
    if kwargs.has_key('linestyle') is False:
        kwargs['linestyle'] = { key: '-' if numpy.sign(factors[key]) > 0 else '--' for key in sortedkeys }
    plot_dictionary(var,var,xfunc=lambda v: xfunc(func(v)),
                    yfunc=lambda v: func(v).data,sortedkeys=sortedkeys,
                    **kwargs)
    pylab.xlabel('Longitude [degree]')

def save4gill(field,filename,NY=256):
    '''
    field - an instance of geodat.nc.Variable
    '''
    import sphere_grid.grid_func as sphere_grid
    ref_var = geodat.nc.getvar('../input/t170/gill.nc','heating')
    regridded = geodat.nc.pyferret_regrid(field,ref_var,'XY')#2*NY,NY)
    regridded.varname = 'Q'
    regridded.attributes['history'] = '' # remove history
    regridded.ensureMasked()
    regridded.data[regridded.data.mask] = 0.
    geodat.nc.savefile(filename,regridded,overwrite=True)


def proper_filename(casename):
    import re
    
    newname = re.sub(r"[0-9]\.[0-9]",
                     lambda m: m.group(0)[0]+'p'+m.group(0)[-1],
                     casename)
    return newname


def dudx_qc(u,q,dx):
    conform_region = geodat.nc.conform_region(u,q)
    def conform(var):
        var.data[var.data.mask] = 0.
        var = geodat.nc.regrid(var,360,180)
        return var.getRegion(**conform_region)
    u = conform(u)
    q = conform(q)
    return geodat.nc.gradient(u,dx).runave(2,dx)*q

def region_regrid(var,region,nlon=360,nlat=180):
    var.data[var.data.mask] = 0.
    var = geodat.nc.regrid(var,nlon,nlat).getRegion(**region)
    return var

def conform(var1,var2,nlon=360,nlat=180):
    region = geodat.nc.conform_region(var1,var2)
    var1 = region_regrid(var1,region,nlon,nlat)
    var2 = region_regrid(var2,region,nlon,nlat)
    return var1,var2


def normalize_ax(var,ref,ax):
    ''' Normalize variable (var) with the magnitude of the reference field (ref)
    averaged along the axis ax
    Input:
    var -- geodat.nc.Variable
    ref -- geodat.nc.Variable
    ax  -- string. e.g.: "TY","Y","X"
    '''
    return var/var.wgt_ave(ax)*ref.wgt_ave(ax)


def normalize_ax_anom(var,ref,ax,ref_profile=None):
    ''' var is where you take the profile from
    ref is where you take the magnitude from
    '''
    if ref_profile is None: ref_profile = var
    var_norm = normalize_ax(ref_profile,ref,ax)
    var_norm_anom = var_norm - ref
    var_norm_anom_diff = var - var_norm
    return var_norm,var_norm_anom,var_norm_anom_diff

def xyxnorm_anom(var,ref,ax='xyx',ref_profile=None,npart=50,nstop=1000,
                 animated=False,wait=1,animate_func=None,tolerance=0.01):
    ''' Perform xyx or yxy incremental normalize_ax
    Return var_norm,var_norm_anom,var_norm_anom_diff,anom1,anom2
    var_norm: Normalized var
    var_norm_anom: Anomalies that add to zero
    var_norm_diff: the rest of the anomalies
    anom1: anomaly corresponding to normalization along ax[0]
    anom2: anomaly corresponding to normalization along ax[1]
    '''
    import numpy
    import pylab
    if animate_func is None:
        def animate(var,wait=wait):
            import time
            geodat.nc.contourf(var)
            pylab.show()
            time.sleep(wait)
    else:
        animate=animate_func
    
    if ref_profile is None: ref_profile = var
    axis1 = ax[0].upper()
    axis2 = ax[1].upper()
    axis3 = ax[2].upper()
    assert axis1 == axis3
    # first step
    if animated: animate(ref)
    var_norm = normalize_ax(ref_profile,ref,axis1)
    anom1 = (var_norm - ref)/float(npart)/2.
    var_norm = ref + anom1
    if animated: animate(var_norm)
    var_norm_new = normalize_ax(ref_profile,var_norm,axis2)
    anom2 = (var_norm_new - var_norm)/float(npart)
    var_norm += anom2
    if animated: animate(var_norm)
    anom = anom2
    nstep = 1
    while anom.data.std() > anom1.data.std()/npart*tolerance or \
            anom.data.std() > anom2.data.std()/npart*tolerance :
        #for i in xrange(npart-1):
        var_norm_new = normalize_ax(ref_profile,var_norm,axis1)
        anom = (var_norm_new - var_norm)/float(npart)
        anom1 += anom 
        var_norm += anom
        if animated: animate(var_norm)
        var_norm_new = normalize_ax(ref_profile,var_norm,axis2)
        anom = (var_norm_new - var_norm)/float(npart)
        anom2 += anom
        var_norm += anom
        if animated: animate(var_norm)
        nstep += 1
        if nstep > nstop:
            print "anom std:"+str(anom.data.std()*npart)
            print "cumulative anom1 std:"+str(anom1.data.std())
            print "cumulative anom2 std:"+str(anom2.data.std())
            raise Exception("Failed to converge within "+str(nstop)+" iterations.")
    # Converged!
    print "Converged after "+str(nstep)+" iterations."
    print "anom std:"+str(anom.data.std()*npart)
    print "cumulative anom1 std:"+str(anom1.data.std())
    print "cumulative anom2 std:"+str(anom2.data.std())
    var_norm_new = normalize_ax(ref_profile,var_norm,axis1)
    anom = (var_norm_new - var_norm)/float(npart)/2.
    var_norm += anom
    if animated: animate(var_norm)
    anom1 += anom
    var_norm_anom = var_norm - ref
    var_norm_anom_diff = var - var_norm
    return var_norm,var_norm_anom,var_norm_anom_diff,anom1,anom2   


def yshuffle_rank(var,ref):
    ''' Given the variable (var), order the field along Y for each longitude and time,
    replace the grid with the value from the reference field (ref) that has the same ranking
    along Y.
    Assume the variable has a structure of (TIME,LAT,LON)
    '''
    assert var.getCAxes()[0] == 'T'
    assert var.getCAxes()[1] == 'Y'
    assert var.getCAxes()[2] == 'X'
    import numpy
    import warnings
    rearranged = numpy.ma.zeros(var.data.shape)
    lon,lat = numpy.meshgrid(var.getLongitude(),var.getLatitude())
    coslat = numpy.cos(numpy.radians(lat))
    if isinstance(ref.data,numpy.ma.core.MaskedArray):
        warnings.warn("Reference field has masked value, "+\
                      "they are replaced by zero within this routine")
        ref.data[ref.data.mask] = 0.
    if isinstance(var.data,numpy.ma.core.MaskedArray):
        warnings.warn("Variable field has masked value, "+\
                      "they are replaced by zero within this routine")        
        var.data[var.data.mask] = 0.
    
    lat = var.getLatitude()
    # for loop for the first and last axis
    for itime in range(rearranged.data.shape[0]):
        for ilon in range(rearranged.data.shape[-1]):
            newpos = numpy.argsort(numpy.argsort(var.data[itime,:,ilon],axis=None),axis=None)
            rearranged[itime,:,ilon] = numpy.sort(ref.data[itime,:,ilon],axis=None)[newpos].reshape(var.data[itime,:,ilon].shape)
            oldcoslat = coslat[:,ilon].ravel()[newpos].reshape(var.data[itime,:,ilon].shape)
            #ratio = oldcoslat/coslat[:,ilon]
            #rearranged[itime,:,ilon] *= ratio
    
    return geodat.nc.Variable(data=rearranged,parent=var,varname=var.varname+'_y_shuffled',history='Shuffled in y direction using '+ref.varname)


def yshuffle_rank_anom(var,ref,ref_pos=None,ref_anom=None):
    ''' 
    var: the original anomaly that the shuffling tries to achieve
    ref: the reference precipitation where shuffling takes the values
    ref_pos: the precipitation/temperature field from which the rank is computed
    ref_anom: the reference precipitation from which the anomaly refers to
    '''
    if ref_anom is None: ref_anom = ref
    if ref_pos is None: ref_pos = var
    var_norm = yshuffle_rank(ref_pos,ref)
    var_norm_anom = var_norm - ref_anom
    var_norm_anom_diff = var - var_norm
    return var_norm,var_norm_anom,var_norm_anom_diff


def xshuffle_rank(var,ref):
    # for loop for the first and second axis
    import numpy
    rearranged = numpy.ma.zeros(var.data.shape)
    lon,lat = numpy.meshgrid(var.getLongitude(),var.getLatitude())
    coslat = numpy.cos(numpy.radians(lat))
    if isinstance(ref.data,numpy.ma.core.MaskedArray):
        ref.data[ref.data.mask] = 0.
    if isinstance(var.data,numpy.ma.core.MaskedArray):
        var.data[var.data.mask] = 0.
    
    for itime in range(rearranged.data.shape[0]):
        for ilat in range(rearranged.data.shape[1]):
            newpos = numpy.argsort(numpy.argsort(var.data[itime,ilat,:],axis=None),axis=None)
            rearranged[itime,ilat,:] = numpy.sort(ref.data[itime,ilat,:],axis=None)[newpos].reshape(var.data[itime,ilat,:].shape)
            # correction for latitude difference is not needed
            #oldcoslat = coslat.ravel()[newpos].reshape(var.data[itime,...].shape)
            #ratio = oldcoslat/coslat
            #rearranged[itime,...] *= ratio
    
    return geodat.nc.Variable(data=rearranged,parent=var,varname=var.varname+'_x_shuffled',history='Shuffled in x direction using '+ref.varname)

def xshuffle_rank_anom(var,ref,ref_pos=None,ref_anom=None):
    ''' 
    var: the original anomaly that the shuffling tries to achieve
    ref: the reference precipitation where shuffling takes the values
    ref_pos: the precipitation/temperature field from which the rank is computed
    ref_anom: the reference precipitation from which the anomaly refers to
    '''
    if ref_anom is None: ref_anom = ref
    if ref_pos is None: ref_pos = var
    var_norm = xshuffle_rank(ref_pos,ref)
    var_norm_anom = var_norm - ref_anom
    var_norm_anom_diff = var - var_norm
    return var_norm,var_norm_anom,var_norm_anom_diff


def yshuffle_xshuffle_rank_anom(var,ref,ref_pos=None,ref_anom=None):
    ''' 
    var: the original anomaly that the shuffling tries to achieve
    ref: the reference precipitation where shuffling takes the values
    ref_pos: the precipitation/temperature field from which the rank is computed
    ref_anom: the reference precipitation from which the anomaly refers to
    '''
    if ref_anom is None: ref_anom = ref
    if ref_pos is None: ref_pos = var
    var_norm = yshuffle_rank(ref_pos,ref)
    var_norm = xshuffle_rank(ref_pos,var_norm)
    var_norm_anom = var_norm - ref_anom
    var_norm_anom_diff = var - var_norm
    return var_norm,var_norm_anom,var_norm_anom_diff


def xshuffle_yshuffle_rank_anom(var,ref,ref_pos=None,ref_anom=None):
    ''' 
    var: the original anomaly that the shuffling tries to achieve
    ref: the reference precipitation where shuffling takes the values
    ref_pos: the precipitation/temperature field from which the rank is computed
    ref_anom: the reference precipitation from which the anomaly refers to
    '''
    if ref_anom is None: ref_anom = ref
    if ref_pos is None: ref_pos = var
    var_norm = xshuffle_rank(ref_pos,ref)
    var_norm = yshuffle_rank(ref_pos,var_norm)
    var_norm_anom = var_norm - ref_anom
    var_norm_anom_diff = var - var_norm
    return var_norm,var_norm_anom,var_norm_anom_diff


def xyxshuffle_rank_anom(var,ref,ref_pos=None,ref_anom=None,nitn=2):
    ''' 
    var: the original anomaly that the shuffling tries to achieve
    ref: the reference precipitation where shuffling takes the values
    ref_pos: the precipitation/temperature field from which the rank is computed
    ref_anom: the reference precipitation from which the anomaly refers to
    '''
    if ref_anom is None: ref_anom = ref
    if ref_pos is None: ref_pos = var
    # first step
    var_norm = xshuffle_rank(ref_pos,ref)
    anom = (var_norm - ref_anom)/float(nitn)/2.
    var_norm = ref + anom
    xanom = anom
    var_norm_new = yshuffle_rank(ref_pos,var_norm)
    anom = (var_norm_new - var_norm)/float(nitn)
    var_norm += anom
    yanom = anom
    for i in xrange(1,nitn):
        var_norm_new = xshuffle_rank(ref_pos,var_norm)
        anom = (var_norm_new - var_norm)/float(nitn)
        var_norm += anom
        xanom += anom
        var_norm_new = yshuffle_rank(ref_pos,var_norm)
        anom = (var_norm_new - var_norm)/float(nitn)
        var_norm += anom
        yanom += anom
    # last step
    var_norm_new = xshuffle_rank(ref_pos,var_norm)
    anom = (var_norm_new - var_norm)/float(nitn)/2.
    var_norm += anom
    xanom += anom
    
    var_norm_anom = var_norm - ref_anom
    var_norm_anom_diff = var - var_norm
    return var_norm,var_norm_anom,var_norm_anom_diff



def shuffle_rank(var,ref):
    ''' Given the variable (var), order the field within the spatial domain for each time,
    replace the grid with the value from the reference field (ref) that has the same ranking
    in the spatial domain.
    Assume the variable has a structure of (TIME,LAT,LON)
    '''
    assert var.getCAxes()[0] == 'T'
    assert var.getCAxes()[1] == 'Y'
    assert var.getCAxes()[2] == 'X'    
    import numpy
    rearranged = numpy.ma.zeros(var.data.shape)
    lon,lat = numpy.meshgrid(var.getLongitude(),var.getLatitude())
    coslat = numpy.cos(numpy.radians(lat))
    if isinstance(ref.data,numpy.ma.core.MaskedArray):
        ref.data[ref.data.mask] = 0.
    if isinstance(var.data,numpy.ma.core.MaskedArray):
        var.data[var.data.mask] = 0.
    
    for itime in range(rearranged.data.shape[0]):
        newpos = numpy.argsort(numpy.argsort(var.data[itime,...],axis=None),axis=None)
        rearranged[itime,...] = numpy.sort(ref.data[itime,...],axis=None)[newpos].reshape(var.data[itime,...].shape)
        oldcoslat = coslat.ravel()[newpos].reshape(var.data[itime,...].shape)
        coslat_new = coslat.ravel().reshape(var.data[itime,...].shape)
        #ratio = oldcoslat/coslat_new
        #rearranged[itime,...] *= ratio
    return geodat.nc.Variable(data=rearranged,parent=var,varname=var.varname+'_shuffled',history='Shuffled using '+ref.varname)

def shuffle_rank_anom(var,ref,ref_pos=None,ref_anom=None):
    ''' 
    var: the original anomaly that the shuffling tries to achieve
    ref: the reference precipitation where shuffling takes the values
    ref_pos: the precipitation/temperature field from which the rank is computed
    ref_anom: the reference precipitation from which the anomaly refers to
    '''
    if ref_anom is None: ref_anom = ref
    if ref_pos is None: ref_pos = var
    var_norm = shuffle_rank(ref_pos,ref)
    var_norm_anom = var_norm - ref_anom
    var_norm_anom_diff = var - var_norm
    return var_norm,var_norm_anom,var_norm_anom_diff


def movement(var,delta,ax):
    ''' Use numpy.roll to shift the variable (var) along axis (ax) by delta
    '''
    if ax in var.getCAxes():
            iaxis = var.getCAxes().index(ax)
    else:
        if ax.upper() == 'X':
            iaxis = -1
        elif ax.upper() == 'Y':
            iaxis = -2
        else:
            raise Exception("Unknown axis identifier:"+str(ax))
    
    axis = var.dims[iaxis].data
    ndelta = int(delta/numpy.abs(numpy.diff(axis)).mean())
    newdata = numpy.roll(var.data,ndelta,iaxis)
    return geodat.nc.Variable(data=newdata,parent=var,history="Rolled along axis "+ax+" by "+str(delta))


def threshold_region(t,land_mask,t_thres=27.,
                     ITCZ_region=dict(lat=(-20.,20.),lon=(100.,270.))):
    ''' Given the temperature field, apply land mask and return the location 
    where the temperature is higher than t_thres(default=27.), within the 
    ITCZ_region
    '''
    apply_land_mask = lambda v,region,mask: \
                      v.getRegion(**region).apply_mask(mask.getRegion(**region).data>0.)
    t = apply_land_mask(t,ITCZ_region,land_mask)
    t.data[t.data.mask] = 0.
    lon,lat = numpy.meshgrid(t.getLongitude(),t.getLatitude())
    return t.data>(273.13+t_thres),lon,lat


def find_movement(t_total,tclim,land_mask,t_thres=27.,
                  ITCZ_region=dict(lat=(-20.,20.),lon=(100.,270.))):
    ''' Compute the mean longitude where the total temperature exceeds t_thres (default=27C),
    do the same with the climatological temperature and then compute the difference,
    i.e. zonal shift of precipitating region
    '''
    t_region,lon,lat = threshold_region(t_total,land_mask,t_thres,ITCZ_region)
    tclim_region,lon,lat = threshold_region(tclim,land_mask,t_thres,ITCZ_region)
    lontmp = numpy.ma.sum(lon[numpy.newaxis,:,:]*(t_region),axis=-1)/\
             numpy.ma.sum(t_region,axis=-1)
    lonclm = numpy.ma.sum(lon[numpy.newaxis,:,:]*(tclim_region),axis=-1)/\
             numpy.ma.sum(tclim_region,axis=-1)
    return lontmp - lonclm

def fill_precip(temp,precip_c,land_mask,temp_thres,region):
    ''' Given the temperature field (temp) and the climatological precip (precip_c),
    find the precipitating region defined by a temperature threshold using the 
    temperature function. Fill the region with the average precipitation from the
    climatology.  Return the constructed precipitation.
    '''
    # find region of precipitating temperature
    precip_region,lon,lat = threshold_region(temp,land_mask,temp_thres,region)
    ones = geodat.nc.Variable(data=numpy.ones(precip_region.shape),
                            parent=temp.getRegion(**region))
    total_area= ones.wgt_ave(axis=[-2,-1])
    ones[~precip_region] = 0.
    precip_area = ones.wgt_ave(axis=[-2,-1])
    # mask away missing months
    precip_area[temp.wgt_ave(axis=[-2,-1]).squeeze().data.mask] = numpy.ma.masked
    ave_pclim = precip_c.getRegion(**region).wgt_ave(axis=[-2,-1])
    new_p_amp = ave_pclim
    # fill the precipitating region
    ones = ones*new_p_amp
    return ones

def nonlinearity(d):
    ''' dictionary={'1.5xEN':1.0,'Neg1.5xEN':-1.0}
    return {'1.5xEN':(1+(-1))=0.}
    '''
    result = {}
    for key in d.keys():
        if key[0] != 'N':
            r = d[key]+d['Neg'+key]
            result[key] = r
    return result


def gilloutput_plain_max(panel,model,field,ref_panel='P',normalize=False):
    import numpy
    import ENSO.maxu_r
    assert ref_panel in gilloutput.keys()
    max_u_value = { casename: ENSO.maxu_r.max_anom(
            gilloutput[ref_panel][model][casename][field],lon_width=20.)
                  for casename in gilloutput['P'][model].keys() }
    max_u_loc = { casename: ENSO.maxu_r.max_anom_loc(
            gilloutput[ref_panel][model][casename][field],lon_width=20.)
                  for casename in gilloutput['P'][model].keys() }
    if normalize:
        #norm = numpy.abs(numpy.array([max_u_value[casename] 
        #                              for casename in gilloutput['P'][model].keys()])).mean()
        norm = numpy.abs(max_u_value['1.0xEN'])
    else:
        norm = 1.
    
    max_u = { casename: (gilloutput[panel][model][casename][field].\
                  getRegion(lon=(max_u_loc[casename]-10.,max_u_loc[casename]+10.),
                            lat=(-2.,2.)).wgt_ave().squeeze().data)/norm
              for casename in gilloutput[panel][model].keys() }
    
    return max_u

def gilloutput_nonlinear(panel,model,field,ref_panel='P',normalize=False):
    max_u = gilloutput_plain_max(panel,model,field,ref_panel,normalize)
    
    return nonlinearity(max_u)


def NDJF(var):
    ''' Return geodat.nc.TimeSlices(var,11.,2.,'m')
    '''
    return geodat.nc.TimeSlices(var,11.,2.,'m')


def MJJA(var):
    ''' Return geodat.nc.TimeSlices(var,5.,8.,'m')
    '''
    return geodat.nc.TimeSlices(var,5.,8.,'m')
