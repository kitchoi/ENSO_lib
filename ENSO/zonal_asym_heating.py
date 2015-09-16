import numpy
import pylab

import geodat

def zonal_dev(var):
    ''' Return zonal asymmetric component
    Input: var - a geodat.nc.Variable instance
    '''
    var.ensureMasked()
    var2 = var - var.zonal_ave()
    var2.addHistory('Zonal asymmetric part')
    return var2

def dotproduct(var1,var2):
    ''' Return the dotproduct of two variables.
    Dot Product = Area average of (Var1 * Var2)
    Var1 and Var2 are assumed to be (lat,lon)
    Input: var1,var2 - geodat.nc.Variable instances
    Output: Scalar (masked)
    '''
    var1.ensureMasked()
    var2.ensureMasked()
    return (var1 * var2).wgt_ave().data.squeeze()

def norm(var):
    ''' Return the norm of a variable
    Norm = root-mean-square value of the field
    Input: var - a geodat.nc.Variable instance
    Output: Scalar (masked)
    '''
    absvar = dotproduct(var,var)
    absvar = numpy.sqrt(dotproduct(var,var))
    return absvar.squeeze()


def normalize_abs(var):
    ''' Return a normalized field
    Input: var - a geodat.nc.Variable instance
    Output: var/norm(var)
    '''
    var.ensureMasked()
    return var / norm(var)


def mask_per(var,per):
    ''' Absolute values smaller than per*range_of_var/2. will be masked away
    Input:
    var - geodat.nc.Variable
    per - percentage
    '''
    var.ensureMasked()
    small = per_range(var.data,per)
    newvar = geodat.nc.Variable(data=mask_small_numpy(var.data,small),parent=var,
                              history='Masked '+str(per)+' percentage of range')
    
    return newvar

def mask_small(var,small):
    ''' Absolute values smaller than the value "small" will be masked away
    Input: 
    var - an geodat.nc.Variable
    small - a positive scalar
    '''
    var.ensureMasked()
    newvar = geodat.nc.Variable(data=mask_small_numpy(var.data,small),
                              parent=var,
                              history='Masked abs(value) < '+str(small))
    return newvar

def mask_small_numpy(data,small):
    ''' Absolute values smaller than the value "small" will be masked away
    Input: 
    data - numpy masked array
    small - a positive scalar
    '''
    assert isinstance(data,numpy.ma.core.MaskedArray)
    newdata = numpy.ma.array(data)
    assert (newdata.mask == data.mask).all()
    assert id(newdata) != id(data)
    if small < 0.: small = numpy.abs(small)
    print small
    newdata = numpy.ma.masked_inside(newdata,-1*small,small)
    return newdata

def project(var1,var2):
    ''' Return the projection of var1 onto var2
    '''
    var1.ensureMasked()
    var2.ensureMasked()
    # To avoid multiplying and averaging very small values
    # Basically doing norm(var1*var2)/norm(var2*var2)
    factor = numpy.min([norm(var1),norm(var2)])
    sign = numpy.sign(numpy.mean(var1.data * var2.data))
    return sign * norm((var1/factor) * (var2/factor)) \
       / norm((var2/factor) * (var2/factor))

#  Return percentage*range/2
def per_range(data,per):
    return (data.max() - data.min())/2.*per/100. 

# Procedures
def xdev_field(Q):
    def xdev_oper(xdev):
        d = {}
        d['normalized'] = normalize_abs(xdev)
        d['norm'] = norm(xdev)
        return d
    
    xdev = {}
    xdev_data = zonal_dev(Q)
    xdev['Whole'] = xdev_oper(xdev_data)
    xdev['50%filtered'] = xdev_oper(mask_per(xdev_data,50.))
    xdev['5%filtered'] = xdev_oper(mask_per(xdev_data,5.))
    return xdev

