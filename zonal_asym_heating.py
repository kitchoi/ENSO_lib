import util
import numpy
import pylab

def zonal_dev(var):
    ''' Return zonal asymmetric component
    Input: var - a util.nc.Variable instance
    '''
    var.ensureMasked()
    return var - var.zonal_ave()

def dotproduct(var1,var2):
    ''' Return the dotproduct of two variables.
    Dot Product = Area average of (Var1 * Var2)
    Var1 and Var2 are assumed to be (lat,lon)
    Input: var1,var2 - util.nc.Variable instances
    Output: Scalar (masked)
    '''
    var1.ensureMasked()
    var2.ensureMasked()
    return (var1 * var2).wgt_ave().data.squeeze()

def norm(var):
    ''' Return the norm of a variable
    Norm = root-mean-square value of the field
    Input: var - a util.nc.Variable instance
    Output: Scalar (masked)
    '''
    absvar = dotproduct(var,var)
    absvar = numpy.sqrt(dotproduct(var,var))
    return absvar.squeeze()


def normalize_abs(var):
    ''' Return a normalized field
    Input: var - a util.nc.Variable instance
    Output: var/norm(var)
    '''
    var.ensureMasked()
    return var / norm(var)


def mask_per(var,per):
    ''' Absolute values smaller than per*range_of_var/2. will be masked away
    Input:
    var - util.nc.Variable
    per - percentage
    '''
    var.ensureMasked()
    small = per_range(var.data,per)
    return mask_small(var,small)

def mask_small(var,small):
    ''' Absolute values smaller than the value "small" will be masked away
    Input: 
    var - util.nc.Variable
    small - a positive scalar
    '''
    if small < 0.: small = numpy.abs(small)
    var.data = numpy.ma.masked_inside(var.data,-1*small,small)
    return var

def project(var1,var2):
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

