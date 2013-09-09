import util
import numpy
import pylab

def zonal_dev(var):
    var.ensureMasked()
    return var - var.zonal_ave()

def dotproduct(var1,var2):
    var1.ensureMasked()
    var2.ensureMasked()
    return (var1 * var2).wgt_ave().data.squeeze()

def norm(var):
    absvar = dotproduct(var,var)
    absvar = numpy.sqrt(dotproduct(var,var))
    return absvar.squeeze()


def normalize_abs(var):
    var.ensureMasked()
    return var / norm(var)


def mask_per(var,per):
    ''' Absolute values smaller than per*range_of_var/2. will be masked away
    '''
    var.ensureMasked()
    small = per_range(var.data,per)
    return mask_small(var,small)

def mask_small(var,small):
    ''' Absolute values smaller than the value "small" will be masked away
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
    sign = numpy.sign(numpy.corrcoef(var1.data,var2.data))
    return sign*norm((var1/factor) * (var2/factor)) \
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

