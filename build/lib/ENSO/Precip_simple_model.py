
def Precip_anom_simple(p,lon_transit=180.):
    ''' Given a regional precipitation (assumed to be within the Pacific in this context)
    Return a dictionary D
    D['P_y_max'] = the maximum zonal mean precipitation with a 4-degree running mean filter along the Y axis
    D['P_y_0']   = zonal mean precipitation from -2S - 2N
    D['P_E']
    '''
    import numpy
    P_y = p.wgt_ave(''.join([ax for ax in p.getCAxes() if ax != 'Y'])).squeeze()
    P_x = p.getRegion(lat=(-2.,2.)).\
        wgt_ave(''.join([ax for ax in p.getCAxes() if ax != 'X'])).squeeze()
    results = {}
    results['P_y_max'] = P_y.runave(4.,'Y').data.max()
    results['P_y_0'] = P_y.runave(4.,'Y').data[numpy.abs(P_y.getLatitude()).argmin()]
    results['P_E'] = P_x.data[P_x.getLongitude()>lon_transit].mean()
    results['P_W'] = P_x.data[P_x.getLongitude()<lon_transit].mean()
    return results

def P_model_1(p,lon_transit=180.):
    Ps = Precip_anom_simple(p,lon_transit)
    P_EN = (Ps['P_y_max']-Ps['P_y_0'])*2.
    P_LN = Ps['P_E']
    r_P = (P_EN-P_LN)/(P_EN+P_LN)
    return r_P

def P_model_2(p,lon_transit=180.):
    Ps = Precip_anom_simple(p,lon_transit)
    P_EN = (Ps['P_W']-Ps['P_E'])*2.
    P_LN = Ps['P_E']
    r_P = (P_EN-P_LN)/(P_EN+P_LN)
    return r_P


def r_P(pwarm,pcold,lon_transit=180.):
    Ps_warm = Precip_anom_simple(pwarm,lon_transit)
    Ps_cold = Precip_anom_simple(pcold,lon_transit)
    T_EN = Ps_warm['P_E']-Ps_warm['P_W']
    T_LN = Ps_cold['P_E']-Ps_cold['P_W']
    r_P = (T_EN + T_LN)/(T_EN - T_LN)
    return r_P
