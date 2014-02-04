import sys as _sys


def Wave_Speeds(drho,Hbar,verbose=False,fout=_sys.stdout):
    ''' Calculate the Kelvin and Rossby wave speed
    '''
    import numpy
    g = 9.8
    ck = numpy.sqrt(g*drho*Hbar)
    cr = ck/3.
    if verbose: 
        fout.write("Speed of Kelvin wave [m/s]:"+str(ck)+"\n")
        fout.write("Speed of Rossby wave [m/s]:"+str(cr)+"\n")
        fout.flush()
    return ck,cr

def wind(temp,gam,r,
         warmseasonal=None,coldseasonal=None,mon=1):
    '''
    temp - temperature anomaly
    gam  - regression coefficient between wind and temperature anomalies
    r    - nonlinearity
    warmseasonal - a numpy array of length 12 (months), the value at a particular month (specified by "mon") is to be multiplied to the wind anomaly when temp is positive, default = 1
    coldseasonal - similar to warmseasonal, for times when temp is negative
    mon  - used when warmseasonal or coldseasonal is specified, range of mon should be 1-12.
    '''
    import numpy
    if warmseasonal is None: warmseasonal = numpy.ones(12)
    if coldseasonal is None: coldseasonal = numpy.ones(12)
    assert len(warmseasonal) == 12
    assert len(coldseasonal) == 12
    if temp>0.:
        u = warmseasonal[mon-1]*(gam+r)*temp
    else:
        u = coldseasonal[mon-1]*(gam-r)*temp
    return u

def dT(h,T,a,b,e):
    ''' dT = a*h-b*T-e*(T^3.) '''
    import numpy
    return a*h-b*T-e*numpy.power(T,3.)

def LagTime(ck,cr,x_W,x_E,x_force,dx=0.):
    lag_short = (x_W - x_force)*1100./ck/864./30.
    lag_long = (x_force - x_W)*1100./cr/864./30.+(x_E-x_W)*1100./ck/864./30.
    return lag_short,lag_long

def oscillator(tstep,params):
    '''
    
    '''
    import numpy
    istep = 0
    nlag_long = int(params['lag_long']/tstep)
    nlag_short = int(params['lag_short']/tstep)
    T = params['T0']
    h_slow = [params['h0'],]*nlag_long
    h_fast = [params['h0'],]*nlag_short
    f_dT = params['dT_func']
    f_wind = params['wind_func']
    f_noise = params['noise_func']
    while True:
        h_now = h_slow.pop() + h_fast.pop()
        dT = f_dT(h=h_now,T=T,
                  a=params['a'],b=params['b'],e=params['e'])
        yield T
        T += dT*tstep
        mon = int(istep*tstep % 12) + 1
        if istep*tstep % 1 < tstep*2:
            noise = f_noise()
        u = f_wind(temp=T,gam=params['gam'],r=params['r'],
                   warmseasonal=params['warmseasonal'],
                   coldseasonal=params['coldseasonal'],
                   mon=mon) + noise
        h_fast.insert(0,params['c']*u)
        h_slow.insert(0,params['d']*u*-1.)
        istep += 1
