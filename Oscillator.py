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

def dT_func(h,T,a,b,e):
    ''' dT = a*h-b*T-e*(T^3.) '''
    import numpy
    return a*h-b*T-e*numpy.power(T,3.)

def LagTime(ck,cr,x_W,x_E,x_force,dx=0.):
    lag_short = (x_W - x_force)*1100./ck/864./30.
    lag_long = (x_force - x_W)*1100./cr/864./30.+(x_E-x_W)*1100./ck/864./30.
    return lag_short,lag_long

def oscillator(tstep,a,b,c,d,e,gam,r,
               lag_long,lag_short,
               T0,h0,noise_generator,
               warmseasonal=None,coldseasonal=None):
    '''
    
    '''
    import numpy
    istep = 0
    nlag_long = int(lag_long/tstep)
    nlag_short = int(lag_short/tstep)
    T = T0
    h_slow = [h0,]*nlag_long
    h_fast = [h0,]*nlag_short
    while True:
        h_now = h_slow.pop() + h_fast.pop()
        dT = dT_func(h=h_now,T=T,a=a,b=b,e=e)
        yield T
        T += dT*tstep
        mon = int(istep*tstep % 12) + 1
        if istep*tstep % 1 < tstep*2:
            noise = noise_generator()
        u = wind(temp=T,gam=gam,r=r,
                 warmseasonal=warmseasonal,
                 coldseasonal=coldseasonal,
                 mon=mon) + noise
        h_fast.insert(0,c*u)
        h_slow.insert(0,d*u*-1.)
        istep += 1
