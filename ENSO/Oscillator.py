''' Conceptual model for ENSO.  This is a rewrite of a Fortran program used in
`Choi et al (2013) <http://dx.doi.org/10.1175/JCLI-D-13-00045.1>`_.

The main function is :py:func:`oscillator` which is a generator that yields ENSO
SST anomalies indefinitely after being initialised.

'''
import sys as _sys
import numpy


def wind(temp,gam,r,
         warmseasonal=None,coldseasonal=None,mon=1):
    '''  Compute the zonal wind anomaly given an SST anomaly

    Assume a piecewise linear relationship between the zonal wind stress anomaly
    and the SST anomaly (see
    `Choi et al (2013) <http://dx.doi.org/10.1175/JCLI-D-13-00045.1>`_.)

    Args:
        temp (number): temperature anomaly
        gam  (number): mean regression coefficient between wind stress and
           temperature anomalies (i.e. gamma in Eq.2 of
           `Choi et al (2013) <http://dx.doi.org/10.1175/JCLI-D-13-00045.1>`_.)
        r (number): nonlinearity, greater than -1. and less than 1.  Zero means
            no nonlinearity.  Positive values mean a stronger air-sea coupling
            coefficient during warm condition than during cold condition
        warmseasonal (scalar or a numpy array of numbers): for varying coupling
            coefficient in different calendar month for the warm condition.
            The value is to be multiplied to the wind anomaly. Length = 12 (Jan,
            Feb,...).  Default is 1 (i.e. no seasonality)
        coldseasonal  (scalar or a numpy array of numbers): similar to
            warmseasonal, but for cold condition.
        mon  (int): calendar month (1-12).  Only meaningful when warmseasonal or
            coldseasonal is not 1.
    '''
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
    ''' dT = a*h-b*T-e*(T^3.)

    Args:
        a (positivie number): regression coefficient between SST anomaly growth
            rate and thermocline depth anomaly
        b (positive number): surface damping time scale (in /month)
        e (positive number): cubic damping rate (i.e. :math:`\epsilon` in  Eq.1
            of Choi et al. 2013)

    Returns:
        numeric : dT (K/month)
    '''
    return a*h-b*T-e*numpy.power(T,3.)


def oscillator(tstep,a,b,c,d,e,gam,r,
               lag_long,lag_short,
               T0,h0,noise_generator,noise_update_freq=0.2,
               warmseasonal=None,coldseasonal=None):
    ''' Generator for simulating ENSO SST anomaly under a conceptual framework
    and a nonlinear air-sea coupling coefficient. (see
    `Choi et al. (2013) <http://dx.doi.org/10.1175/JCLI-D-13-00045.1>`_)

    Args:
        tstep (number): time step size (in month)
        a (positivie number): regression coefficient between SST anomaly growth
            rate and thermocline depth anomaly, as in :py:func:`dT_func`
        b (positive number): surface damping time scale (in /month), as in
            :py:func:`dT_func`
        c (positive number): regression coefficient between SST anomaly growth
            rate and the zonal wind stress anomaly with a time lag characterised
            by the positive feedbacks (i.e. :math:`c'` in Eq.1 of Choi et al.
            2013)
        d (positive number): negative of the regression coefficient between SST
            anomaly growth rate and the zonal wind stress anomaly with a time
            lag characterised by the negative feedbacks (i.e. :math:`d'` in
            Eq.1 of Choi et al. 2013)
        e (positive number): cubic damping rate (i.e. :math:`\epsilon` in  Eq.1
            of Choi et al. 2013)
        gam  (number): mean regression coefficient between wind stress and
           temperature anomalies (i.e. gamma in Eq.2 of
           `Choi et al (2013) <http://dx.doi.org/10.1175/JCLI-D-13-00045.1>`_.)
        r (number): nonlinearity, greater than -1. and less than 1.  Zero means
            no nonlinearity.  Positive values mean a stronger air-sea coupling
            coefficient during warm condition than during cold condition
        lag_long (number): time lag for the negative feedback to arrive (in
            month)
        lag_short (number): time lag for the positive feedback to arrive (in
            month)
        T0 (number): Initial SST anomaly
        h0 (number): Initial thermocline depth anomaly
        noise_generator (generator): to be added to the zonal wind stress anomaly
        noise_update_freq (number): how frequently the noise is updated (in
           month)
        warmseasonal (scalar or an array): see :py:func:`wind`
        coldseasonal (scalar or an array): see :py:func:`wind`

    Yields:
        T (number): SST anomaly at the current time step
        
    '''
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
        if istep % int(noise_update_freq/tstep) == 0:
            noise = noise_generator()
        u = wind(temp=T,gam=gam,r=r,
                 warmseasonal=warmseasonal,
                 coldseasonal=coldseasonal,
                 mon=mon) + noise
        h_fast.insert(0,c*u)
        h_slow.insert(0,d*u*-1.)
        istep += 1


def wave_speeds(drho,hbar,verbose=False,fout=None):
    ''' Calculate the Kelvin and Rossby wave speed

    Args:
       drho (number): approximate density ratio between the surface mixed layer
          and the deeper ocean (less than 1.)
       hbar (number): mean thickness of the surface mixed layer
       verbose (bool)
       fout (File Object): if None, default is :py:data:`sys.stdout`

    Returns:
       (number, number): Kelvin wave speed, Rossby wave speed, in m/s
    '''
    if fout is None:
        fout = _sys.stdout
    g = 9.8
    ck = numpy.sqrt(g*drho*hbar)
    cr = ck/3.
    if verbose:
        fout.write("Speed of Kelvin wave [m/s]:"+str(ck)+"\n")
        fout.write("Speed of Rossby wave [m/s]:"+str(cr)+"\n")
        fout.flush()
    return ck,cr


def lag_time(ck,cr,x_W,x_E,x_force):
    ''' Compute the lag times for (1) Kelvin waves and (2) Rossby waves +
    reflected Kelvin waves to reach the eastern boundary.

    i.e. the difference of the two is the time lag between positive feedbacks
    and negative feedbacks of ENSO

    Args:
       ck (number): Kelvin wave speed in m/s
       cr (number): Gravest Rossby wave speed in m/s
       x_W (number): Longitude of western boundary (in degree)
       x_E (number): Longitude of eastern boundary (in degree)
       x_force (number): Longitude of the wind forcing

    Returns:
       (number, number): lag time for positive feedbacks, lag time for negative
            feedbacks (units: month)
    '''
    lag_short = (x_W - x_force)*1100./ck/864./30.
    lag_long = (x_force - x_W)*1100./cr/864./30.+(x_E-x_W)*1100./ck/864./30.
    return lag_short,lag_long
