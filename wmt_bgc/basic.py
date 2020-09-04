# Some basic functions for watermass transformation calculation
import xarray as xr
import gsw

def densityflux(Qheat,Qfw,S=35,alpha=1E-4,beta=1E-3,Cp=3992.0):
    '''Calculate density flux from heat and freshwater fluxes.
    
    Parameters
    ----------
    Qheat : {xr.DataArray}
        Heat flux [W m-2]
    Qfw : {xr.DataArray}
        Freshwater flux [kg m-2 s-1]
    S : float or {xr.DataArray}
        Salinity [g kg-1]
    alpha : float or {xr.DataArray}
        Thermal expansion coefficient [K-1]
    beta : float of {xr.DataArray}
        Haline contraction coefficient [kg g-1]
    Cp : float
        Specific heat capacity of seawater [J kg-1 K-1; default = 4200]
    
    Returns
    -------
    densityflux : {xr.DataSet}
        densityflux : Density flux [kg m-2 s-1]
        densityflux_Q : Density flux due to heat flux [kg m-2 s-1]
        densityflux_FW : Density flux due to freshwater flux [kg m-2 s-1]
    '''
    densityflux = xr.Dataset()
    densityflux['densityflux_Qheat'] = (alpha/Cp)*Qheat
    densityflux['densityflux_Qfw'] = -Qfw*S*beta
    densityflux['densityflux'] = densityflux['densityflux_Qheat'] + densityflux['densityflux_Qfw']
    
    return densityflux

def densitytendency_from_heat_and_salt(heat_tend,salt_tend,alpha=1E-4,beta=1E-3,Cp=3992.0):
    '''Calculate tendency of locally-referenced potential density, from
    the tendencies of heat and salt.
    
    Parameters
    ----------
    heat_tend : {xr.DataArray}
        Heat tendency, [W m-3 | W m-2] (if depth-integrated)
    salt_tend : {xr.DataArray}
        Salt tendency, [kg m-3 | kg m-2]
    alpha : float or {xr.DataArray}
        Thermal expansion coefficient [K-1]
    beta : float of {xr.DataArray}
        Haline contraction coefficient [kg g-1]
    Cp : float
        Specific heat capacity of seawater [J kg-1 K-1; default = 4200]
    
    Returns
    -------
    densitytendency : {xr.DataSet}
        densitytendency : Density tendency [kg m-3 s-1 | kg m-2 s-1]
        densitytendency_heat : Density tendency due to heat flux [kg m-3 s-1 | kg m-2 s-1]
        densitytendency_salt : Density tendency due to freshwater flux [kg m-3 s-1 | kg m-2 s-1]
    '''
    densitytendency = xr.Dataset()
    densitytendency['heat'] = (alpha/Cp)*heat_tend
    densitytendency['salt'] = beta*salt_tend*1000 # Factor of 1000 converts salt_tend to g m-2 s-1
    densitytendency['total'] = densitytendency['heat'] + densitytendency['salt']
    
    return densitytendency

# -------------------------------- #
# Gibbs-Seawater function wrappers #

def gsw_p_from_z(z, lat, geo_strf_dyn_height=0, sea_surface_geopotential=0):
    '''xarray wrapper for gsw.p_from_z
    
    From gsw function:
    
    Calculates sea pressure from height using computationally-efficient
    75-term expression for density, in terms of SA, CT and p (Roquet et al.,
    2015).  Dynamic height anomaly, geo_strf_dyn_height, if provided,
    must be computed with its p_ref = 0 (the surface). Also if provided,
    sea_surface_geopotental is the geopotential at zero sea pressure. This
    function solves Eqn.(3.32.3) of IOC et al. (2010) iteratively for p.

    Parameters
    ----------
    z : array-like
        Depth, positive up, m
    lat : array-like
        Latitude, -90 to 90 degrees
    geo_strf_dyn_height : array-like
        dynamic height anomaly, m^2/s^2
            Note that the reference pressure, p_ref, of geo_strf_dyn_height must
            be zero (0) dbar.
    sea_surface_geopotential : array-like
        geopotential at zero sea pressure,  m^2/s^2

    Returns
    -------
    p : array-like, dbar
        sea pressure
        ( i.e. absolute pressure - 10.1325 dbar )'''
    
    return xr.apply_ufunc(gsw.p_from_z,
                         z, lat, geo_strf_dyn_height, sea_surface_geopotential,
                         dask='allowed')

def gsw_sigma0(SA,CT):
    '''xarray wrapper for gsw.sigma0
    
    From gsw function:
    
    Calculates potential density anomaly with reference pressure of 0 dbar,
    this being this particular potential density minus 1000 kg/m^3.  This
    function has inputs of Absolute Salinity and Conservative Temperature.
    This function uses the computationally-efficient expression for
    specific volume in terms of SA, CT and p (Roquet et al., 2015).

    Parameters
    ----------
    SA : array-like
        Absolute Salinity, g/kg
    CT : array-like
        Conservative Temperature (ITS-90), degrees C

    Returns
    -------
    sigma0 : array-like, kg/m^3
        potential density anomaly with
        respect to a reference pressure of 0 dbar,
        that is, this potential density - 1000 kg/m^3.'''
    
    return xr.apply_ufunc(gsw.sigma0,
                         SA, CT,
                         dask='parallelized',output_dtypes=[SA.dtype])

def gsw_alpha(SA,CT,p):
    '''xarray wrapper for gsw.alpha
    
    From gsw function:
    
    Calculates the thermal expansion coefficient of seawater with respect to
    Conservative Temperature using the computationally-efficient expression
    for specific volume in terms of SA, CT and p (Roquet et al., 2015).

    Parameters
    ----------
    SA : {xr.DataArray}
        Absolute Salinity, g/kg
    CT : {xr.DataArray}
        Conservative Temperature (ITS-90), degrees C
    p : {xr.DataArray}
        Sea pressure (absolute pressure minus 10.1325 dbar), dbar

    Returns
    -------
    alpha : array-like, 1/K
        thermal expansion coefficient
        with respect to Conservative Temperature'''
    
    return xr.apply_ufunc(gsw.alpha,
                         SA, CT, p,
                         dask='allowed')

def gsw_beta(SA,CT,p):
    '''xarray wrapper for gsw.beta
    
    From gsw function:
    
    Calculates the saline (i.e. haline) contraction coefficient of seawater
    at constant Conservative Temperature using the computationally-efficient
    75-term expression for specific volume in terms of SA, CT and p
    (Roquet et al., 2015).

    Parameters
    ----------
    SA : array-like
        Absolute Salinity, g/kg
    CT : array-like
        Conservative Temperature (ITS-90), degrees C
    p : array-like
        Sea pressure (absolute pressure minus 10.1325 dbar), dbar

    Returns
    -------
    beta : array-like, kg/g
        saline contraction coefficient
        at constant Conservative Temperature'''
    
    return xr.apply_ufunc(gsw.beta,
                         SA, CT, p,
                         dask='allowed')

# ----------------------------- #
# Other thermodynamic functions #

def sigmantr_mjd05(S,T):
    '''Calculates approximate neutral density from potential temperature
    and salinity, using the empirically-derived functional form of
    MacDougall and Jackett (2005)

    Parameters
    ----------
    S : {xr.DataArray}
        Salinity, g/kg OR psu
    T : {xr.DataArray}
        Potential Temperature, degrees C

    Returns
    -------
    sigmantr : {xr.DataArray}
        approximation of neutral density, kg/m3

    This function was copied across from cdfsigntr.f90 from CDFTOOLS package 
    by G.MacGilchrist (gmacgilchrist@gmail.com)'''

    import numpy as np

    dl_t = T;
    dl_s = S;
    dl_sr= np.sqrt(np.abs(dl_s));

    ### Numerator
    # T-polynome
    dl_r1=((-4.3159255086706703E-4*dl_t+8.1157118782170051E-2 )*dl_t+2.2280832068441331E-1 )*dl_t+1002.3063688892480E0;
    # S-T Polynome
    dl_r2=(-1.7052298331414675E-7*dl_s-3.1710675488863952E-3*dl_t-1.0304537539692924E-4)*dl_s;
    ### Denominator
    # T-Polynome
    dl_r3=(((-2.3850178558212048E-9*dl_t-1.6212552470310961E-7)*dl_t+7.8717799560577725E-5)*dl_t+4.3907692647825900E-5)*dl_t+1.0e0;
    # S-T Polynome
    dl_r4=((-2.2744455733317707E-9*dl_t*dl_t+6.0399864718597388E-6)*dl_t-5.1268124398160734E-4 )*dl_s;
    # S-T Polynome
    dl_r5=(-1.3409379420216683E-9*dl_t*dl_t-3.6138532339703262E-5)*dl_s*dl_sr;

    ### Neutral density
    sigmantr = ( dl_r1 + dl_r2 ) / ( dl_r3 + dl_r4 + dl_r5 ) - 1000E0;

    return sigmantr