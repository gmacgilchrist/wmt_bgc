# Some basic functions for watermass transformation calculation
import xarray as xr
import gsw

def densityflux(Qheat,Qfw,S=35,alpha=1E-4,beta=1E-3,Cp=3992.0):
    '''Calculate density flux from heat and freshwater fluxes.
    
    Parameters
    ----------
    FW : {xr.DataArray}
        Freshwater flux [kg m-2 s-1]
    Q : {xr.DataArray}
        Heat flux [W m-2]
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
    
    # It's a curiosity here that I have to specify .values. When gsw operates 
    # on a dask array, it collapses the first dimension. Instead, extract the
    # underlying numpy array. This likely slows performance, so may need to be
    # flagged in future. Unclear if it is a dask issue or a gsw issue.
    return xr.apply_ufunc(gsw.sigma0,
                         SA.values, CT.values,
                         dask='allowed')

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

