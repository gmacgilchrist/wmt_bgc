# Some basic functions for watermass transformation calculation
import xarray as xr

def calc_densityflux(Qheat,Qfw,S=35,alpha=1E-4,beta=1E-3,Cp=3992.0):
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