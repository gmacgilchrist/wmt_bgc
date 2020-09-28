#!/usr/bin/env python
# Series of functions pertaining to calculating watermass transformation
import xarray as xr
import numpy as np
from xhistogram.xarray import histogram
    
def calc_E(l,dl,l_i_vals,c=None,weight=None,dims=['xh','yh','zl']):
    '''Transport of mass or tracer across contours of [l], due to processes [dl].
    l : xr.DataArray;
        Intensive variable (e.g. temp) that defines layers across which transport
        will be determined.
    dl : xr.DataArray;
        Tendency of l due to (individual or total sum of) diffusive processes,
        e.g. heat tendencies
        Multiple processes are included as Dataarrays inside a dataset.
    l_i_vals : ndarray or xr.DataArray;
        Interface values of l; transport will be across midpoints of these interfaces
    c : None or scalar or xr.DataArray
        If included, evaluating transport of tracer as opposed to volume
        A scalar could be density, to get mass rather than volume transport
    weight : xr.DataArray;
        Distribution of weights by which to multiply [dl] to remove spatial
        dimension of units. E.g. heat flux in Wm-2 should be multiplied by the area
        of the grid cell to recover total heating.
    dims : list of string;
        Dimensions along which to perform histogram.'''
    
    import xarray as xr
    import numpy as np
    from xhistogram.xarray import histogram
    
    # Get the spacing of contours of l
    delta_l_vals = np.diff(l_i_vals)
    # xhistogram has some curiousities around nan-values, so exclude them
    nanmask = np.isnan(l)
    
    # Specify what should be integrated
    # (this is rather clunky with if statements, but is meant to avoid loading
    # unnecessary variables, such as dummy [c]'s or [weight]'s)
    if c is None:
        if weight is None:
            weights = dl
        else:
            weights = dl*weight
    else:
        if weight is None:
            weights = c*dl
        else:
            weights = c*dl*weight
            
    # Integrate [weights] within layers of [l] and divide by [delta_l]
    E = histogram(l.where(~nanmask),
                  bins=[l_i_vals],
                  weights=weights.where(~nanmask),
                  dim=dims,
                  block_size=1)/delta_l_vals
    
    return E

def calc_E_wrapper(l,dls,l_i_vals,c=None,weight=None,dims=['xh','yh','zl']):
    '''Wrapper for calc_E to allow evaluation of contributions from different processes,
    each represented as a DataArray in the Dataset [dls].
    l : xr.DataArray;
        Intensive variable (e.g. temp) that defines layers across which transport
        will be determined.
    dls : xr.Dataset;
        Tendencies of l due to differnt diffusive processes,
        e.g. horizontal and vertical diffusion tendencies
        Multiple processes are included as Dataarrays inside a dataset.
    l_i_vals : ndarray or xr.DataArray;
        Interface values of l; transport will be across midpoints of these interfaces
    c : None or scalar or xr.DataArray
        If included, evaluating transport of tracer as opposed to volume
        A scalar could be density, to get mass rather than volume transport
    weight : xr.DataArray;
        Distribution of weights by which to multiply [dl] to remove spatial
        dimension of units. E.g. heat flux in Wm-2 should be multiplied by the area
        of the grid cell to recover total heating.
    dims : list of string;
        Dimensions along which to perform histogram.'''
    
    # Set up dataset for contributions to E
    E = xr.Dataset()
    for var in dls.data_vars:
        dl = dls[var]
        E[var] = calc_E(l,dl,l_i_vals,c=c,weight=weight,dims=dims)
        
    return E

def calc_volumetric_cumsum(l,p,l_l_vals,weight=None,dims=['xh','yh','zl'],greaterthan=True,):
    '''Integration of quantity [p] across volume with [l] greater than layer defined
    by contours l_l_vals.
    
    *** IMPORTANT *** 
    Limits of l_l_vals must span the full range of l.
    *****************
    
    l : xr.DataArray;
        Intensive variable, to define volume boundary
    p : xr.DataArray;
        Quantity to be integrated (e.g. thickness)
    l_l_vals : ndarray or xr.DataArray
        Values of the contours for which the integration will be calculated
        *** This is currently set up such that the layer values are increasing ***
        *** I expect it can be generalized, but have not explored this yet     ***
    weight : Distribution of weights by which to multiply [p] to remove spatial
        dimension of units. E.g. thickness in units 'm' should be multiplied by the area
        of the grid cell to recover total volume.
    greaterthan : boolean;
        True if the integral should be over contours 
        greater than each contour. False for less than.'''
    
    # xhistogram has some curiousities around nan-values, so exclude them
    nanmask = np.isnan(l)
    
    # Specify what should be integrated
    if weight is None:
        weights = p
    else:
        weights = p*weight
        
    # Integrate p between each contour layer
    P_l = histogram(l.where(~nanmask),bins=[l_l_vals],weights=weights.where(~nanmask),dim=dims,block_size=1)
    # Cumulatively sum each layer (concatenate zeros at start), and take away from total sum to reverse the order of the summation
    # i.e. this the cumulative sum of integrated [p] from largest to smallest [l_l]
    P_l_cumsum = xr.concat([xr.zeros_like(P_l.isel({l.name+'_bin':0})),P_l.cumsum(l.name+'_bin')],dim=l.name+'_bin')
    if greaterthan:
        # Take cumulative sum away from total sum to reverse the order
        # Assign coordinates to match the contours (rather than the interfaces)
        P = (P_l.sum(l.name+'_bin')-P_l_cumsum).assign_coords({l.name+'_bin':l_l_vals})
    else:
        P = P_l_cumsum.assign_coords({l.name+'_bin':l_l_vals})
                                                                          
    return P

def calc_P_wrapper(l,ps,l_l_vals,weight=None,dims=['xh','yh','zl'],greaterthan=True):
    '''Wrapper for calc_P to allow integration of numerous quantities,
    each represented as a DataArray in the Dataset [ps].
    
    *** IMPORTANT *** 
    Limits of l_l_vals must span the full range of l.
    *****************
    
    l : xr.DataArray;
        Intensive variable (e.g. temp) that defines layers across which transport
        will be determined.
    ps : xr.Dataset;
        Quantities for integration.
        Multiple quantities are included as Dataarrays inside a dataset.
    l_l_vals : ndarray or xr.DataArray
        Values of the contours for which the integration will be calculated
        *** This is currently set up such that the layer values are increasing ***
        *** I expect it can be generalized, but have not explored this yet     ***
    weight : xr.DataArray
        Distribution of weights by which to multiply [p] to remove spatial
        dimension of units. E.g. thickness in units 'm' should be multiplied by the area
        of the grid cell to recover total volume.
    greaterthan : boolean;
        True if the integral should be over contours 
        greater than each contour. False for less than.'''
    
    # Set up dataset for contributions to E
    P = xr.Dataset()
    for var in ps.data_vars:
        p = ps[var]
        P[var] = calc_volumetric_cumsum(l,p,l_l_vals,weight=weight,dims=dims,greaterthan=greaterthan)
        
    return P

def calc_dPdt(l,p,l_l_vals,weight=None,greaterthan=True,dims=['xh','yh','zl'],dim_time='time',delta_t=None,new_time=None):
    '''Take the volumetric cumulative sum of quantity [p] and calculate the change over time.
    Within this function, the newly created dPdt can be *realigned* with other quantities, 
    by defining a [new_time]. This is basically just a wrapper to simultaneously calculate
    P, and its change over time.
    
    The most common use of this function will be to calculate the change in volume/mass/tracer
    content over time.
    
    l : xr.DataArray;
        Intensive variable, to define volume boundary
    p : xr.DataArray;
        Quantity to be integrated (e.g. thickness)
    l_l_vals : ndarray or xr.DataArray
        Values of the contours for which the integration will be calculated
        *** This is currently set up such that the layer values are increasing ***
        *** I expect it can be generalized, but have not explored this yet     ***
    weight : Array of weights by which to multiply [p] to remove spatial
        dimension of units. E.g. density units in units 'kgm-3' should be multiplied by the volume
        of the grid cell to recover total mass.
    greaterthan : boolean;
        True if the integral should be over contours 
        greater than each contour. False for less than.
    dim_time : str
        Name of time dimension in DataArrays
    delta_t : xr.DataArray
        Array of time differences between entries in [l] and [p]
    new_time : xr.DataArray or np.array
        Option to reassign time coordinates after differentiating, to align with other objects.'''
    
    if new_time is None:
        new_time = l[dim_time]
    
    # If not defined, calculate the time from the loaded DataArrays
    if delta_t is None:
        delta_t_vals = l[dim_time].diff(dim_time).values.astype('timedelta64[s]').astype('int')
        delta_t = xr.DataArray(delta_t_vals,dims=[dim_time],coords=({dim_time:new_time}))
    
    # Integrate the volume/mass/tracer
    P = calc_volumetric_cumsum(l,p,l_l_vals,weight=weight,dims=dims,greaterthan=greaterthan)
    # Calculate the change over time
    dPdt = P.diff(dim_time).assign_coords({dim_time:new_time})/delta_t
    
    return dPdt