#!/usr/bin/env python
# Series of functions pertaining to calculating watermass transformation
import xarray as xr
import numpy as np
from xhistogram.xarray import histogram
#from xarrayutils import vertical_coordinates as vc

def calc_G(l,dl,l_i_vals,area,plot=False):
    '''Mass transport across contours of [l].
    l : intensive variable, to evaluate transport across
    dl : summed tendency of (vertically integrated) l due to diffusive processes;
        e.g. heat tendency if l if temperature
    l_i_vals : interface values of l
    area : 2d distribution of horizontal area (dx*dy)'''
    
    # Get the spacing of contours of l
    delta_l_vals = np.diff(l_i_vals)
    # xhistogram has some curiousities around nan-values, so exclude them
    nanmask = np.isnan(l)
    # Integrate [dl] within layers of [l] and divide by [delta_l]
    G = histogram(l.where(~nanmask),bins=[l_i_vals],weights=(dl*area).where(~nanmask),dim=['xh','yh','zl'],block_size=1)/delta_l_vals
    
    if plot:
        G.plot()
    return G

def calc_E(c,l,dl,l_i_vals,area,plot=False):
    '''Transport of tracer [c] mass across contours of [l].
    c : tracer for which to evaluate transport (units of tracer mass per kg)
    l : intensive variable, to evaluate transport across
    dl : summed tendency of (vertically integrated) l due to diffusive processes;
        e.g. heat tendency if l if temperature
    l_i_vals : interface values of l
    area : 2d distribution of horizontal area (dx*dy)'''
    
    # Get the spacing of contours of l
    delta_l_vals = np.diff(l_i_vals)
    # xhistogram has some curiousities around nan-values, so exclude them
    nanmask = np.isnan(l)
    # Integrate [c*dl] within layers of [l] and divide by [delta_l]
    E = histogram(l.where(~nanmask),bins=[l_i_vals],weights=(c*dl*area).where(~nanmask),dim=['xh','yh','zl'],block_size=1)/delta_l_vals
    if plot:
        E.plot()
    return E

def calc_P(p,l,l_i_vals,area):
    '''Integration of quantity [p] across volume with [l] greater than *mid-point* 
    values between layer interfaces [l_i_vals] (i.e. the center layer values).
    This is done so that the output of calc_P aligns with that of calc_G and calc_E,
    when given the same l_i_vals.
    
    *** IMPORTANT *** 
    Limits of l_i_vals should be such that their mid-points span the full range of l 
    *****************
    
    p : quantity to be integrated (e.g. tendencies of independent tracer [c])
    l : intensive variable, to define volume boundary
    l_i_vals : interface values of l 
        (integral is done for l[i] >= 0.5*(l_i_vals[i]+l_i_vals[i+1]))
    area : 2d distribution of horizontal area (dx*dy)'''
    
    # Get the mid-points of the layers (as defined by their interfaces)
    l_l_vals = 0.5*(l_i_vals[:-1]+l_i_vals[1:])
    # xhistogram has some curiousities around nan-values, so exclude them
    nanmask = np.isnan(l)
    # Integrate p between each layer mid-point
    P_l = histogram(l.where(~nanmask),bins=[l_l_vals],weights=(p*area).where(~nanmask),dim=['xh','yh','zl'],block_size=1)
    # Cumulatively sum each layer (concatenate zeros at start), and take away from total sum to reverse the order of the summation
    # i.e. this the cumulative sum of integrated [p] from largest to smallest [l_l]
    # Concatenate zeroes
    # At the same time, reassign the coordinates to align with the layer mid-points
#     P = xr.concat([P_l.sum(l.name+'_bin'),(P_l.sum(l.name+'_bin')-P_l.cumsum(l.name+'_bin'))],
#                   dim=l.name+'_bin').assign_coords({l.name+'_bin':l_l_vals})
    P_l_cumsum = xr.concat([xr.zeros_like(P_l.isel({l.name+'_bin':0})),P_l.cumsum(l.name+'_bin')],dim=l.name+'_bin')
    P = (P_l.sum(l.name+'_bin')-P_l_cumsum).assign_coords({l.name+'_bin':l_l_vals})
                                                                          
    return P

def calc_E_old(ds,l,l_i_vals,dl,c,area,z='depth',xdim='xh',ydim='yh',zldim='zl',zidim='zi',binning=None):
    '''Evaluation of the transport of tracer [c] across contours of tracer [l]
    l and c should be given as strings and both be contained in Dataset ds
    dl corresponds to the diffusive time tendency of tracer *content* of l,
    i.e. the tendency of rho*l*h where h is layer thickness, and rho is in situ
    density, thus in units of [kg m^-3]*[l tracer unit]*[m]*[s-1].
    Then
        E(l') = \frac{1}{\Delta l}\iint_{l=l'} c*dl dA
    This uses the fact that the tracer content tendency is equal to the 
    rho * tracer tendency integrated within a layer (which is formulation for
    wmt in Groeskamp et al, 2019).'''
    
    E = xr.Dataset()
    
    # Calculate tracer at layer centres from interface values
    l_l_vals = 0.5*(l_i_vals[:-1]+l_i_vals[1:])
    # Calculate delta_tracer for each layer
    delta_l_vals = np.diff(l_i_vals)
    # Place in DataArrays
    l_i=xr.DataArray(l_i_vals,dims=['l_i'],coords={'l_i':l_i_vals})
    l_l=xr.DataArray(l_l_vals,dims=['l_l'],coords={'l_l':l_l_vals})
    delta_l = xr.DataArray(delta_l_vals,dims=['l_l'],coords={'l_l':l_l_vals})

    # Calculate cdl
    cdl = ds[c]*dl

    # Bin into layers of l, and sum up across space
    if binning is None:
        work = (cdl*area).sum(dim=[xdim,ydim])
        work = work.rename({vertc:'l_l'})
    if binning=='xhistogram':
        nanmask = np.isnan(cdl)
        work = histogram(
                    ds[l].where(~nanmask), 
                    bins=[l_i.values], 
                    dim=['xh','yh','zl','time'], 
                    weights=(cdl*area).where(~nanmask),
                    block_size=None
                    )
        work = work.rename({l+'_bin':'l_l'})
    if binning=='busecke':
        l_depth_i = vc.linear_interpolation_regrid(ds[z+'_l'], ds[l], l_i, z_bounds=ds[z+'_i'], target_value_dim='l_i', z_bounds_dim=zidim, z_dim=zldim)
        cdl_remapped = conservative_remap(cdl,z_bnds_source=ds[z+'_i'], z_bnds_target=l_depth_i,
                                           z_dim=zldim, z_bnd_dim=zidim, z_bnd_dim_target='regridded', mask=True)
        work = (cdl_remapped*area).sum(dim=[xdim,ydim])
        work = work.rename({'remapped':'l_l'})
        E['l_depth_i'] = l_depth_i
    
    E['E'] = work/delta_l
    E['dE']= E['E'].interp(l_l=l_i,method='linear').diff('l_i').assign_coords(l_i=l_l.values).drop('l_l').rename({'l_i':'l_l'})
    
    return E