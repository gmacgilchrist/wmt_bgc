#!/usr/bin/env python
# Estimate vertical mass transport (wmo) by the divergence of the horizontal mass transports. The vertical mass transport across an
# interface is the cumulative integral starting from the bottom of a water column. The sign convention is w>0 is an upward transport,
# (i.e., towards the surface of the ocean). By this convention, then div(u,v)<0 implies a negative (downward) transport and vice versa.
# Based on verticalvelocity.py, written by Andrew Shao (andrew.shao@noaa.gov) 29 September 2017
# Adapted to work with xarray by Graeme MacGilchrist (graemem@princeton.edu) 7 June 2019

import xarray as xr
import numpy as np

def calc_w_continuity(u,v,z,wrapx = True,wrapy = True):
    ### SET UP ###
    # Predefine a w variable, with appropriate dimensions and all values set to zero
    w = xr.DataArray(np.zeros([u.time.size,z.size,u.yh.size,v.xh.size]),coords=[u.time,z,u.yh,v.xh],dims=['time','z_i','yh','xh'])
    # Set NaNs to zero
    uo = u.fillna(0)
    vo = v.fillna(0)
    # Take difference of mass transport between neighbours along x and y dimensions (wrapping end points around)
    du = uo.roll(xq=1,roll_coords=False)-uo
    dv = vo.roll(yq=1,roll_coords=False)-vo
    
    ### CALCULATE W ###
    # Add du and dv (from surface to second-to-last index)
    w.loc[dict(z_i=w.z_i[w.z_i<w.z_i.max()])] += du.values
    w.loc[dict(z_i=w.z_i[w.z_i<w.z_i.max()])] += dv.values
    # If the domain is not wrapped in the x/y-direction, the westernmost/southernmost point is zero,
    # so subtract the wrapped value (equivalent to subtracting the easternmost/northernmost value in the un-wrapped variable)
    if not wrapx:
        w.loc[dict(z_i=w.z_i[w.z_i<w.z_i.max()], xh=w.xh[0])] += -uo.loc[dict(xq=uo.xq[-1])].values
    if not wrapy:
        w.loc[dict(z_i=w.z_i[w.z_i<w.z_i.max()], yh=w.yh[0])] += -vo.loc[dict(yq=vo.yq[-1])].values
    # Sum w cumulatively from the bottom
    w = w.assign_coords(z_i_neg=-w.z_i) # Assign a dummy coordinate to allow summation from bottom
    w = w.sortby('z_i_neg').cumsum(dim='z_i').sortby('z_i') # Flip to sum from bottom, then flip again to normal orientation
    
    ### APPLY MASK ###
    # Recover mask of u and v velocities
    masku = xr.ufuncs.isfinite(u) & xr.ufuncs.isfinite(u.roll(xq=1,roll_coords=False))
    maskv = xr.ufuncs.isfinite(v) & xr.ufuncs.isfinite(v.roll(yq=1,roll_coords=False))
    # Place them on the same coordinate grid
    masku = masku.drop('xq').rename({'xq':w.xh.name})
    maskv = maskv.drop('yq').rename({'yq':w.yh.name})
    # Combine mask
    mask = masku & maskv
    # Extend the mask array to concide with the w-grid
    # (if the bottom value on the horizontal grid is masked, so too is the bottom interface)
    maskw = xr.concat([mask,mask.isel(z_l=-1)],dim='z_l')
    # Set the z-coordinate to match that of the w grid
    maskw = maskw.rename({'z_l':w.z_i.name})
    maskw = maskw.assign_coords(z_i=w.z_i)
    # Apply mask to w velocities
    w = w.where(maskw,np.nan)
    
    return w
