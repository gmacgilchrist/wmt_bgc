#!/usr/bin/env python
# xhistogram binning of model data, nabbed from jbusecke

import xarray as xr
import numpy as np
from xhistogram.xarray import histogram

def vertical_rebin(data, bin_data, bins, dz, vert_dim="st_ocean"):
    nanmask = np.isnan(data)
    # Should we also check the bin data for nans?
    full_sum = histogram(
        bin_data.where(~nanmask),
        bins=[bins],
        weights=(data * dz).where(~nanmask),
        dim=[vert_dim],
    )
    return full_sum

def vertical_rebin_wrapper(
    ds,
    bin_data_name,
    bins,
    dz_name="dz",
    vert_dim="st_ocean",
    return_average=True,
    debug=False,
    verbose=False
    ):
    """A wrapper for the core functionality in `vertical_rebin`.
    Accepts datasets and calculates the average over the new depth coordinates.
    """
    ds = ds.copy()
    ds_rebinned = xr.Dataset()
    
    ones = xr.ones_like(ds[dz_name])
    
    dz_rebinned = vertical_rebin(
        ones,
        ds[bin_data_name],
        bins,
        ds[dz_name],
        vert_dim=vert_dim,
    )
    for var in ds.data_vars:
        if ds[var].dtype == 'float':
            if verbose:
                print(var)
            ds_rebinned[var] = vertical_rebin(
                ds[var], ds[bin_data_name], bins, ds[dz_name], vert_dim=vert_dim
            )
    if return_average:
        ds_rebinned = (
            ds_rebinned / dz_rebinned
        )  # this might cause a lot of overhead...i can try to deactivate if the save fails.

    ds_rebinned[dz_name] = dz_rebinned

    return ds_rebinned

def total_rebin_layerintegral(ds,bin_data,bins,dim,area,block_size='auto',verbose=False):
    """Rebin dataset [ds] in multiple dimensions,
    and integrate terms within new layers.
    """
    ds = ds.copy()
    ds_rebinned = xr.Dataset()
    for var in ds.data_vars:
        if ds[var].dtype == 'float':
            if verbose:
                print(var)
            nanmask = np.isnan(ds[var])
            ds_rebinned[var] = histogram(
                bin_data.where(~nanmask), 
                bins=[bins], 
                dim=dim, 
                weights=(ds[var]*area).where(~nanmask),
                block_size=block_size
            )
    return ds_rebinned