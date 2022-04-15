import xarray as xr
import os

import matplotlib.pyplot as plt
import gfdl_utils as gu

from wmt_inert_tracer.wmt import wmt
from wmt_inert_tracer.wmt import lbin_define
from wmt_inert_tracer.preprocessing import preprocessing

###
# Parameters
# ----------
# Dask
dashboard_address = 8726
# Model configuration
config_id = 'CM4_piControl_c192_OM4p125_v5'
pathDict = {'pp':('/archive/Raphael.Dussin/'+
                  'FMS2019.01.03_devgfdl_20210706/'+
                  'CM4_piControl_c192_OM4p125_v5/'+
                  'gfdl.ncrc4-intel18-prod-openmp/pp'),
           'out':'av',
           'local':'monthly_5yr',
           'time':'0051-0055',
           'add':'*'}
ppname_3d = 'ocean_monthly_z_d2'
ppname_wfo = 'ocean_monthly_d2'
# Region
basin = "Atlantic" # Specify "Global" for no basin mask
# WMT calculation
l = 'sigma2'
dl = 0.05
bins = lbin_define(28,38,dl)
method = 'xhistogram'
# ----------
# Outpath
pathout = ('/work/gam/wmt/'
           +config_id+'/'+pathDict['out']+'/'+pathDict['local']+'/')
fileout = 'wmt.G.l-'+l+'.dl-'+str(dl)+'.basin-'+basin+'.method-'+method
if os.path.isdir(pathout):
    print('Associated path exists.')
else:
    print('Associated path does not exist, so making.')
    cmd = ("mkdir -p %s &" %pathout)
    out = os.system(cmd)
# if os.path.isfile(pathout+fileout+'.nc'):
    ### ISSUE WARNING THAT FILE ALREADY EXISTS
    ### ACCEPT CONTINUATION TO DO CALCULATION ANYWAY
# -----------

###
# Dask cluster
# ------------
# Set up dask cluster
# This configuration appears to work more often than not
from dask.distributed import Client
from dask_jobqueue import SLURMCluster

print('Setting up cluster.')
cluster = SLURMCluster(queue='stage7', cores=4, processes=4, project='gfdl_o',
                       memory="96GB", walltime="02:00:00",
                       scheduler_options={"dashboard_address": f":{dashboard_address}"},
                      job_extra=["-C bigmem"])
cluster.scale(n=16)
client = Client(cluster)
# ------------

###
# Load data
# ---------
# Load 3D tendencies
pathDict['ppname'] = ppname_3d
print('Loading data.')
path = gu.core.get_pathspp(**pathDict)
ds = gu.core.open_frompp(**pathDict)
gridpath = gu.core.get_pathstatic(pathDict['pp'],pathDict['ppname'])
grid = xr.open_dataset(gridpath).assign_coords({'xh':ds['xh'],'yh':ds['yh']})

# Load surface flux data
pathDict['ppname'] = ppname_wfo
ds_surf = gu.core.open_frompp(**pathDict)
ds = xr.merge([ds,ds_surf])
# ----------

###
# Region selection
# ----------------
if basin == "Global":
    ds = ds
else:
    if basin == "Southern":
        basin_code = 1
    elif basin == "Atlantic":
        basin_code = 2
    elif basin == "Pacific":
        basin_code = 3
    elif basin == "Arctic":
        basin_code = 4
    elif basin == "Indian":
        basin_code = 5
    cond = grid["basin"]==basin_code
    ds = ds.where(cond,0)
    
###
# Pre-processing
# ---------
print('Pre-processing.')
# Preprocess data
ds = preprocessing(ds, grid, decode_times=False, verbose=True, rechunk=True)

# Specific heat capacity (J/kg/K)
Cp = 3992.
# Sea surface density (kg/m^3)
rho = 1035
# ----------

###
# WMT calculation
# ---------------
print('Calculating WMT.')
# Create WMT class
dd = wmt(ds, Cp=Cp, rho=rho)
# WMT calculation
G = dd.G(l, bins = bins, group_tend=False, method=method)
print('Loading result.')
%time G = G.load()

###
# Save to /work
G.to_netcdf(pathout+fileout+'.nc')