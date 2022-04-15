#!/nbhome/Graeme.Macgilchrist/miniconda3/envs/py39/bin/python
#SBATCH --job-name=hist_age-sigma2
#SBATCH --output=hist_age-sigma2.o%j
#SBATCH --time=00:30:00
#SBATCH -p analysis
#SBATCH -C bigmem

import xarray as xr
from xhistogram.xarray import histogram
import gfdl_utils as gu
import gsw
import glob
import xesmf as xe
import numpy as np

ts=196
te=200

config_id = 'CM4_piControl_c192_OM4p125_v6_alt1'
ppin = '/archive/Raphael.Dussin/FMS2019.01.03_devgfdl_20210706/'
compiler = 'gfdl.ncrc4-intel18-prod-openmp'

pathDict = {'pp':(ppin+
                  config_id+'/'
                  compiler+'/pp'),
           'out':'ts',
           'local':'annual/10yr',
           'time':'0191-0200'}

### Load data
# Age tracer
pathDict['ppname'] = 'ocean_annual_z_d2'
pathDict['add'] = 'agessc'
ds_agessc = gu.core.open_frompp(**pathDict)
pathDict['add'] = 'volcello'
ds_volcello = gu.core.open_frompp(**pathDict)
# Merge
ds_d2 = xr.merge([ds_agessc,ds_volcello])
# Grid
grid_d2 = xr.open_dataset(gu.core.get_pathstatic(pathDict['pp'],pathDict['ppname']))

# T and S
pathDict['ppname'] = 'ocean_annual_z'
pathDict['add'] = 'so'
ds_so = gu.core.open_frompp(**pathDict)
pathDict['add'] = 'thetao'
ds_thetao = gu.core.open_frompp(**pathDict)
# Merge
ds_ts = xr.merge([ds_so,ds_thetao])
# Grid
grid = xr.open_dataset(gu.core.get_pathstatic(pathDict['pp'],pathDict['ppname']))


ppout=/work/gam/zarr/

for t in $(seq $ts 5 $te); do
    time=$(printf "%04d" $t)-$(printf "%04d" $(($t+4)))
    # Check if the input file exists
    infile=${ppin}${config_id}/${compiler}/pp/${ppname}/${out}/${local}/${ppname}.${time}.01.nc
    if test -f "$infile"; then
        for add in {01..12}; do
            infile=${ppin}${config_id}/${compiler}/pp/${ppname}/${out}/${local}/${ppname}.${time}.${add}.nc
            outfile=${ppout}${config_id}/${compiler}/pp/${ppname}/${out}/${local}/
            # Only create it if it doesn't exist already
            if [ ! -d $outfile/${ppname}.${time}.${add} ]; then
                sbatch /home/Graeme.Macgilchrist/utils/process_zarr_simple.py ${infile} ${outfile}
            fi
    done
    fi
done