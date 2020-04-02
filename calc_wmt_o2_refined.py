# Wrapper script to calculate the error associated with evaluating layer-wise
# budgets of oxygen, under varying degrees of grid refinement.
# The calculation is sluggish (and inefficient) so I am putting it into a
# python script that I can run outside a notebook.

# Most of what is here is grabbed from calc_wmt_o2.ipynb
print('Here we go. Importing packages.')
import xarray as xr
import numpy as np
import budgetcalcs as bc
import calc_wmt as wmt

### FUNCTIONS ###
# First define some functions to do the grid refinement and the wmt calculation
### GRID REFINEMENT ###
def calc_refine(da,refineby,variable_type,vertc='zl'):
    nk = len(da[vertc])
    # Set vertical coordinate to layer index
    da=da.assign_coords({vertc:np.linspace(1,nk,nk)})
    # Assign a k-value for the interfaces
    k_i = np.linspace(0.5,nk+0.5,nk+1)
    # Develop the super grid, based on the interfaces
    k_i_target = np.linspace(0.5,nk+0.5,nk*refineby+1)
    # Get the value of the layers on the super grid
    k_l_target = 0.5*(k_i_target[1:]+k_i_target[:-1])
    
    # Refine the grid
    if variable_type == 'intensive':
        da_refined = da.interp({vertc:k_l_target},method='linear',kwargs={'fill_value':'extrapolate'})
    if variable_type == 'extensive':
        da_refined = xr.zeros_like(da.interp({vertc:k_l_target}))
        for k in range(nk):
            index = np.arange(k*refineby,(k+1)*refineby)
            vertc_ones = xr.DataArray(np.ones(shape=(refineby)),dims=[vertc],coords={vertc:k_l_target[index]})
            chunk = (da.isel({vertc:k})/refineby)*vertc_ones
            # Input array must have same dimensional order as indexed array
            ### THERE MUST BE A MORE EFFICIENT WAY TO DO THIS ###
            if len(da.dims)==1:
                da_refined.loc[{vertc:k_l_target[index]}]=chunk
            elif len(da.dims)==2:
                da_refined.loc[{vertc:k_l_target[index]}]=chunk.transpose(list(da.dims)[0],list(da.dims)[1])
            elif len(da.dims)==3:
                da_refined.loc[{vertc:k_l_target[index]}]=chunk.transpose(list(da.dims)[0],list(da.dims)[1],list(da.dims)[2])
            elif len(da.dims)==4:
                da_refined.loc[{vertc:k_l_target[index]}]=chunk.transpose(list(da.dims)[0],list(da.dims)[1],list(da.dims)[2],list(da.dims)[3])
    return da_refined

#######################

print('Loading data.')
# Load the data
rootdir = '/archive/gam/ESM4/DECK/ESM4_piControl_D/gfdl.ncrc4-intel16-prod-openmp/'
runname = '11'
filename = '08990101.ocean_daily.nc'
ds = xr.open_dataset(rootdir+runname+'/history/'+filename).squeeze()

filename_grid = '08990101.ocean_static_no_mask_table.nc'
grid = xr.open_dataset(rootdir+runname+'/history/'+filename_grid)
rho0=ds['rhozero'].values
cp=ds['cpocean'].values

print('Calculating budget.')
# Budget terms
heat_terms = ['opottemptend','T_advection_xy','Th_tendency_vert_remap',
              'boundary_forcing_heat_tendency','internal_heat_heat_tendency',
              'opottempdiff','opottemppmdiff','frazil_heat_tendency']
salt_terms = ['osalttend','S_advection_xy','Sh_tendency_vert_remap',
              'boundary_forcing_salt_tendency','osaltdiff','osaltpmdiff']
o2_terms = ['o2h_tendency','o2_advection_xy','o2h_tendency_vert_remap',
            'o2_dfxy_cont_tendency','o2_vdiffuse_impl','jo2']

# Close oxygen budget
o2_tend = o2_terms[0]
# Correct MOM6 tendencies to account for mass in cell
# i.e. convert from [mol kg^-1 m s^-1] to [mol m^-2 s^-1]
for term in o2_terms[:-1]:
    ds[term] *= rho0
    
### THIS A HACK WHILE I WORK OUT THE VDIFFUSE_IMPL TERMS ###
# Calculate residual error
# OXYGEN
tendsum,error = bc.calc_budget(ds,o2_terms[1:],o2_terms[0],plot=False)
ds['o2_vdiffuse_impl']=error

# Calculate material derivative and diffusive terms
signsLHS = [-1,1,1]
signsRHS = [1,1,1,1,1]
# HEAT
termsLHS = heat_terms[:3]
termsRHS = heat_terms[3:]
Dheat, dheat, error_heat = bc.calc_materialderivative(ds,termsLHS,signsLHS,termsRHS,signsRHS,plot=False)

# Binning variables
delta_l = 0.25
l_i_vals = np.arange(-4,36,delta_l)
l_l_vals = 0.5*(l_i_vals[1:]+l_i_vals[:-1])
l_l_vals = l_l_vals[:-1]

rs=6
rss=np.array([2**i for i in range(rs)])
residual = np.zeros(shape=(len(l_l_vals),rs))
for r in range(rs):
    # Time-mean : for evaluating dia-boundary transport and integrated process tendencies
    l = ds['temp'] # Time-mean volume-defining tracer
    l_name = l.name+'_bin' # Naming of binning variable as will be defined by xhistogram
    dl = dheat/cp # Sum of diffusive tendencies for volume-defining tracer
    c = ds['o2'] # Time-mean of budget tracer

    # Snapshots: for evaluating budget tracer content tendency
    # NOTE: time-mean i corresponds to the snapshots at i and i-1
    # so, for example, diff(snap[1]-snap[0])/dt = mean[1]
    l_snap = ds['temp_snap'] # Snapshots of volume-defining tracer
    c_snap = ds['o2_snap'] # Snapshots of budget tracer
    h_snap = ds['thkcello_snap'] # Snapshots of layer thickness (for tracer content calculation)
    rch_snap = rho0*c_snap*h_snap

    # Grid dimensions
    area = grid.areacello # Grid dimensions

    # Time-mean tendencies of budget tracer due to different processes
    h_c = ds['o2_dfxy_cont_tendency'] # Parameterized horizontal diffusion
    f_c = ds['o2_vdiffuse_impl']      # Vertical diffusion and surface fluxes
    b_c = ds['jo2']                   # Biological processes

    # Refine grid?
    print('Refinement: '+str(2**r))
    if r==0:
        refine=False
    else:
        refine=True
        refineby=2**r

    # Refine vertical grid
    print('... refining.')
    if refine:
        l = calc_refine(l, refineby=refineby, variable_type='intensive')
        dl = calc_refine(dl, refineby=refineby, variable_type='extensive')
        c = calc_refine(c, refineby=refineby, variable_type='intensive')
        l_snap = calc_refine(l_snap, refineby=refineby, variable_type='intensive')
        rch_snap = calc_refine(rch_snap, refineby=refineby, variable_type='extensive')

        h_c = calc_refine(h_c, refineby=refineby, variable_type='extensive')
        f_c = calc_refine(f_c, refineby=refineby, variable_type='extensive')
        b_c = calc_refine(b_c, refineby=refineby, variable_type='extensive')

    print('... calculating wmt.')
    # Calculation of budget tracer content tendency, derived from snapshots
    # C = wmt.calc_P(rho0*c_snap*h_snap,l_snap,l_i_vals,area) # Binning at snapshots
    C = wmt.calc_P(rch_snap,l_snap,l_i_vals,area) # Binning at snapshots
    dCdt = C.differentiate('time').isel(time=1) # Finite difference in time
    dCdt = dCdt.rename({l_snap.name+'_bin':l_name}) # Rename dimension for consistency
    # Calculation of E : budget tracer tendency due to dia-boundary mass transport 
    E_c = wmt.calc_E(c,l,dl,l_i_vals,area,plot=False).isel(time=1)
    # Calculation of P^n : volume integrated budget tracer tendencies
    H_c = wmt.calc_P(h_c,l,l_i_vals,area).isel(time=1)
    F_c = wmt.calc_P(f_c,l,l_i_vals,area).isel(time=1)
    B_c = wmt.calc_P(b_c,l,l_i_vals,area).isel(time=1)

    residual[:,r] = (dCdt-E_c-H_c-F_c-B_c).values

    # residual[:,r] = calc_wmt(l_i_vals,area,rch_snap_now,l_snap_now,l_name,c_now,l_now,dl_now,h_c_now,f_c_now,b_c_now)
    print('... RMSE = '+str(np.sqrt(np.mean(residual[:,r]**2))))

savestr = 'data/processed/residuals_dl'+str(delta_l)+'-rs'+str(rs)+'.nc'
print('Refinements done. WMT calcs done. Saving residuals to netcdf in '+savestr+'.')

residual_array = xr.DataArray(residual,dims=['l_l','r'],coords={'l_l':l_l_vals,'r':rss})
residual_array.name = 'residual'
residual_array.to_netcdf(savestr)