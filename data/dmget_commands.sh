#!/bin/bash

root=/archive/oar.gfdl.cmip6/ESM4/DECK/ESM4_piControl_D/gfdl.ncrc4-intel16-prod-openmp/pp

ppname=ocean_cobalt_omip_tracers_month_z
out=av
local=monthly_5yr
time=03[0-4]*
var=*

paths=${root}/${ppname}/${out}/${local}/${ppname}.${time}.${var}.nc
dmget ${paths} &
