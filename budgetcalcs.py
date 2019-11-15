#!/usr/bin/env python
# A collection of functions for calculating and plotting 3D tracer budgets.
# It is presumed in each case that the terms in the budget are known.
# Created August 2019 by Graeme A. MacGilchrist (gmacgilchrist@gmail.com)

import xarray as xr
from matplotlib import pyplot as plt

def calc_budget(ds, terms, tend, omit=[], vertc='zl', plot=True, errors=[-1E-12,1E-12]):
    
    '''Sum the terms in [terms] and compare to [tend].
    Return the sum of the terms and the error.'''
    
    tend_sum = 0.0
    for term in terms:
        tend_sum += ds[term]
    for term in omit:
        tend_sum -= ds[term]
    
    error = ds[tend]-tend_sum
    precision = xr.ufuncs.fabs(error)/xr.ufuncs.fabs(ds[tend])
    
    tend_sum.name = 'tend_sum'
    error.name = 'error'
    precision.name = 'precision'
    
    if plot:
        
        if len(ds[tend].dims)>2:
            raise Exception("Reduce dimensions of input dataset to plot")
            
        elif len(ds[tend].dims)==2:
            fig,ax = plt.subplots(nrows = 4, figsize=(10,15))
            
            im=ax[0].pcolormesh(ds.xh,ds.yh,ds[tend])
            ax[0].set_title('total tendency')
            plt.colorbar(im,ax=ax[0])
            im=ax[1].pcolormesh(ds.xh,ds.yh,tend_sum)
            ax[1].set_title('sum of terms')
            plt.colorbar(im,ax=ax[1])
            im=ax[2].pcolormesh(ds.xh,ds.yh,error)
            ax[2].set_title('error (sum of tendencies minus total tendency)')
            im.set_clim(errors)
            plt.colorbar(im,ax=ax[2])
            im=ax[3].pcolormesh(ds.xh,ds.yh,precision)
            ax[3].set_title('precision (ratio of error and tendency (absolute))')
            im.set_clim([0,1E-4])
            plt.colorbar(im,ax=ax[3])
            
        else:
            k = range(ds[vertc].size)
            fig,ax = plt.subplots(ncols = 3, figsize=(12,7))

            ax[0].plot(ds[tend],k,'.-',label='total tendency')
            ax[0].plot(tend_sum,k,'.-',label='sum of terms')
            ax[0].invert_yaxis()
            ax[0].legend(loc='best')
            ax[0].set_title('sum and total tendency')

            ax[1].plot(error,k)
            ax[1].invert_yaxis()
            ax[1].set_title('error\n sum of tendencies minus total tendency')
            ax[1].set_xlim(errors)

            ax[2].plot(precision,k)
            ax[2].invert_yaxis()
            ax[2].set_title('precision\n ratio of error and tendency (absolute)')
            ax[2].set_xlim([0,1E-4])
    
    return tend_sum, error

def plot_budgetterms(ds,terms,omit=[],vertc='zl'):
    
    '''Plot the individual terms in [terms]'''
    
    n=len(terms)
    
    if len(ds[terms[0]].dims)>2:
        raise Exception("Reduce dimensions of input dataset to plot")
        
    elif len(ds[terms[0]].dims)==2:
        fig,ax = plt.subplots(nrows = n, figsize=(10,n*5))
        for i in range(n):
            if terms[i] not in omit:
                if n>1:
                    im=ax[i].pcolormesh(ds.xh,ds.yh,ds[terms[i]])
                    ax[i].set_title(terms[i])
                    plt.colorbar(im,ax=ax[i])
                else:
                    im=ax.pcolormesh(ds.xh,ds.yh,ds[terms[i]])
                    ax.set_title(terms[i])
                    plt.colorbar(im,ax=ax)
    
    else:
        k = range(ds[vertc].size)
        fig,ax = plt.subplots(nrows = 1, figsize=(5,7))
        for i in range(n):
            if terms[i] not in omit:
                ax.plot(ds[terms[i]],k,'.-',label=terms[i])
        ax.legend(bbox_to_anchor=(1.04,0.5),loc='center left')
        ax.invert_yaxis()
        ax.set_title('terms')
        
def calc_materialderivative(ds,termsLHS,signsLHS,termsRHS,signsRHS,vertc='zl',plot=True):
    
    '''Calculate the sum of terms corresponding to the material derivative.
    The LHS should include eulerian time tendency and advective components.
    The RHS should include all of the diffusive tendencies.'''

    dim = len(ds[termsLHS[0]].dims)
    
    LHS=0.0
    RHS=0.0
    
    t=0
    for term in termsLHS:
        LHS+=signsLHS[t]*ds[term]
        t+=1
    t=0
    for term in termsRHS:
        RHS+=signsRHS[t]*ds[term]
        t+=1
    
    error=LHS+RHS
    
    LHS.name='LHS'
    RHS.name='RHS'
    
    if plot:
        if dim>2:
                raise Exception("Reduce dimensions of input dataset to plot")

        elif dim==2:
                raise Exception("Presently only doing summed variables for 1D columns")

        else:
                k = range(ds[vertc].size)
                fig,ax = plt.subplots(ncols = 1, figsize=(5,7))

                ax.plot(LHS,k,'.-',label='LHS')
                ax.plot(RHS,k,'.-',label='RHS')
                ax.plot(error,k,'.-',label='error (LHS plus RHS)')
                ax.invert_yaxis()
                ax.legend(loc='best')
                ax.set_title('summed terms')
                
    return LHS, RHS, error