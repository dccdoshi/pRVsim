import numpy as np
import matplotlib.pylab as plt
from scipy.stats import norm
from astropy.time import Time
import PyAstronomy.pyasl as pya 
from astropy import constants as const
from scipy.signal import argrelextrema
from scipy.interpolate import InterpolatedUnivariateSpline,Akima1DInterpolator
from numba import jit
from typing import Any, List, Optional, Tuple, Union
from scipy.ndimage import binary_dilation
from scipy.optimize import minimize_scalar
from interpolater import *
from convolution import *
        
def get_sigma_int(Q,SNR):
    '''
    Q is the Bouchy Quality Factor
    m is the H magnitude of the star
    R is the resolution sampling of the spectrograph
    D is the diameter of the telescope in m
    t is the exposure time in s
    '''

    Q = Q/1e4
    sig_int_for_100 = 4.3121251347*Q**2 + 4.3939259825*Q + 2.2313266390

    sigma_int = sig_int_for_100 * (SNR/100)**2

    return sigma_int

def get_quality_factor(A0,lam):
    dA0dlam = np.gradient(A0,lam)
    W = (lam*dA0dlam)**2/A0
    Q = np.sqrt(np.sum(W))/np.sqrt(np.sum(A0))

    return Q

class Velocity_Finder():
    def __init__(self,template,phoenix_wgrid,inst_wgrid,berv,base=False,inst_res=70000,T_epochs=0,fix=False):
        '''
        In this class we will find the relative velocity between the template and an observation, 
        this relative velocity should only be the planetary signal. 

        INPUTS:
        template: this is the template that we've chosen and built
        phoenix_wgrid: this is the native wavelength grid
        inst_wgrid: this is the instrumental wavelength grid
        berv: this is the total shift we have to make to the template before finding the relative shift

        base: if we are using the base spectrum to find the RV or not
        inst_res: the broadening factor that we will need if using base

        '''
        self.template = template.copy()
        self.phoenix_wgrid = phoenix_wgrid
        self.instrument_wgrid = inst_wgrid
        self.berv = berv
        self.fix = fix

        self.base = base
        self.inst_res = inst_res
        if not(self.base):
            self.T_factor = 1/np.sqrt(T_epochs)
        else:
            self.T_factor = 0

        if self.fix:
            if self.base:
                self.I_factor = 0
            else:
                self.I_factor = 1
        else:
            self.I_factor=0


        pass

    def new_model(self, dv):
        '''
        This function is used to shift the template by various dv values that represent 
        different planetary signals. This shifted model will then be compared to the true observation
        signal to estimate the likilihood of the RV value

        INPUTS:
        dv: an array of radial velocities that encompass all values that planetary signal might be 

        OUTPUTS:
        shifted_model: an array of the template shifted by the various dv values
        '''

        # If we are using the base spectrum
        if self.base:
            # Calculate the doppler shift by the proposed dv
            shifted_grid = relativistic_shift(self.phoenix_wgrid, dv+self.berv)
            shifted_model = np.zeros((len(dv),len(self.instrument_wgrid)))

            for i in range(len(dv)):
                # Interpolate to find the shifted flux
                inbetween = interpolate(shifted_grid[i],self.template,self.phoenix_wgrid)
                # Include the broadening factor that we applied for the instrument
                wgrid, spec = gauss_convolve(self.phoenix_wgrid, inbetween, self.inst_res, n_fwhm=7, res_rtol=1e-6, mode='same', i_plot=True)
                # Interpolate to the instrument grid
                shifted_model[i] = interpolate(wgrid,spec, self.instrument_wgrid)

        # If we are using an empirical template
        else:
            # Calculate the doppler shift by the proposed rv
            shifted_grid = relativistic_shift(self.phoenix_wgrid, dv+self.berv)
            shifted_model = np.zeros((len(dv),len(self.instrument_wgrid)))
            for i in range(len(dv)):
                # Interpolate the shifted flux onto the instrument grid 
                shifted_model[i] = interpolate(shifted_grid[i],self.template, self.instrument_wgrid)

        return shifted_model

    def chi2(self,v,data,sig):
        '''
        This function estimates the chi2 of various dv values for the observation provided
        based on Equation 2 of (Silva et al. 2022).

        INPUTS:
        data: the observation spectrum
        v: the list of dv values to test
        sig: the sigma of the gaussian noise applied to the data

        OUTPUTS:
        chi2: the chi2 value calculated
        '''
        if isinstance(v, (float, int)):
            v = np.array([v])

        # Calculate the shifted model
        model_y = self.new_model(v)


        # To deal with NaNs if there are NaNs in spectrum
        # Dont consider the NaN regimes when calculating the likelihood
        nan_indices = np.isnan(data)

        # Take NaNs in neighbouring points as well as these would be affected by bad interpolation

        # Define the structure for dilation (a matrix of True values for a radius of 75)
        structure = np.ones((151), dtype=bool)

        # Expand the NaN mask using binary dilation
        expanded_nan_mask = binary_dilation(nan_indices, structure=structure)

        # Apply the expanded NaN mask to the other array
        model_y[:,expanded_nan_mask] = np.nan



        # Only consider the middle portions as the ends may be affected by bad interpolation
        start = int(len(data)*0.05)
        end = int(len(data)*0.95)


        # Determine the uncertainty
        sigt = (sig*self.T_factor)**2 # Uncertainty of template
        sigo  = sig**2 # Uncertainty of observation

        if self.I_factor == 1:
            Q = get_quality_factor(model_y[0][start:end],self.instrument_wgrid[start:end])
            SNR = np.median(np.sqrt(model_y[0][start:end]))
            sigi = get_sigma_int(Q,SNR)**2

        else:
            sigi = 0

        sig = sigo+sigt+sigi

        # Will not use ends of spectrum as they will be affected by convolution 
        # This is taken as Equation 2 from (Silva et al. 2022)
        residual = ((data[start:end]-model_y[:,start:end]))**2/sig[start:end]
        chi2 = np.nansum(residual)

        return chi2

    def get_values_for_QI(self):
        start = int(len(self.instrument_wgrid)*0.05)
        end = int(len(self.instrument_wgrid)*0.95)
        # Calculate the shifted model
        model_y = self.new_model([0])
        Q = get_quality_factor(model_y[0][start:end],self.instrument_wgrid[start:end])
        SNR = np.median(np.sqrt(model_y[0][start:end]))
        sigi = get_sigma_int(Q,SNR)**2

        return Q,sigi,SNR

    def find_dv(self, data, dv, sig):
        '''
        This function estimates the the planetary signal. This method is taken from (Silva et al. 2022) "A novel framework for 
        semi-Bayesian radial velocities through template matching". It is part of the S-BART methodology. 

        INPUTS:
        data: the observation spectrum
        dv: the list of dv values to test
        sig: the sigma of the gaussian noise applied to the data

        OUTPUTS:
        rvorder: the proposed RV for the order
        unc_dv: the proposed dv for this iteration
        '''
        
        result = minimize_scalar(self.chi2,args=(data,sig),method='brent')
        rvmin, xm = result.x, result.fun
        xmp1 = self.chi2(rvmin+dv,data,sig)
        xmm1 = self.chi2(rvmin-dv,data,sig)

        # Equation 3 from paper
        rvorder = rvmin - (dv/2)*(xmp1-xmm1)/(xmp1+xmm1-(2*xm))
        unc_dv = (2*dv**2)/(xmm1-(2*xm)+xmp1) 

        return rvorder, unc_dv

    def find_unc(self,data,sig,rvmin,dv):
        '''
        Again this calculates the uncertainty based on equation 3 from the (Silva et al. 2022) paper.
        '''

        xm = self.chi2(rvmin,data,sig)
        xmp1 = self.chi2(rvmin+dv,data,sig)
        xmm1 = self.chi2(rvmin-dv,data,sig)
        unc_dv = np.sqrt((2*dv**2)/(xmm1-(2*xm)+xmp1))

        return unc_dv

    def get_zscore(self,data,dv,sig):
        if isinstance(dv, (float, int)):
            dv = np.array([dv])
        # Calculate the shifted model
        model_y = self.new_model(dv)

        # To deal with NaNs if there are NaNs in spectrum
        # Dont consider the NaN regimes when calculating the likelihood
        nan_indices = np.isnan(data)

        # Take NaNs in neighbouring points as well as these would be affected by bad interpolation

        # Define the structure for dilation (a matrix of True values for a radius of 75)
        structure = np.ones((151), dtype=bool)

        # Expand the NaN mask using binary dilation
        expanded_nan_mask = binary_dilation(nan_indices, structure=structure)

        # Apply the expanded NaN mask to the other array
        model_y[:,expanded_nan_mask] = np.nan



        # Only consider the middle portions as the ends may be affected by bad interpolation
        start = int(len(data)*0.05)
        end = int(len(data)*0.95)

        # Determine the uncertainty
        sigt = (sig*self.T_factor)**2 
        sigo  = sig**2
        sig = np.sqrt(sigo+sigt)


        # The std of this should be 1 if not affected by interpolation errors
        zscore = ((data[start:end]-model_y[:,start:end]))/sig[start:end]

        return np.mean(zscore)




 