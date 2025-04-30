import numpy as np
import matplotlib.pylab as plt
from scipy.stats import norm
from astropy.time import Time
import PyAstronomy.pyasl as pya 
from astropy import constants as const
from scipy.interpolate import InterpolatedUnivariateSpline
from typing import Any, List, Optional, Tuple, Union
from interpolater import *

class Template():
    def __init__(self,bervs,upsampled_wgrid,inst_wgrid):
        '''
        This class creates the templates from the observations.
        The upsampled_wgrid is the space in which the template will reside. We are keeping this as the native
        wavegrid used to make the observations so that we can easily see the intepolation error

        INPUTS:
        bervs: the BERV values of the observations
        upsampled_wgrid: the upsampled wavelength grid the template should be in and individual observations will be upsampled to
        inst_wgrid: the instrument wavelength grid that the obesrvations are in
        '''
        self.berv = -bervs
        self.upsampled_wgrid = upsampled_wgrid
        self.inst_wgrid = inst_wgrid
        pass

    def shift_back_by_bervs(self, flux,version='univariate'):
        """
        Apply a doppler shift and then interpolate for shifted flux to move templates back to "0 RV"

        :param wavegrid: wave grid to shift
        :param flux: flux to shift

        :return: np.ndarray, the shifted flux
        :return: np.ndarray, the upsampled spectrum of each observation
        """

        # Doppler Shift the wavelength grid
        shifted_grid = relativistic_shift(self.upsampled_wgrid, self.berv)

        # Interpolate back to original grid
        self.shifted_flux = np.zeros((len(self.berv),len(self.upsampled_wgrid)))
        upsampled_flux = np.zeros((len(self.berv),len(self.upsampled_wgrid)))

        # Iterate over each spectrum
        for i in range(len(self.shifted_flux)): 
            # Upsample each observation to the upsampled space
            upsampled_flux[i] = interpolate(self.inst_wgrid,flux[i],self.upsampled_wgrid,version)

            # Doppler shift it by -Berv
            self.shifted_flux[i] = interpolate(shifted_grid[i],upsampled_flux[i],self.upsampled_wgrid,version)



        return self.shifted_flux, upsampled_flux


    def create_template(self):
        '''
        This function creates the template by taking the median of fluxes from each wavelength 
        
        OUTPUTS:
        the template
        '''
        template = np.median(self.shifted_flux,axis=0)

        return template
