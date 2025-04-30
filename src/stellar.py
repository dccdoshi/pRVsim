import numpy as np
import pandas as pd
from astropy import constants as const
from phoenix_models import PhoenixInterpGrid
from convolution import get_wv_constant_res
from astropy.io import fits
from scipy.signal import argrelextrema
import astropy.units as u
from astropy.constants import h, c
from astropy import constants as const
from interpolater import *
speed_of_light_ms = const.c.value

class Stellar_Spectrum():
    def __init__(self,order=20,T=2300, logg=1,metal=1,starting_resolution=200_000,oversample=1):
        '''
        This is where we define the PHOENIX stellar model that we are getting. From Antoine's code
        we get units for the spectra in 'erg/s/cm^2/cm' we convert this to 'photon/s/cm^s/nm' to match
        our use case.

        order = SPIROU order
        T = Temperature of star
        logg = logg of star
        metal = metallicity of star
        starting_resolution = the initial resolution retrieved from the PHOENIX spectra
        oversample = this is the oversampling factor to increase the resolution of the PHOENIX spectra

        '''
        
        self.order = order
        self.T = T
        self.logg = logg
        self.metal = metal
        self.starting_resolution = starting_resolution
        self.oversample = oversample

        #
        # Let us define the wavelength grid that corresponds with the SPIROU order
        #

        e = fits.open('../data/3444961B5D_pp_e2dsff_AB_wavesol_ref_AB.fits')
        # e = fits.open('../data/NIRPs_wavelength_solution.fits')
        wgrid =  np.sort(e[1].data[order]) 
        self.spirou_wgrid = np.ascontiguousarray(wgrid.byteswap().newbyteorder())

        #
        # Now let us retrieve the PHOENIX stellar model using Antoine's code
        #
        ## PHOENIX Inputs ##############
        phnx_inputs = dict()
 
        # Parameters (range or fixed parameter)
        phnx_inputs['teff'] = [T]
        phnx_inputs['logg'] = [logg]
        phnx_inputs['metal'] = [metal]
        phnx_inputs['alpha'] = 0.0  # Constant so not interpolated (can be if needed)

        ## Other parameters ##############
        # wavelenngth range (from data to be fitted)
        wv_range = [np.min(self.spirou_wgrid/1000), np.max(self.spirou_wgrid/1000)]  # from nm to microns

        phnx_inputs['wv_range'] = wv_range  # full range: [0.955, 2.516]
        phnx_inputs['resolution'] = starting_resolution  # Resolution of the output: 'Taken from Spectral Resolution from the broadening of stellar spectra (6)	70,000'
        phnx_inputs['oversampling'] = oversample  # Oversampling of the grid used for interpolation
        phnx_inputs['n_fwhm'] = 7  # Where to cut the convolution kernel
        phnx_inputs['output_wv_grid'] = None # The grid used for interpolation can be directly specified
        phnx_inputs['method'] = 'linear'   # Type of interpolation (more than linear might be too heavy for MCMC)
        phnx_inputs['query'] = True  # Access to internet? (will download missing PHOENIX models)

        ## Create the spectra: 
        phoenix_spec = PhoenixInterpGrid(**phnx_inputs)
        spec = phoenix_spec.flux['Spectrum'].to_numpy()[0] # in erg/s/cm^s/cm
        


        ### Convert into appropriate units
        self.phoenix_wgrid = phoenix_spec.wgrid*1000 # from microns to nm


        flux_density_erg = spec * u.erg / (u.s * u.cm**2 * u.cm)
        flux_density_erg = flux_density_erg.to(u.erg / (u.s * u.cm**2 * u.nm))

        # Define the wavelength at which the flux density is measured
        Lambda = self.phoenix_wgrid * u.nm  # Example wavelength in nm

        # Calculate the energy of a single photon at the given wavelength
        energy_per_photon = (h * c / Lambda).to(u.erg) / u.ph

        # Convert the flux density to photons/s/cm^2/nm
        self.stellar = (flux_density_erg / energy_per_photon).to(u.ph / (u.s * u.cm**2 * u.nm)).value
 
        return 

    
    def incorporate_SNR(self, SNR,use_inst_wgrid,use_spirou_wgrid,new_res):
        '''
        Change the flux of the spectra to match a certain SNR per a 1km/s pixel
        SNR = SNR of your individual observations
        '''
        if use_inst_wgrid:
            if use_spirou_wgrid:
                # Take the median pixel size in spirou
                R_spec = np.median(self.spirou_wgrid[1:]/np.diff(self.spirou_wgrid))
            else:
                # Use given resolution 
                R_spec = new_res

        # Determine the pixel size of the detector
        pixel_size = speed_of_light_ms/R_spec
        
        # Normalize the per pixel flux according to a certain SNR if it were on a 1km/s pixel
        pixel_flux = (SNR**2*pixel_size)/(1000)
        self.stellar = pixel_flux*(self.stellar /np.median(self.stellar))

        return self.stellar 
        

