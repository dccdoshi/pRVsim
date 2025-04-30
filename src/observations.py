import numpy as np
from scipy.stats import norm
from astropy.time import Time
import PyAstronomy.pyasl as pya 
from astropy import constants as const
from hapi import *
from convolution import *
from interpolater import * 
import torch 
import torch 
from astropy import constants as const

speed_of_light_ms = const.c.value
class Observations():

    def __init__(self, native_wavegrid, spirou_wavegrid, num,observational_params=[-70.73, -29.26, 217.37, -62.67, 2400.0]):
        '''
        This class is used to make the observations based off of our true signals. 
        Here, we will make num amount of observations with the observational parameters that provide
        longitude, latitude, ra, dec, altitude of the telescope.

        INPUTS:
        native_wavegrid: is the wavegrid that the "true" signal is in (the native grid)
        inst_wavegrid: this is the original SPIROU wavelength grid according to the order we are looking at 
        num: is the number of epochs/observations
        observational_params: parameters to deteremine the BERV of our observations
        '''
        self.native_wgrid = native_wavegrid
        self.spirou_wgrid = spirou_wavegrid
        self.num = num
        self.long = observational_params[0]
        self.lat = observational_params[1]

        self.ra = observational_params[2]
        self.dec = observational_params[3]
        self.alt = observational_params[4]
        pass

    def dates(self,start_date,end_date,seed=5):
        '''
        This function creates a julian date grid based on a given start date and end date in datetime format
        It returns the grid in datetime format however the julian date grid is used for BERV calculations. 
        The dates are slightly randomized to best replicate real observations.

        INPUTS:
        start_date: when the observations start in datetime format
        end_date: when the observations end in datetime format
        seed: this is a seed to determine the randomness of the dates that we observe

        OUTPUTS:
        normal_dates: the grid space of dates that will be used for observations in datetime format
        '''
        # Create Dates 
        # Convert start and end dates to Julian dates
        start_jd = Time(start_date).jd
        end_jd = Time(end_date).jd

        # Create a NumPy array of Julian dates spanning two years
        num_points = self.num  # Adjust the number of points as needed

        # Create ordered julian dates
        self.julian_dates = np.linspace(start_jd, end_jd, num_points)

        #
        # Randomize dates slightly
        #

        np.random.seed(seed)
        N = self.num
        dates = self.julian_dates.copy()

        # Introduce some randomness to the dates
        random_offsets = np.random.normal(0, 7, size=N)  # Adjust standard deviation as needed
        dates = dates + random_offsets

        # Sort dates to maintain order after adding randomness
        dates.sort()

        # Identify chunks to remove
        num_chunks = 4  # Number of chunks to remove
        chunk_size = int(0.05 * N)  # Size of each chunk, for example 5% of N

        chunk_starts = np.random.choice(np.arange(N - chunk_size), num_chunks, replace=False)

        # Remove chunks
        indices_to_remove = []
        for start in chunk_starts:
            indices_to_remove.extend(np.arange(start, start + chunk_size))
        indices_to_remove = np.unique(indices_to_remove)  # Ensure indices are unique and sorted
        dates = np.delete(dates, indices_to_remove)

        # Number of dates to add back
        num_to_add = len(indices_to_remove)

        # Determine the remaining chunks where dates can be added
        remaining_indices = np.setdiff1d(np.arange(N), indices_to_remove)
        remaining_dates = dates

        # Add new random dates within the remaining chunks
        new_dates = np.random.choice(remaining_dates, num_to_add, replace=True)
        new_offsets = np.random.uniform(-2, 2, size=num_to_add)  # Adjust as needed
        new_dates = new_dates + new_offsets

        # Combine the existing dates and new dates
        dates = np.concatenate((dates, new_dates))

        # Sort the dates to keep a chronological order
        dates.sort()
        self.julian_dates = dates

        # Create array to keep track of datetime dates of the same grid 
        normal_dates = Time(self.julian_dates,format='jd', scale='utc').to_datetime()
        return normal_dates

    def RV_signal(self,planet_amp,syst_velo=0):
        '''
        Here we create an RV signal for the time grid that we are using
        This can include berv, the system velocity, and the planet amplitude

        INPUTS:
        syst_velo: (should be in km/s) and is the system's relative velocity
        planet amp: (should be in m/s) are floats and is the amplitude of the planet's signal

        OUTPUTS: 
        self.RV: This is the RV signal for each of the specified dates provided
        '''

        # Initialize Signal
        self.RV = np.zeros(len(self.julian_dates))

        # Add Berv
        berv = np.zeros(len(self.julian_dates))
        for i in range(len(self.julian_dates)):
            b = pya.helcorr(self.long,self.lat,self.alt,self.ra,self.dec,self.julian_dates[i])
            berv[i] = b[0]*1e3 #m
        self.berv = berv


        # Add System Velocity
        syst_velo = syst_velo * 1e3 #m
        self.sys = syst_velo

        # Add Planet (just a simple sin curve)
        A = planet_amp
        w=0.08
        self.planet = A*np.sin(w*self.julian_dates)

        # Add it all together
        self.RV=self.planet+self.sys+self.berv

        return self.RV, self.planet
 
    def doppler_shift(self,flux):
        """
        Apply a doppler shift to the native wavelength grid and then interpolate for shifted flux

        INPUT:
        flux = the flux that will be doppler shifted

        OUTPUT:
        self.spectra = doppler shifted fluxes according to the RV curve

        """

        # Here we simply shift the stellar spectrum with the given wgrid, the native wgrid should be high enough resolution to do this right
        shifted_grid = relativistic_shift(self.native_wgrid,self.RV)

        # Shift the flux according to the doppler shift by interpolating
        self.spectra = np.zeros((len(self.RV),len(self.native_wgrid)))
        for i in range(len(shifted_grid)):
            self.spectra[i] = interpolate(shifted_grid[i],flux, self.native_wgrid)

        return self.spectra.copy()

    def telluric_contaminate(self,tellurics):
        '''
        This function contaminates the spectra with a given telluric spectrum
        '''
        self.spectra *= tellurics
        return self.spectra

    def instrument_captures(self,instrument_res,use_inst_wgrid,use_spirou_wgrid,new_res):
        '''
        This function determines the isntrument wavelength grid and places the spectra 
        into the resolution and wavelength grid of the instrument

        INPUTS:
        instrument_res = broadening factor to spectra due to instrument
        use_inst_wgrid = bool for if we degrade to instrument resolution
        use_spirou_wgrid = bool for if we use SPIROU's exact wavelength solution otherwise use a constant resolution grid 

        OUTPUTS:
        instrument_wgrid = the final wgird that is chosen for the instrument
        instrument_spectra = what the final spectra look like in instrument resolution
        convolved = what the spectra look like before being placed in instrument resolution
        '''

        # First, determine the grid to use
        if use_inst_wgrid:
            if use_spirou_wgrid:
                use_grid = self.spirou_wgrid
            else:
                # Use given resolution 
                R = new_res
                dv_res = speed_of_light_ms / R

                # Get a wavelength grid that has constant resolution sampling based off of the SPIROU wavelength grid bounds
                use_grid = get_magic_grid(self.spirou_wgrid[0],self.spirou_wgrid[-1],dv_res)

        else:
            use_grid = self.native_wgrid

        self.instrument_wgrid = use_grid

        # Now let's place the spectra into the grids
        instrument_spectra = np.zeros((len(self.RV),len(self.instrument_wgrid)))

        # To keep track of what the spectrum looks like before we degrade into instrument resolution
        convolved = np.zeros((len(self.RV),len(self.native_wgrid)))

        # Iterate over each spectrum
        for i in range(len(self.spectra)):
            # First we convolve the spectrum to broaden it
            wgrid, spec = gauss_convolve(self.native_wgrid, self.spectra[i], instrument_res, n_fwhm=7, res_rtol=1e-6, mode='same', i_plot=True)
            convolved[i] = spec

            # Interpolate onto the instrument grid (binning function was not working right just using interpolating for now)
            instrument_spectra[i] = interpolate(wgrid,spec, self.instrument_wgrid)
        self.instrument_spectra = instrument_spectra

        return self.instrument_wgrid.copy(), self.instrument_spectra.copy(), convolved.copy()

    def incorporate_SNR(self, SNR,normalizing_constant):
        '''
        Change the flux of the spectra to match a certain SNR 
        '''

        for i in range(len(self.instrument_spectra)):
            self.instrument_spectra[i] = SNR**2*(self.instrument_spectra[i]/normalizing_constant)

        return self.instrument_spectra.copy()

    def gaussian(self,seed=5):
        '''
        To add poisson noise, approximating to gaussian since counts will be over 100

        INPUTS:
        seed = the seed for the specific gaussian instance

        OUTPUTs:
        instrument_spectra = the noisy spectra
        sig = the uncertainty on each point

        '''
        with np.errstate(invalid='ignore'):
            # The uncertainty is the sqrt of the flux
            sig = np.sqrt(self.instrument_spectra)

        np.random.seed(seed)

        # Create the noise spectrum
        noise = sig*np.random.normal(0, 1,sig.shape)

        # Add noise to observations
        self.instrument_spectra = self.instrument_spectra + noise

        return self.instrument_spectra.copy(), sig.copy()

    def poisson(self,seed=5):
        '''
        To add poisson noise, approximating to gaussian since counts will be over 100

        INPUTS:
        seed = the seed for the specific gaussian instance

        OUTPUTs:
        instrument_spectra = the noisy spectra
        sig = the uncertainty on each point

        '''
        with np.errstate(invalid='ignore'):
            # The uncertainty is the sqrt of the flux
            sig = np.abs(np.sqrt(self.instrument_spectra))

        np.random.seed(seed)

        # Create the noise spectrum
        self.instrument_spectra = np.random.poisson(lam=self.instrument_spectra)

        # Add noise to observations
        # self.instrument_spectra = self.instrument_spectra + noise

        return self.instrument_spectra.copy(), sig.copy()

