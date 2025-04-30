import numpy as np
from scipy.stats import norm
from astropy.time import Time
from astropy import constants as const
import PyAstronomy.pyasl as pya

from convolution import *
from interpolater import *

speed_of_light_ms = const.c.value

class Observations:
    def __init__(self, native_wavegrid, instrument_wavegrid, num,
                 observational_params=[-70.73, -29.26, 217.37, -62.67, 2400.0]):
        """
        Initialize the Observations class.

        Parameters
        ----------
        native_wavegrid : ndarray
            Wavelength grid for the unshifted, high-resolution spectrum.
        instrument_wavegrid : ndarray
            Wavelength grid corresponding to the instrumental sampling (e.g., SPIROU).
        num : int
            Number of epochs (observation times).
        observational_params : list
            Observatory parameters in the form [longitude, latitude, RA, DEC, altitude].
        """
        self.native_wgrid = native_wavegrid
        self.instrument_wavegrid = instrument_wavegrid
        self.num = num
        self.long, self.lat, self.ra, self.dec, self.alt = observational_params

    def dates(self, start_date, end_date, seed=5):
        """
        Generate random observation dates between start and end date.

        Parameters
        ----------
        start_date : str
            Start date in ISO format (YYYY-MM-DD).
        end_date : str
            End date in ISO format.
        seed : int
            Random seed for reproducibility.

        Returns
        -------
        normal_dates : ndarray
            Array of datetime objects representing observation dates.
        """
        start_jd = Time(start_date).jd
        end_jd = Time(end_date).jd
        self.julian_dates = np.linspace(start_jd, end_jd, self.num)

        np.random.seed(seed)
        dates = self.julian_dates.copy()
        dates += np.random.normal(0, 7, size=self.num)
        dates.sort()

        # Introduce artificial observational gaps
        num_chunks = 4
        chunk_size = int(0.05 * self.num)
        chunk_starts = np.random.choice(np.arange(self.num - chunk_size), num_chunks, replace=False)
        indices_to_remove = np.unique(np.concatenate([
            np.arange(start, start + chunk_size) for start in chunk_starts
        ]))
        dates = np.delete(dates, indices_to_remove)

        # Replace removed dates with dates near other dates
        new_dates = np.random.choice(dates, len(indices_to_remove), replace=True)
        new_dates += np.random.uniform(-2, 2, size=len(indices_to_remove))
        dates = np.sort(np.concatenate((dates, new_dates)))

        self.julian_dates = dates
        return Time(dates, format='jd', scale='utc').to_datetime()

    def planet_signal(self, amplitude, omega=0.08):
        """
        Generate a sinusoidal radial velocity signal for a planet.

        Parameters
        ----------
        amplitude : float
            RV semi-amplitude in m/s.
        omega : float
            Angular frequency of the sinusoidal variation.

        Returns
        -------
        signal : ndarray
            RV signal due to planetary motion at each epoch.
        """
        return amplitude * np.sin(omega * self.julian_dates)

    def RV_signal(self, planet_amp, syst_velo=0):
        """
        Generate full RV signal including BERV, system velocity, and planet signal.

        Parameters
        ----------
        planet_amp : float
            RV semi-amplitude of the planet (in m/s).
        syst_velo : float
            Systemic velocity of the star (in km/s).

        Returns
        -------
        RV : ndarray
            Total RV at each epoch [m/s].
        planet : ndarray
            Pure planetary signal [m/s].
        """
        N = len(self.julian_dates)
        self.RV = np.zeros(N)

        self.berv = np.array([
            pya.helcorr(self.long, self.lat, self.alt, self.ra, self.dec, jd)[0] * 1e3
            for jd in self.julian_dates
        ])
        self.sys = syst_velo * 1e3
        self.planet = self.planet_signal(planet_amp)

        self.RV = self.berv + self.sys + self.planet
        return self.RV, self.planet

    def doppler_shift(self, flux, version='univariate'):
        """
        Doppler shift a model spectrum across all epochs and interpolate to native grid.

        Parameters
        ----------
        flux : ndarray
            Input spectrum (same grid as self.native_wgrid).
        version : str
            Interpolation version to use (e.g., 'univariate').

        Returns
        -------
        spectra : ndarray
            Doppler-shifted spectra at each epoch on the native grid.
        """
        shifted_grid = relativistic_shift(self.native_wgrid, self.RV)
        self.spectra = np.array([
            interpolate(shifted, flux, self.native_wgrid, version)
            for shifted in shifted_grid
        ])
        return self.spectra.copy()

    def instrument_captures(self, instrument_res, use_inst_wgrid=True, use_spirou_wgrid=True, new_res=None):
        """
        Convolve spectra to instrument resolution and optionally re-grid to instrument wavelength grid.

        Parameters
        ----------
        instrument_res : float
            Spectral resolution to use for convolution.
        use_inst_wgrid : bool
            Whether to interpolate to an instrumental wavelength grid.
        use_spirou_wgrid : bool
            If True, use provided instrument_wavegrid. Otherwise, generate a new grid.
        new_res : float or None
            Resolution to use for generating new grid, if use_spirou_wgrid is False.

        Returns
        -------
        instrument_wgrid : ndarray
            Final wavelength grid used.
        instrument_spectra : ndarray
            Instrument-sampled spectra at each epoch.
        convolved : ndarray
            Spectra convolved but still on native grid.
        """
        if use_inst_wgrid:
            if use_spirou_wgrid:
                use_grid = self.instrument_wavegrid
            else:
                dv_res = speed_of_light_ms / new_res
                use_grid = get_magic_grid(self.instrument_wavegrid[0], self.instrument_wavegrid[-1], dv_res)
        else:
            use_grid = self.native_wgrid

        self.instrument_wgrid = use_grid
        N = len(self.RV)
        instrument_spectra = np.zeros((N, len(use_grid)))
        convolved = np.zeros((N, len(self.native_wgrid)))

        for i in range(N):
            wgrid, spec = gauss_convolve(
                self.native_wgrid, self.spectra[i],
                instrument_res, n_fwhm=7, res_rtol=1e-6,
                mode='same', i_plot=True
            )
            convolved[i] = spec
            instrument_spectra[i] = interpolate(wgrid, spec, self.instrument_wgrid)

        self.instrument_spectra = instrument_spectra
        return self.instrument_wgrid.copy(), instrument_spectra.copy(), convolved.copy()

    def poisson(self, seed=5):
        """
        Add Poisson (count-based) noise to the spectra.

        Parameters
        ----------
        seed : int
            Random seed for reproducibility.

        Returns
        -------
        instrument_spectra : ndarray
            Noisy spectra.
        sig : ndarray
            Estimated 1-sigma uncertainty per pixel.
        """
        with np.errstate(invalid='ignore'):
            sig = np.abs(np.sqrt(self.instrument_spectra))

        np.random.seed(seed)
        valid = ~np.isnan(self.instrument_spectra) & (self.instrument_spectra > 0)
        self.instrument_spectra[valid] = np.random.poisson(lam=self.instrument_spectra[valid])
        return self.instrument_spectra.copy(), sig.copy()
