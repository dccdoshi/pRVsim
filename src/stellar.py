import numpy as np
from astropy.io import fits
from astropy import units as u
from astropy.constants import h, c
from phoenix_models import PhoenixInterpGrid
from interpolater import *

speed_of_light_ms = c.value

class StellarSpectrum:
    def __init__(self, order=20, T=2300, logg=1, metal=1, starting_resolution=200_000, 
                 oversample=1, instrument="spirou"):
        """
        Initializes a stellar spectrum using PHOENIX models and resamples to match a given instrument order.

        Parameters
        ----------
        order : int
            Spectral order of the instrument (e.g., SPIROU or NIRPS).
        T : float
            Effective temperature of the star in Kelvin.
        logg : float
            Surface gravity of the star (log g in cgs units).
        metal : float
            Metallicity [Fe/H] of the star.
        starting_resolution : int
            The spectral resolution to which the PHOENIX spectrum is initialized.
        oversample : float
            Oversampling factor applied to the PHOENIX model which increases sampling resolution.
        instrument : str
            Instrument name, either 'spirou' or 'nirps'. Determines which wavelength file to use.
        """

        self.order = order
        self.T = T
        self.logg = logg
        self.metal = metal
        self.starting_resolution = starting_resolution
        self.oversample = oversample
        self.instrument = instrument.lower()

        # Error handling for order range
        if self.instrument == "spirou" and order > 47:
            raise ValueError(f"Order {order} is too high for SPIROU (max 47).")
        if self.instrument == "nirps" and order > 68:
            raise ValueError(f"Order {order} is too high for NIRPS (max 68).")

        # Load wavelength solution based on instrument
        if self.instrument == "spirou":
            wavelength_file = '../data/SPIRou_wavelength_solution.fits'
        elif self.instrument == "nirps":
            wavelength_file = '../data/NIRPs_wavelength_solution.fits'
        else:
            raise ValueError(f"Unknown instrument '{instrument}'. Use 'spirou' or 'nirps'.")

        e = fits.open(wavelength_file)
        wgrid = np.sort(e[1].data[order]) 
        self.inst_wgrid = np.ascontiguousarray(wgrid.byteswap().newbyteorder())

        # Setup PHOENIX model parameters
        phnx_inputs = {
            'teff': [T],
            'logg': [logg],
            'metal': [metal],
            'alpha': 0.0,
            'wv_range': [np.min(self.inst_wgrid / 1000), np.max(self.inst_wgrid / 1000)],  # µm
            'resolution': starting_resolution,
            'oversampling': oversample,
            'n_fwhm': 7,
            'output_wv_grid': None,
            'method': 'linear',
            'query': True
        }

        phoenix_spec = PhoenixInterpGrid(**phnx_inputs)
        spec = phoenix_spec.flux['Spectrum'].to_numpy()[0]  # erg/s/cm²/cm
        self.phoenix_wgrid = phoenix_spec.wgrid * 1000  # convert µm → nm

        # Convert to photons/s/cm²/nm
        flux_density_erg = (spec * u.erg / (u.s * u.cm**2 * u.cm)).to(
            u.erg / (u.s * u.cm**2 * u.nm)
        )
        Lambda = self.phoenix_wgrid * u.nm
        energy_per_photon = (h * c / Lambda).to(u.erg) / u.ph
        self.stellar = (flux_density_erg / energy_per_photon).to(
            u.ph / (u.s * u.cm**2 * u.nm)
        ).value

    def incorporate_SNR(self, SNR, use_inst_wgrid=True, new_res=None):
        """
        Rescales the flux to simulate observational SNR assuming a constant photon noise per 1 km/s pixel.

        Parameters
        ----------
        SNR : float
            Desired signal-to-noise ratio at a 1 km/s pixel scale.
        use_inst_wgrid : bool
            If True, use instrument resolution from wavelength grid; otherwise use `new_res`.
        new_res : float or None
            Optional resolution to use when `use_inst_wgrid` is False.

        Returns
        -------
        np.ndarray
            Stellar flux array scaled to desired SNR.
        """

        if use_inst_wgrid:
            # Estimate resolution R = lambda / delta_lambda
            R_spec = np.median(self.inst_wgrid[1:] / np.diff(self.inst_wgrid))
        else:
            if new_res is None:
                raise ValueError("`new_res` must be provided if `use_inst_wgrid=False`.")
            R_spec = new_res

        pixel_size_kms = speed_of_light_ms / R_spec
        pixel_flux = (SNR ** 2 * pixel_size_kms) / 1000  # Normalize to 1 km/s
        self.stellar = pixel_flux * (self.stellar / np.median(self.stellar))

        return self.stellar
