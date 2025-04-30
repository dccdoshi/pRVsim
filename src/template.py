import numpy as np
from interpolater import interpolate, relativistic_shift

class Template:
    def __init__(self, bervs, upsampled_wgrid, inst_wgrid):
        """
        Class to generate a high-resolution stellar template by shifting observations
        back to a common rest frame and taking the median across epochs.

        Parameters
        ----------
        bervs : array-like
            Barycentric Earth radial velocities for each observation (in m/s).
        upsampled_wgrid : array-like
            High-resolution wavelength grid used to upsample all spectra and store the template.
        inst_wgrid : array-like
            Instrumental wavelength grid where the observed spectra are sampled.
        """
        self.berv = -bervs  # Reverse the BERVs to shift back to zero RV
        self.upsampled_wgrid = upsampled_wgrid
        self.inst_wgrid = inst_wgrid

    def shift_back_by_bervs(self, flux, version='univariate'):
        """
        Shift observed spectra back to the stellar rest frame and upsample to a high-resolution grid.

        Parameters
        ----------
        flux : array-like
            Observed flux values for each epoch, assumed to be on inst_wgrid. Shape (n_epochs, n_pixels).
        version : str, optional
            Interpolation method to use ('univariate', etc.).

        Returns
        -------
        shifted_flux : ndarray
            Spectra shifted to rest frame and interpolated onto upsampled_wgrid. Shape (n_epochs, n_pixels).
        upsampled_flux : ndarray
            Spectra simply interpolated onto upsampled_wgrid (no RV correction). Shape (n_epochs, n_pixels).
        """
        n_epochs = len(self.berv)
        shifted_grid = relativistic_shift(self.upsampled_wgrid, self.berv)

        self.shifted_flux = np.zeros((n_epochs, len(self.upsampled_wgrid)))
        upsampled_flux = np.zeros_like(self.shifted_flux)

        for i in range(n_epochs):
            # Upsample observation to template grid
            upsampled_flux[i] = interpolate(self.inst_wgrid, flux[i], self.upsampled_wgrid, version)

            # Doppler shift by -BERV (to rest frame)
            self.shifted_flux[i] = interpolate(shifted_grid[i], upsampled_flux[i], self.upsampled_wgrid, version)

        return self.shifted_flux, upsampled_flux

    def create_template(self):
        """
        Create a template spectrum by taking the median across all shifted observations.

        Returns
        -------
        template : ndarray
            Median-combined spectrum on the upsampled wavelength grid.
        """
        return np.median(self.shifted_flux, axis=0)
