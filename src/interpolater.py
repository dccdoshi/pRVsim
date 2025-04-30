import torch 
from astropy import constants as const
import numpy as np 
from scipy.signal import argrelextrema
from scipy.interpolate import (
    InterpolatedUnivariateSpline,
    make_interp_spline,
    interp1d,
    CubicSpline,
    PchipInterpolator
)

speed_of_light_ms = const.c.value

def interpolate(x,y,xs,version='univariate'):
    '''Interpolating function'''
    if version=='univariate':
        spl = InterpolatedUnivariateSpline(x, y,ext=3)
        return spl(xs)

    elif version == 'make':
        spline = make_interp_spline(x, y, k=3)
        return spline(xs)

    elif version == 'np':
        spline = make_interp_spline(x, y, k=3)
        return spline(xs)

    elif version == 'interp1d':
        f = interp1d(x, y, kind='cubic', fill_value='extrapolate')
        return f(xs)

    elif version == 'cubic':
        cs = CubicSpline(x, y, bc_type='natural')  # or 'clamped'
        return cs(xs)

    elif version == 'numpy':
        return np.interp(xs, x, y)  # linear only

    elif version == 'pchip':
        pchip = PchipInterpolator(x, y)
        return pchip(xs)
    
    

def get_magic_grid(wave0: float, wave1: float, dv_grid: float = 500):
    """
    magic grid is a standard way of representing a wavelength vector it is set
    so that each element is exactly dv_grid step in velocity. If you shift
    your velocity, then you have a simple translation of this vector.
    :param wave0: float, first wavelength element
    :param wave1: float, second wavelength element
    :param dv_grid: float, grid size in m/s
    :return:
    """
    # default for the function is 500 m/s
    # the arithmetic is a but confusing here, you first find how many
    # elements you have on your grid, then pass it to an exponential
    # the first element is exactely wave0, the last element is NOT
    # exactly wave1, but is very close and is set to get your exact
    # step in velocity
    # get the length of the magic vector
    logwaveratio = np.log(wave1 / wave0)
    len_magic = int(np.floor(logwaveratio * speed_of_light_ms / dv_grid))
    # need to update the final wavelength so we have exactly round steps
    wave1 = np.exp(len_magic * dv_grid / speed_of_light_ms) * wave0
    # redefining wave1 to have a round number of velocity bins
    logwaveratio = np.log(wave1 / wave0)
    # get the positions for "magic length"
    plen_magic = np.arange(len_magic)
    # define the magic grid to use in ccf
    magic_grid = np.exp((plen_magic / len_magic) * logwaveratio) * wave0
    # return the magic grid
    return magic_grid

def upsample_grids(wavelength_grid, flux, dv=1000):
    '''
    This upsamples the grids before interoplating during the doppler shift.
    This will bring it up to a resolution of 1000m/s for spirou
    '''

    # Initialize new wavelength grid with magic grid
    upsample_wavelength_grid = get_magic_grid(wavelength_grid[0],wavelength_grid[-1],dv)
    if flux.ndim==1:
        upsampled_flux = interpolate(wavelength_grid,flux,upsample_wavelength_grid)
    else:
        upsampled_flux = np.zeros((len(flux),len(upsample_wavelength_grid)))
        for i in range(len(flux)):
            upsampled_flux[i] = interpolate(wavelength_grid,flux[i],upsample_wavelength_grid)

    # print("upsampled",upsample_wavelength_grid.shape)
    return upsample_wavelength_grid, upsampled_flux

def relativistic_shift(wgrid, velocity):
    if isinstance(velocity, (float, int)):

        velocity = np.array([velocity])

    # relativistic calculation (1 - v/c)
    part1 = 1 - (velocity / speed_of_light_ms)
    # relativistic calculation (1 + v/c)
    part2 = 1 + (velocity / speed_of_light_ms)

    # shift the wavegrid based on doppler shift
    new_grid = wgrid * np.sqrt(part1/ part2)[:, np.newaxis]

    return new_grid

def peak_envelope(spectrum):
    # Find local maxima indices
    maxima_indices = argrelextrema(spectrum, np.greater)[0]
    # Initialize envelope array
    envelope = np.zeros_like(spectrum)
    
    # Set the envelope values to the maximum values within the neighborhood of each maximum
    for idx in maxima_indices:
        start = max(0, idx - 4088)  # Adjust the neighborhood size as needed
        end = min(len(spectrum), idx + 4088)  # Adjust the neighborhood size as needed
        max_value = np.max(spectrum[start:end])
        envelope[start:end] = max_value
        
    return envelope

def bin_to_instrument(wgrid,flux,new_grid):
    """
    Bins the theoretical spectra into the wavelength grid of the instrument.
    Ensures the output flux matches the shape of the new wavelength grid.
    """
    # Calculate bin edges from the new grid
    bin_edges = np.concatenate(([new_grid[0]], (new_grid[:-1] + new_grid[1:]) / 2, [new_grid[-1]]))
    
    # Digitize the original grid to find which bin each point belongs to
    bin_indices = np.digitize(wgrid, bin_edges) - 1

    # Ensure indices are within valid range
    bin_indices = np.clip(bin_indices, 0, len(new_grid) - 1)
    
    # Sum up flux in each bin using np.bincount
    new_flux = np.bincount(bin_indices, weights=flux, minlength=len(new_grid))
    
    return new_flux

def _h_poly(t):
    """Helper function to compute the 'h' polynomial matrix used in the
    cubic spline.

    Args:
        t (Tensor): A 1D tensor representing the normalized x values.

    Returns:
        Tensor: A 2D tensor of size (4, len(t)) representing the 'h' polynomial matrix.

    """

    tt = t[None, :] ** (torch.arange(4, device=t.device)[:, None])
    A = torch.tensor(
        [[1, 0, -3, 2], [0, 1, -2, 1], [0, 0, 3, -2], [0, 0, -1, 1]],
        dtype=t.dtype,
        device=t.device,
    )
    return A @ tt


def cubic_spline_torch(
    x: torch.Tensor, y: torch.Tensor, xs: torch.Tensor, extend: str = "const"
) -> torch.Tensor:
    """CONNOR'S CODE FROM ASTROPHOT
    Compute the 1D cubic spline interpolation for the given data points
    using PyTorch.

    Args:
        x (Tensor): A 1D tensor representing the x-coordinates of the known data points.
        y (Tensor): A 1D tensor representing the y-coordinates of the known data points.
        xs (Tensor): A 1D tensor representing the x-coordinates of the positions where
                    the cubic spline function should be evaluated.
        extend (str, optional): The method for handling extrapolation, either "const" or "linear".
                                Default is "const".
                                "const": Use the value of the last known data point for extrapolation.
                                "linear": Use linear extrapolation based on the last two known data points.

    Returns:
        Tensor: A 1D tensor representing the interpolated values at the specified positions (xs).

    """
    mask = ~(torch.isnan(x) | torch.isnan(y))
    x = x[mask]
    y = y[mask]

    m = (y[1:] - y[:-1]) / (x[1:] - x[:-1])
    m = torch.cat([m[[0]], (m[1:] + m[:-1]) / 2, m[[-1]]])
    idxs = torch.searchsorted(x[:-1], xs) - 1
    dx = x[idxs + 1] - x[idxs]
    hh = _h_poly((xs - x[idxs]) / dx)
    ret = hh[0] * y[idxs] + hh[1] * m[idxs] * dx + hh[2] * y[idxs + 1] + hh[3] * m[idxs + 1] * dx
    if extend == "const":
        ret[xs > x[-1]] = y[-1]
    elif extend == "linear":
        indices = xs > x[-1]
        ret[indices] = y[-1] + (xs[indices] - x[-1]) * (y[-1] - y[-2]) / (x[-1] - x[-2])
    return ret.numpy()