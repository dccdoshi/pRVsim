from astropy import constants as const
import numpy as np 
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
