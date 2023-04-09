import pickle
import matplotlib.pyplot as plt
from numpy.fft import fftshift
import numpy as np
from scipy.interpolate import CubicSpline, RegularGridInterpolator
from scipy.io import savemat

# Helper functions


def moving_average(a, n=5):
    # source: https://stackoverflow.com/questions/14313510/how-to-calculate-rolling-moving-average-using-python-numpy-scipy
    # Get moving average of array a, with n values averaged over
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def get_resampled_mtf(one_sided_mtf, frequencies, new_frequencies, N_average=8):
    """ Gets a resampled two-sided mtf from a one sided MTF

    Args:
        one_sided_mtf (np.array): one sided mtf from algorithm output
        frequencies (np.array): associated frequencies with the one sided mtf
        new_frequencies (np.array): frequencies to resample the MTF to, NOT fftshifted
        N_average (int, optional): values to average in moving average function. Defaults to 8.

    Returns:
        two_sided_mtf (np.array): fft_shifted two_sided MTF, moving averaged and resampled
    """
    # Take moving average, to de-noise
    # Append ones to keep first value as 1
    ones_appended_array = np.concatenate(
        [np.ones(N_average - 1), one_sided_mtf])
    averaged_one_sided_mtf = moving_average(ones_appended_array, N_average)

    # Resample using a cubic spline
    cs = CubicSpline(frequencies, averaged_one_sided_mtf)
    right = cs(new_frequencies[0:len(new_frequencies)//2])
    left = cs(-new_frequencies[len(new_frequencies)//2:])
    two_sided_mtf = np.concatenate([left, right])
    return two_sided_mtf


def generate_interpolated_filter(mtf_pickle, N_pixels, e, N_average=8):
    """Given an MTF pickle, which contains our calculated MTF

    Args:
        mtf_pickle (a 3-tuple): , with frequency, depth and mtf_stack information
        N_pixels (int): final image width --> we interpolate to this dimension
        e (float): pixel size (mm)
        N_average (int, optional): values to average in moving average function. Defaults to 8.

    Returns:
        result (np.array): 2D filter with dims (N_pixels, N_pixels0)
    """
    freqs = mtf_pickle[0]  # line pairs per mm
    depths = mtf_pickle[1]  # mm
    # mtf_stack has shape (N_depths, N_sampledfrequencies)
    mtf_stack = mtf_pickle[2]

    N_depths = len(depths)
    # This is the fourier freqs if we FT an image with width N_pixels, pixel size e
    new_frequencies = np.fft.fftfreq(N_pixels, d=e)

    # Interpolate in frequency direction
    measured_filter = np.zeros((N_depths, N_pixels))
    for i in range(N_depths):
        measured_filter[i] = get_resampled_mtf(
            mtf_stack[i], freqs, new_frequencies, N_average=N_average)

    # Linearly interpolate/ extrapolate in the depth direction
    new_defocuses = np.linspace(-N_pixels * e / 2, +
                                N_pixels * e / 2, N_pixels)  # defocuses (mm)
    X, Y = np.meshgrid(fftshift(new_frequencies), new_defocuses, indexing='xy')
    interp = RegularGridInterpolator([fftshift(
        new_frequencies), depths], measured_filter.T, method='linear', bounds_error=False, fill_value=None)
    result = interp((X, Y)).reshape(N_pixels, N_pixels)

    # Set < 0  values to 0
    result[result < 0] = 0
    return result
