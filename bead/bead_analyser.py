import pandas as pd
import numpy as np
from skimage import io
from skimage.transform import rotate
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from scipy.stats import binned_statistic


class BeadDataset:
    def __init__(self, file_handle, pixel_size=6.45e-3):
        """ 
        Args:
            file_handle (str): file handle for bead slices, 
                e.g. '../../OPT Shared files/2_Saved reconstructions/fd_focal scan/fd_infocus/reco'
            pixel_size (float): pixel_dimensions. Defaults to 6.45e-3 (mm).
        """
        self.image_filepath = file_handle
        im = io.imread(self.image_filepath + '0000' + '.tif')
        self.image_size = len(im)
        self.e = pixel_size
        self.c = self.image_size//2 + 1
        self.dy = 50

    def set_bead_locations(self, x, y, slices):
        # Store bead locations (x,y,slices)
        # Compute their (r, theta) locations
        self.angles = np.arctan2((y-self.c), (x-self.c))*180/np.pi
        # print(self.c)
        # print(self.angles)
        self.r = np.sqrt((x-self.c)**2 + (y-self.c)**2) * \
            self.e  # radial distance from centre
        self.r_idx = np.sqrt((x-self.c)**2 + (y-self.c)**2)
        self.slice_numbers = slices.astype(int)
        self.num_beads = len(self.angles)
        pass

    def get_rotated_slices(self, polarity=-1):
        # Polarity is a fudge factor to get the angles right. Suspect due to image indexing direction vs coordinate dirn.
        self.polarity = polarity
        # Rotates slices based on their (x,y) positions
        self.strpadded_slices = [str(number).zfill(4)
                                 for number in self.slice_numbers]
        self.unrotated_slices = np.zeros(
            (self.num_beads, self.image_size, self.image_size))
        self.rotated_slices = np.zeros(
            (self.num_beads, self.image_size, self.image_size))

        for i in range(self.num_beads):
            r = self.r[i]
            im = io.imread(self.image_filepath +
                           self.strpadded_slices[i] + '.tif')
            self.unrotated_slices[i] = im
            rotated_image = rotate(im, polarity*self.angles[i])
            self.rotated_slices[i] = rotated_image  # /
        return self.rotated_slices

    def gen_bead_figure(self, display=True):
        # Generate a normalised figure with beads rotated and summed
        self.get_rotated_slices(self.polarity)
        self.bead_figure = self.rotated_slices.copy()
        # print(self.c, self.dy)
        maxes = np.max(
            self.bead_figure[:, self.c-self.dy:self.c+self.dy, self.c:], axis=(1, 2))
        for i in range(self.num_beads):
            self.bead_figure[i] /= maxes[i]
        self.bead_figure = np.sum(self.bead_figure, axis=0)

        if display:
            plt.imshow(self.bead_figure[self.c -
                       self.dy:self.c+self.dy, self.c:])
        return self.bead_figure

    def fit_all_beads(self):
        # Automatically fit the radial and tangential widths, using the rotated slices
        # Fits with a Gaussian
        # sets radial_withs and tangential_widths, based on the std deviation of the Gaussian
        # The std deviation has units of microns

        tangential_widths = []
        radial_widths = []
        tangential_width_errors = []
        radial_width_errors = []

        p0 = [0, 15, 1]
        for i in range(self.num_beads):
            bead_horizontal_idx = self.c + int(self.r_idx[i])

            # fit tangential profiles
            array = self.rotated_slices[i, self.c -
                                        self.dy:self.c+self.dy, bead_horizontal_idx]
            popt, pcov = curve_fit(self._gaussian, self._range_from_profile(
                array), array/array.max(), p0=p0)
            tangential_widths.append(popt[1])
            tangential_width_errors.append(np.sqrt(pcov[1, 1]))

            # fit radial profiles
            array = self.rotated_slices[i, self.c,
                                        bead_horizontal_idx - self.dy: bead_horizontal_idx+self.dy]
            popt, pcov = curve_fit(self._gaussian, self._range_from_profile(
                array), array/array.max(), p0=p0)
            radial_widths.append(popt[1])
            radial_width_errors.append(np.sqrt(pcov[1, 1]))

        self.radial_widths = radial_widths
        self.tangential_widths = tangential_widths
        self.tangential_width_errors = tangential_width_errors
        self.radial_width_errors = radial_width_errors
        return

    def bin_bead_widths(self, num_bins):
        """ Bin bead widths, calculated in self.fit_all_beads, into num_bins
            useful for analysis

        Args:
            num_bins (int): _description_
        """
        x = self.r

        # Radial profiles
        self.binned_radial, bin_edges, binnumber = binned_statistic(
            x, self.radial_widths, bins=num_bins)
        self.binned_radial_std, _, _ = binned_statistic(
            x, self.radial_widths, statistic='std', bins=num_bins)

        self.binned_r = (bin_edges[1:] + bin_edges[0:-1])/2
        self.binned_r_err = (bin_edges[1] - bin_edges[0])/2

        # Tangential profiles
        self.binned_tangential, bin_edges, binnumber = binned_statistic(
            x, self.tangential_widths, bins=num_bins)
        self.binned_tangential_std, _, _ = binned_statistic(
            x, self.tangential_widths, statistic='std', bins=num_bins)

        pass

    # Helper functions

    def _range_from_profile(self, array):
        # Turns pixel indexes into distance values (microns)
        return (np.arange(0, len(array))-np.argmax(array)) * self.e * 1000

    def _gaussian(self, x, mu, sigma, A):
        return A*np.exp(-((x-mu)**2/sigma**2 / 2))

# Improvements:
# fit rotated Gaussian instead of rotating the image
