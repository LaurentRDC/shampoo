"""
This module handles reconstruction of phase and intensity images from raw
holograms using "the convolution approach": see Section 3.3 of Schnars & Juptner
(2002) Meas. Sci. Technol. 13 R85-R101 [1]_.

Aberration corrections from Colomb et al., Appl Opt. 2006 Feb 10;45(5):851-63
are applied [2]_.

    .. [1] http://x-ray.ucsd.edu/mediawiki/images/d/df/Digital_recording_numerical_reconstruction.pdf
    .. [2] http://www.ncbi.nlm.nih.gov/pubmed/16512526

"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from collections import Sized, deque, defaultdict

import warnings

from multiprocessing import Pool 
from functools import partial

from .vis import save_scaled_image

import h5py
import numpy as np
from numpy.compat import integer_types

from scipy.ndimage import gaussian_filter, maximum_filter
from scipy.signal import tukey

from skimage.restoration import unwrap_phase as skimage_unwrap_phase
from skimage.io import imread
from skimage.feature import blob_doh

from astropy.utils.exceptions import AstropyUserWarning
from astropy.convolution import convolve_fft, MexicanHat2DKernel

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

import time

# Try importing optional dependency PyFFTW for Fourier transforms. 
# Numpy's fft_intel package might also be preferable
# If the import fails, import scipy's FFT module instead
try:
    from pyfftw.interfaces.scipy_fftpack import fft2, ifft2
except ImportError:
    try:
        from np.fft_intel import fft2, ifft2
    except ImportError:
        from scipy.fftpack import fft2, ifft2

__all__ = ['Hologram', 'ReconstructedWave', 'unwrap_phase']
RANDOM_SEED = 42
TWO_TO_N = [2**i for i in range(13)]


def rebin_image(a, binning_factor):
    # Courtesy of J.F. Sebastian: http://stackoverflow.com/a/8090605
    if binning_factor == 1:
        return a

    new_shape = (a.shape[0]/binning_factor, a.shape[1]/binning_factor)
    sh = (new_shape[0], a.shape[0]//new_shape[0], new_shape[1],
          a.shape[1]//new_shape[1])
    return a.reshape(map(int, sh)).mean(-1).mean(1)

def fftshift(x, additional_shift=None, axes=None):
    """
    Shift the zero-frequency component to the center of the spectrum, or with
    some additional offset from the center.

    This is a more generic fork of `~numpy.fft.fftshift`, which doesn't support
    additional shifts.


    Parameters
    ----------
    x : array_like
        Input array.
    additional_shift : list of length ``M``
        Desired additional shifts in ``x`` and ``y`` directions respectively
    axes : int or shape tuple, optional
        Axes over which to shift.  Default is None, which shifts all axes.

    Returns
    -------
    y : `~numpy.ndarray`
        The shifted array.
    """
    tmp = np.asarray(x)
    ndim = len(tmp.shape)
    if axes is None:
        axes = list(range(ndim))
    elif isinstance(axes, integer_types):
        axes = (axes,)

    # If no additional shift is supplied, reproduce `numpy.fft.fftshift` result
    if additional_shift is None:
        additional_shift = [0, 0]

    y = tmp
    for k, extra_shift in zip(axes, additional_shift):
        n = tmp.shape[k]
        if (n+1)//2 - extra_shift < n:
            p2 = (n+1)//2 - extra_shift
        else:
            p2 = abs(extra_shift) - (n+1)//2
        mylist = np.concatenate((np.arange(p2, n), np.arange(0, p2)))
        y = np.take(y, mylist, k)
    return y
    
def arrshift(x, shift, axes=None):
    """
    Shift array by ``shift`` along ``axes``.


    Parameters
    ----------
    x : array_like
        Input array.
    shift : list of length ``M``
        Desired additional shifts along respective axes
    axes : int or shape tuple, optional
        Axes over which to shift.  Default is None, which shifts all axes.

    Returns
    -------
    y : `~numpy.ndarray`
        The shifted array.
    """
    y = np.asarray(x)
    ndim = len(y.shape)
    if axes is None:
        axes = list(range(ndim))
    elif isinstance(axes, integer_types):
        axes = (axes,)
    
    for j, k in zip(axes, shift):
        n = y.shape[j];
        y = np.roll(y, (n+1)//2 - (-1)*k, axis=j)
    
    return y

def _load_hologram(hologram_path):
    """
    Load a hologram from path ``hologram_path`` using scikit-image and numpy.
    """
    return np.array(imread(hologram_path), dtype=np.float64)


def _find_peak_centroid(image, wavelength=405e-9, gaussian_width=10):
    """
    Smooth the image, find centroid of peak in the image.
    """
    wavelength = np.atleast_1d(wavelength).reshape((1,1,-1))
    F = gaussian_filter(image, gaussian_width) # Filter with a gaussian
    M = maximum_filter(F,3) # Get 8-neighbor maxima
    tfm = M==F # Maxima location TF array
    m = F[tfm] # Maxima
    idx = m.argsort()[::-1] # Sort the maxima, get the sorted indices in descending order
    x, y = np.nonzero(tfm) # Maxima locations
    # Grab top 2*#wavelengths + 1 peaks
    x = x[idx[0:(2*wavelength.shape[2]+1)]]
    y = y[idx[0:(2*wavelength.shape[2]+1)]]
    rsq = (x-image.shape[0]/2)**2 + (y-image.shape[1]/2)**2
    dist = np.sort(rsq)[1:] # Sort distances in ascending order
    idx = np.argsort(rsq)[1:] # Get sorted indices
    order = wavelength.reshape(-1).argsort()
    peaks = np.zeros([wavelength.shape[2],2])
    for o in order:
        i1 = idx[2*o]
        i2 = idx[2*o + 1]
        y1 = y[i1]
        y2 = y[i2]
        i = i1
        if y1 < y2:
            i = i2
        peaks[o,:] = [x[i], y[i]]
        
    return peaks

def _crop_image(image, crop_fraction):
    """
    Crop an image by a factor of ``crop_fraction``.
    """
    if crop_fraction == 0:
        return image

    crop_length = int(image.shape[0] * crop_fraction)

    if crop_length not in TWO_TO_N:
        message = ("Final dimensions after crop should be a power of 2^N. "
                   "Crop fraction of {0} yields dimensions ({1}, {1})"
                   .format(crop_fraction, crop_length))
        warnings.warn(message, CropEfficiencyWarning)

    cropped_image = image[crop_length//2:crop_length//2 + crop_length,
                          crop_length//2:crop_length//2 + crop_length]
    return cropped_image

def _crop_to_square(image):
    """
    Ensure that hologram is square.
    """
    sh = image.shape
    if sh[0] != sh[1]:
        square_image = image[:min(sh), :min(sh)]
    else:
        square_image = image

    return square_image


class CropEfficiencyWarning(AstropyUserWarning):
    pass
    
class MaskSizeWarning(AstropyUserWarning):
    pass
    
class UpdateError(Exception):
    pass
        
class SizeError(Exception):
    pass

class Hologram(object):
    """
    Container for holograms and methods to reconstruct them.
    """
    def __init__(self, hologram, crop_fraction=None, wavelength=405e-9,
                 rebin_factor=1, dx=3.45e-6, dy=3.45e-6):
        """
        Parameters
        ----------
        hologram : `~numpy.ndarray`
            Input hologram. If the hologram was taken with multiple wavelengths,
            the array should be a stack of single-wavelength hologram along axis 2.
        crop_fraction : float
            Fraction of the image to crop for analysis
        wavelength : float [meters] or iterable
            Wavelength of laser. Multiple wavelengths can be given as well.
        rebin_factor : int
            Rebin the image by factor ``rebin_factor``. Must be an even integer.
        dx : float [meters]
            Pixel width in x-direction (unbinned)
        dy : float [meters]
            Pixel width in y-direction (unbinned)

        Notes
        -----
        Non-square holograms will be cropped to a square with the dimensions of
        the smallest dimension. TODO: Why are we not zero-filling out to the next
        power of 2 instead of cropping?
        """
        wavelength = np.atleast_1d(wavelength).reshape((1,1,-1))

        self.crop_fraction = crop_fraction
        self.rebin_factor = rebin_factor

        # Rebin the hologram
        square_hologram = _crop_to_square(np.float64(hologram))
        binned_hologram = rebin_image(square_hologram, self.rebin_factor)

        # Crop the hologram by factor crop_factor, centered on original center
        if crop_fraction is not None:
            self.hologram = _crop_image(binned_hologram, crop_fraction)
        else:
            self.hologram = binned_hologram
        
        self.n = self.hologram.shape[0]
        self.wavelength = wavelength
        self.wavenumber = 2*np.pi/self.wavelength
        self._spectral_peak = None
        self._chromatic_shift = None
        self.dx = dx*rebin_factor
        self.dy = dy*rebin_factor
        self.mgrid = np.mgrid[0:self.n, 0:self.n]
        self.random_seed = RANDOM_SEED
        self.apodization_window_function = None
        self._ft_hologram = None;
        
    @property
    def ft_hologram(self, apodize=True):
        """
        `~numpy.ndarray` of the self.hologram FFT
        """
        if self._ft_hologram is None:
            if apodize==True:
                apodized_hologram = self.apodize(self.hologram)
                self._ft_hologram = fftshift(fft2(apodized_hologram))
            else:
                self._ft_hologram = fftshift(fft2(self.hologram))

        return self._ft_hologram
        
    @property
    def spectral_peak(self):
        if self._spectral_peak is None:
            # Guess 'em
            self.update_spectral_peak(self.fourier_peak_centroid())
        
        return self._spectral_peak
        
    @property
    def chromatic_shift(self):
        if self._chromatic_shift is None:
            self._chromatic_shift = np.zeros_like(self.wavelength);
            # Ideally we'd actually calculate the chromatic shift
        return self._chromatic_shift

    @classmethod
    def from_tif(cls, hologram_path, **kwargs):
        """
        Load a hologram from a TIF file.

        This class method takes the path to the TIF file as the first argument.
        All other arguments are the same as `~shampoo.Hologram`.

        Parameters
        ----------
        hologram_path : str
            Path to the hologram to load
        """
        hologram = _load_hologram(hologram_path)
        return cls(hologram, **kwargs)
        
    def reconstruct(self, propagation_distance, spectral_peak=None, fourier_mask=None, chromatic_shift=None):
        """
        Reconstruct the hologram at all ``propagation_distance`` for all ``self.wavelength``.
        
        Parameters
        ----------
        propagation_distances : float or iterable of float
            Propagation distance(s) to reconstruct
        spectral_peak : `~numpy.ndarray`
            Centroid of spectral peak for wavelength in power spectrum of hologram FT
            (len(self.wavelength) x 2)
        fourier_mask : array_like or None, optional
            Fourier-domain mask. If None (default), a mask is determined from the position of the
            main spectral peak. If array_like, the array will be cast to boolean.

        Returns
        -------
        reconstructed : ReconstructedWave
            Container object for the reconstructed wave.
        """

        propagation_distance = np.atleast_1d(propagation_distance)
        
        # Determine location of spectral peak
        # Did we specify a centroid? OK, use it.
        if spectral_peak is not None:
            self.update_spectral_peak(spectral_peak) 
            
        if chromatic_shift is not None:
            self.update_chromatic_shift(chromatic_shift)
            
        # Ignore Fourier masks that are of incorrect shape
        if fourier_mask is not None and (np.prod(fourier_mask.shape) != 
           np.prod(self.hologram.shape)*np.prod(self.wavelength.shape)*np.prod(propagation_distance.shape)):
            fourier_mask = None
            message = ("Fourier mask dimensions don't match hologram dimensions. Ignoring.")
            warnings.warn(message, MaskSizeWarning)
        
        if propagation_distance.size > 1:
            wave =  self._reconstruct_multithread(propagation_distance, fourier_mask = fourier_mask)
        else:
            wave = self._reconstruct(propagation_distance, fourier_mask)
            wave = np.expand_dims(wave, axis = 2)   # single prop. distance will have the wrong shape
        
        return ReconstructedWave(reconstructed_wave = wave, fourier_mask = fourier_mask, 
                                 wavelength = self.wavelength, depths = propagation_distance)

    def _reconstruct(self, propagation_distance, fourier_mask=None):
        """
        Reconstruct the wave at a single ``propagation_distance`` for a single ``wavelength``.

        Parameters
        ----------
        propagation_distance : float
            Propagation distance [m]
        spectral_peak : integer pair [x,y]
            Centroid of spectral peak for wavelength in power spectrum of hologram FT
        fourier_mask : array_like or None, optional
            Fourier-domain mask. If None (default), a mask is determined from the position of the
            main spectral peak.

        Returns
        -------
        reconstructed_wave : `~numpy.ndarray` ndim 3
            The reconstructed wave as an array of dimensions (X, Y, wavelengths)
        """
        x_peak, y_peak = self.spectral_peak
        
        # Calculate mask radius. TODO: Update 250 to an automated guess based on input values.
        if self.rebin_factor != 1:
            mask_radius = 150./self.rebin_factor
        elif self.crop_fraction is not None and self.crop_fraction != 0:
            mask_radius = 150.*self.crop_fraction
        else:
            mask_radius = 150.
        
        # Either use a Fourier-domain mask based on coords of spectral peak,
        # or a user-specified mask
        if fourier_mask is None:
            mask = self.real_image_mask(x_peak, y_peak, mask_radius)
        else:
            mask = np.asarray(fourier_mask, dtype=np.bool)
        mask = np.atleast_3d(mask)

        # Calculate Fourier transform of impulse response function
        G = self.fourier_trans_of_impulse_resp_func(np.atleast_1d([propagation_distance]*
                                self.wavelength.size).reshape((1,1,-1))-self.chromatic_shift)
        
        # Now calculate digital phase mask. First center the spectral peak for each channel
        x_peak, y_peak = x_peak.reshape(-1), y_peak.reshape(-1)
        shifted_ft_hologram = np.empty_like(np.atleast_3d(mask),dtype=np.complex128)
        for channel in range(self.wavelength.size):
            shifted_ft_hologram[:,:,channel] = arrshift(self.ft_hologram * mask[:,:,channel],
                                                        [-x_peak[channel], 
                                                         -y_peak[channel]],
                                                        axes = (0,1))
        
        # Apodize the result
        psi = self.apodize(shifted_ft_hologram * G)
        digital_phase_mask = self.get_digital_phase_mask(psi)

        # Reconstruct the image
        # fftshift is independent of channel
        psi = np.empty_like(np.atleast_3d(shifted_ft_hologram))

        for channel in range(psi.shape[2]):
            psi[:,:,channel] = arrshift(fftshift(fft2(self.apodize(self.hologram) * digital_phase_mask[:,:,channel], 
                                        axes = (0,1))) * 
                                        mask[:,:,channel],
                                        [-x_peak[channel], 
                                         -y_peak[channel]],
                                        axes = (0,1))
        psi *= G
        
        return fftshift(ifft2(psi, axes = (0,1)), axes = (0,1))

    def get_digital_phase_mask(self, psi):
        """
        Calculate the digital phase mask (i.e. reference wave), as in Colomb et
        al. 2006, Eqn. 26 [1]_.
        Fit for a second order polynomial, numerical parametric lens with least
        squares to remove tilt, spherical aberration.
        .. [1] http://www.ncbi.nlm.nih.gov/pubmed/16512526
        Parameters
        ----------
        psi : `~numpy.ndarray`
            The product of the Fourier transform of the hologram and the Fourier
            transform of impulse response function
        Returns
        -------
        phase_mask : `~numpy.ndarray`
            Digital phase mask, used for correcting phase aberrations.
        """
        inverse_psi = fftshift(ifft2(psi, axes = (0 ,1)), axes = (0, 1))

        unwrapped_phase_image = np.atleast_3d(unwrap_phase(inverse_psi))/2/self.wavenumber
        smooth_phase_image = gaussian_filter(unwrapped_phase_image, [50, 50, 0]) # do not filter along axis 2

        high = np.percentile(unwrapped_phase_image, 99)
        low = np.percentile(unwrapped_phase_image, 1)

        smooth_phase_image[high < unwrapped_phase_image] = high
        smooth_phase_image[low > unwrapped_phase_image] = low

        # Fit the smoothed phase image with a 2nd order polynomial surface with
        # mixed terms using least-squares.
        # This is iterated over all wavelength channels separately
        # TODO: can this be done on the smooth_phase_image along axis 2 instead
        # of direct iteration?
        smooth_phase_image = smooth_phase_image
        channels = np.split(smooth_phase_image, smooth_phase_image.shape[2], axis = 2)
        fits = list()

        # Need to flip mgrid indices for this least squares solution
        y, x = self.mgrid - self.n/2
        x, y = np.squeeze(x), np.squeeze(y)

        for channel in channels:
            v = np.array([np.ones(len(x[0, :])), x[0, :], y[:, 0], x[0, :]**2,
                        x[0, :] * y[:, 0], y[:, 0]**2])
            coefficients = np.linalg.lstsq(v.T, np.squeeze(channel))[0]
            fits.append(np.dot(v.T, coefficients))
            
        field_curvature_mask = np.stack(fits, axis = 2)
        digital_phase_mask = np.exp(-1j*self.wavenumber * field_curvature_mask)

        return digital_phase_mask

    def apodize(self, array, alpha=0.075):
        """
        Force the magnitude of an array to go to zero at the boundaries.
        Parameters
        ----------
        array : `~numpy.ndarray`
            Array to apodize
        alpha : float between zero and one
            Alpha parameter for the Tukey window function. For best results,
            keep between 0.075 and 0.2.
        Returns
        -------
        apodized_arr : `~numpy.ndarray`
            Apodized array
        """
        if self.apodization_window_function is None:
            x, y = self.mgrid
            n = len(x[0])
            tukey_window = tukey(n, alpha)
            self.apodization_window_function = np.atleast_3d(tukey_window[:, np.newaxis] * tukey_window)
        
        # In the most general case, array might represent a multi-wavelength hologram
        apodized_array = np.squeeze(np.atleast_3d(array) * self.apodization_window_function)
        return apodized_array
        
    def fourier_trans_of_impulse_resp_func(self, propagation_distance):
        """
        Calculate the Fourier transform of impulse response function, sometimes
        represented as ``G`` in the literature.
        For reference, see Eqn 3.22 of Schnars & Juptner (2002) Meas. Sci.
        Technol. 13 R85-R101 [1]_,
        .. [1] http://x-ray.ucsd.edu/mediawiki/images/d/df/Digital_recording_numerical_reconstruction.pdf
        Parameters
        ----------
        propagation_distance : float
            Propagation distance [m]
        Returns
        -------
        G : `~numpy.ndarray`
            Fourier transform of impulse response function
        """
        x, y = self.mgrid - self.n/2
        x, y = np.atleast_3d(x), np.atleast_3d(y)
        propagation_distance = np.atleast_3d(propagation_distance)
        first_term = (self.wavelength**2 * (x + self.n**2 * self.dx**2 /
                      (2.0 * propagation_distance * self.wavelength))**2 /
                      (self.n**2 * self.dx**2))
        second_term = (self.wavelength**2 * (y + self.n**2 * self.dy**2 /
                       (2.0 * propagation_distance * self.wavelength))**2 /
                       (self.n**2 * self.dy**2))
        G = np.exp(-1j * self.wavenumber * propagation_distance *
                   np.sqrt(1.0 - first_term - second_term))
        return G
        
    def real_image_mask(self, center_x, center_y, radius):
        """
        Calculate the Fourier-space mask to isolate the real image
    
        Parameters
        ----------
        center_x : `~numpy.ndarray`
            ``x`` centroid [pixels] of real image in Fourier space for each 
            image in a stack.
        center_y : `~numpy.ndarray`
            ``y`` centroid [pixels] of real image in Fourier space for each
            image in a stack.
        radius : float
            Radial width of mask [pixels] to apply to the real image in Fourier
            space
    
        Returns
        -------
        mask : `~numpy.ndarray`
            Binary-valued mask centered on the real-image peak in the Fourier
            transform of the hologram.
        """
        center_x, center_y = np.reshape(center_x, (1, 1, -1)), np.reshape(center_y, (1, 1, -1))
        x, y = self.mgrid
        x, y = x[:,:,None], y[:,:,None]
        x_shift = x-center_x
        y_shift = y-center_y
        mask = np.zeros_like(np.atleast_3d(x_shift), dtype = np.bool)
        mask[(x_shift)**2 + (y_shift)**2 < radius**2] = True

        # exclude corners
        #buffer = 20
        #if self.crop_fraction is not None:
        #    buffer = buffer*self.crop_fraction
        #mask[((x_shift)**2 + (y_shift)**2) < buffer**2] = 0.0

        return mask
    
    def fourier_peak_centroid(self, gaussian_width=10):
        """
        Calculate the centroid of the signal spike in Fourier space near the
        frequencies of the real image.

        Parameters
        ----------
        fourier_arr : `~numpy.ndarray`
            Fourier-transform of the hologram
        margin_factor : int
            Fraction of the length of the Fourier-transform of the hologram
            to ignore near the edges, where spurious peaks occur there.

        Returns
        -------
        pixel : `~numpy.ndarray`
            Pixel at the centroid of the spike in Fourier transform of the
            hologram near the real image.
        """
        
        return _find_peak_centroid(np.abs(self.ft_hologram), self.wavelength, gaussian_width)
        
        
    def _reconstruct_multithread(self, propagation_distances, fourier_mask=None):
        """
        Reconstruct phase or intensity for multiple distances, for one hologram.
        Parameters
        ----------
        propagation_distances : `~numpy.ndarray` or list
            Propagation distances to reconstruct
        threads : int
            Number of threads to use via `~multiprocessing`
        fourier_mask : array_like or None, optional
            Fourier-domain mask. If None (default), a mask is determined from the position of the main
            spectral peak. If array_like, the array will be cast to boolean.
        Returns
        -------
        wave_cube : `~numpy.ndarray`, ndim 4
        """ 
        cube = np.empty(shape = self.hologram.shape + (self.wavelength.size, len(propagation_distances)),
                        dtype = np.complex)
        with Pool(None) as pool:
            slices = pool.map( partial(self._reconstruct, fourier_mask = fourier_mask), propagation_distances)
        
        cube = np.stack(slices, axis = 3)        
        return np.swapaxes(cube, 2, 3)
    
    def update_spectral_peak(self, spectral_peak):
        """
        Update spectral peak centroid values.
        
        Parameters
        ----------
        spectral_peak : `~numpy.ndarray`
            Centroid of spectral peak for wavelength in power spectrum of hologram FT 
            (len(self.wavelength) x 2)
        """
        
        spectral_peak = np.atleast_2d(spectral_peak)
        
        if spectral_peak.shape[1] != 2 or spectral_peak.shape[0] != self.wavelength.shape[2]:
            message = ("Spectral peak array must be of shape {0} by 2. "
                       "{0} is the number of wavelengths."
                       .format(self.wavelength.shape[2]))
            raise UpdateError(message)

        spectral_peak = spectral_peak.astype(int)
        
        self._spectral_peak = spectral_peak.swapaxes(0,1)
        
    def update_chromatic_shift(self,chromatic_shift):
        """
        Update chromatic shift values for changed depth of focus for different wavelengths.
        
        Parameters
        ----------
        spectral_peak : `~numpy.ndarray`
            1xN_wavelength list of depth of focus changes
        """
        chromatic_shift = np.atleast_1d(chromatic_shift).reshape((1,1,-1))
        if chromatic_shift.shape[2] != self.wavelength.shape[2]:
            message = ("Chromatic shift must be of length {0} (number of wavelengths)."
                        .format(self.wavelength.shape[2]))
            raise UpdateError(message)
            
        self._chromatic_shift = chromatic_shift

def unwrap_phase(reconstructed_wave, wavelength=None):
    if wavelength is not None and wavelength.size == 3:
        return _unwrap_phase_multiwavelength(reconstructed_wave, wavelength)
    else:
        return _unwrap_phase(reconstructed_wave)

def _unwrap_phase(reconstructed_wave, seed=RANDOM_SEED):
    """
    2D phase unwrap a complex reconstructed wave.
    Essentially a wrapper around the `~skimage.restoration.unwrap_phase`
    function.
    Parameters
    ----------
    reconstructed_wave : `~numpy.ndarray`
        Complex reconstructed wave
    seed : float (optional)
        Random seed, optional.
    Returns
    -------
    `~numpy.ndarray`
        Unwrapped phase image
    """   
    phase = np.squeeze(2 * np.arctan(reconstructed_wave.imag / reconstructed_wave.real))
    
    # No channel, no need for shenanigans
    if phase.ndim < 3:
        return skimage_unwrap_phase(phase, seed=seed)

    # Each wavelength channel must be done separately
    unwrapped = np.empty_like(reconstructed_wave)
    unwrapped_channels = list()
    for phase_channel in np.dsplit(phase, phase.shape[2]):
        unwrapped_channels.append( skimage_unwrap_phase(np.squeeze(phase_channel),seed=seed) )
    return np.dstack(unwrapped_channels)
    
def _unwrap_phase_multiwavelength(reconstructed_wave, wavelength):
    """
    Perform multi-wavelength phase unwrapping.

    For reference, see N. Warnasooriya and M. K. Kim, Opt. Express 15, 9239-9247 (2007)

    .. [3] https://www.osapublishing.org/oe/fulltext.cfm?uri=oe-15-15-9239&id=139880

    Parameters
    ----------
    reconstructed_wave : `~numpy.ndarray`
        Complex reconstructed wave

    Returns
    -------
    `~numpy.ndarray`
        Unwrapped phase image
    """
    
    # TODO: Add 2-wavelength phase unwrapping.
    
    if reconstructed_wave.shape[2] != 3:
        message = ("_unwrap_phase_multiwavelength only works with 3-wavelength holograms.")
        raise SizeError(message)
    
    # Get the phase maps 
    # np.arctan2 returns in the range (-pi,pi) so we shift to (0, 2*pi)
    phase = np.squeeze(np.arctan2(reconstructed_wave.imag, reconstructed_wave.real) + np.pi)

    # Get the beat wavelengths
    lambda_13 = wavelength[:,:,0]*wavelength[:,:,2]/np.abs(wavelength[:,:,0]-wavelength[:,:,2])
    lambda_23 = wavelength[:,:,1]*wavelength[:,:,2]/np.abs(wavelength[:,:,1]-wavelength[:,:,2])
    lambda_1323 = lambda_13*lambda_23/np.abs(lambda_13-lambda_23)

    # Get the coarse maps
    coarse_13 = phase[:,:,0]-phase[:,:,2]
    coarse_23 = phase[:,:,1]-phase[:,:,2]
    coarse_13[coarse_13<0] = coarse_13[coarse_13<0] + 2*np.pi
    coarse_23[coarse_23<0] = coarse_23[coarse_23<0] + 2*np.pi
    coarse_1323 = coarse_13 - coarse_23
    coarse_1323[coarse_1323<0] = coarse_1323[coarse_1323<0] + 2*np.pi

    # Get the surface profiles
    z_13 = lambda_13*coarse_13/(2*np.pi)
    z_23 = lambda_23*coarse_23/(2*np.pi)
    z_1323 = lambda_1323*coarse_1323/(2*np.pi)

    # Get the integer surface profile for maximum beat wavelength
    z_a = np.rint(z_1323/lambda_13)*lambda_13
    z_b = z_a + z_13
    z_c = z_1323 - z_b
    z_d = z_c
    z_d[z_c[:] > lambda_13/2] = z_c[z_c[:] > lambda_13/2] + lambda_13.reshape(-1)
    z_d[z_c[:] < -lambda_13/2] = z_c[z_c[:] < -lambda_13/2] - lambda_13.reshape(-1)

    # Get surface profiles for individual wavelength
    z = wavelength*phase/(2*np.pi)
    #for channel in range(z.shape[2]):
    #    z[:,:,channel] = wavelength[:,:,channel]*phase[:,:,channel]/(2*np.pi)
    z_e = np.empty_like(z)
    for channel in range(z_e.shape[2]):
        z_e[:,:,channel] = np.rint(z_d/wavelength[:,:,channel])*wavelength[:,:,channel]
    z_f = z_e+z
    z_g = np.empty_like(z_f)
    for channel in range(z_g.shape[2]):
        z_g[:,:,channel] = z_d-z_f[:,:,channel]
    for channel in range(z_g.shape[2]):
        z_f_curr = z_f[:,:,channel]
        z_g_curr = z_g[:,:,channel]
        z_g_curr[z_f_curr[:] > wavelength[:,:,channel]/2] = z_f_curr[z_f_curr[:] > wavelength[:,:,channel]/2] + wavelength[:,:,channel].reshape(-1)
        z_g_curr[z_f_curr[:] < -wavelength[:,:,channel]/2] = z_f_curr[z_f_curr[:] < -wavelength[:,:,channel]/2] - wavelength[:,:,channel].reshape(-1)
        z_g[:,:,channel] = z_g_curr

    return z_g*2*np.pi/wavelength


class ReconstructedWave(object):
    """
    Container for reconstructed waves and their intensity and phase
    arrays.
    """
    def __init__(self, reconstructed_wave, fourier_mask, wavelength, depths):
        """
        Parameters
        ----------
        reconstructed_wave : array_like, complex
            Reconstructed wave. Last axis is wavelength channel. 
        fourier_mask : array_like
            Reconstruction Fourier mask, in 2- or 3- dimensions.
        wavelength : float or array_like
            Wavelength(s) of the reconstructed wave.
        depths : array_like
            Reconstruction depths, corresponding to each slice of `reconstructed_wave` along
            axis 2.
        """
        self.reconstructed_wave = reconstructed_wave
        self.depths = np.atleast_1d(depths)
        self._intensity_image = None
        self._phase_image = None
        self.fourier_mask = np.asarray(fourier_mask, dtype = np.bool)
        self.wavelength = np.atleast_1d(wavelength)
        self.random_seed = RANDOM_SEED
    
    @property
    def intensity(self):
        """
        `~numpy.ndarray` of the reconstructed intensity
        """
        if self._intensity_image is None:
            self._intensity_image = np.abs(self.reconstructed_wave)

        return self._intensity_image

    @property
    def phase(self):
        """
        `~numpy.ndarray` of the reconstructed, unwrapped phase.

        Returns the unwrapped phase using `~skimage.restoration.unwrap_phase`.
        """
        if self._phase_image is None:
            self._phase_image = unwrap_phase(self.reconstructed_wave, self.wavelength)

        return self._phase_image