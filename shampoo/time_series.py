# -*- coding: utf-8 -*-
"""
This module implements storage of holographic time-series
via an HDF5 file.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from collections import Iterable

import h5py
import numpy as np

from .reconstruction import Hologram, ReconstructedWave

class TimeSeries(h5py.File):
    """
    Holographic time-series as an HDF5 archive.

    Attributes
    ----------
    time_points : tuple of floats
        Time-points in seconds
    wavelengths : tuple of floats
        Wavelengths in nm.
    """
    _default_ckwargs = {'chunks': True, 
                        'compression':'lzf', 
                        'shuffle': True}
    
    @property
    def time_points(self):
        return tuple(self.attrs.get('time_points', default = tuple()))
    
    @property
    def wavelengths(self):
        return tuple(self.attrs.get('wavelengths', default = tuple()))
    
    @property
    def depths(self):
        return tuple(self.attrs.get('depths', default = tuple()))

    @property
    def hologram_group(self):
        return self.require_group('holograms')
    
    @property
    def reconstructed_group(self):
        return self.require_group('reconstructed')
    
    @property
    def fourier_mask_group(self):
        return self.require_group('/reconstructed/fourier_masks')

    def add_hologram(self, hologram, time_point = 0):
        """
        Add a hologram to the time-series.

        Parameters
        ----------
        hologram : Hologram

        time_point : float, optional

        Raises
        ------
        ValueError
            If the hologram is not compatible with the current TimeSeries,
            e.g. the wavelengths do not match.
        """
        # TODO: Holograms are stored as UINT8
        #       Is that sensible?
        holo_wavelengths = tuple(hologram.wavelength.reshape((-1)))
        time_point = float(time_point)

        if len(self.time_points) == 0:
            # This is the first hologram. Record the wavelength
            # and this will never change again.
            self.attrs['wavelengths'] = holo_wavelengths
        
        # The entire TimeSeries has the uniform wavelengths
        if not np.allclose(holo_wavelengths, self.wavelengths):
            raise ValueError('Wavelengths of this hologram ({}) do not match the TimeSeries \
                              wavelengths ({})'.format(holo_wavelengths, self.wavelengths))

        # If time-point already exists, we will override the hologram
        # that is already stored there. Otherwise, create a new dataset
        gp = self.hologram_group
        if time_point in self.time_points:
            return gp[str(time_point)].write_direct(np.atleast_3d(hologram.hologram))
        else:
            self.attrs['time_points'] = self.time_points + (time_point, )
            return gp.create_dataset(str(time_point), data = np.atleast_3d(hologram.hologram), 
                                     dtype = np.uint8, **self._default_ckwargs)
    
    def hologram(self, time_point, **kwargs):
        """
        Return Hologram object from archive. Keyword arguments are
        passed to the Hologram constructor.
        
        Parameters
        ----------
        time_point : float
            Time-point in seconds.
        
        Returns
        -------
        out : Hologram

        Raises
        ------
        ValueError
            If the time-point hasn't been recorded in the time-series.
        """
        time_point = float(time_point)
        if time_point not in self.time_points:
            raise ValueError('Time-point {} not in TimeSeries.'.format(time_point))
        
        dset = self.hologram_group[str(time_point)]
        return Hologram(np.array(dset), wavelength = self.wavelengths, **kwargs)

    def reconstruct(self, time_point, propagation_distance, 
                    fourier_mask = None, **kwargs):
        """
        Hologram reconstruction from Hologram.reconstruct(). Keyword arguments
        are also passed to Hologram.reconstruct()
        
        Parameters
        ----------
        time_point : float
            Time-point in seconds.
        propagation_distance : float
            Propagation distance in meters.
        fourier_mask : ndarray or None, optional
            User-specified Fourier mask. Refer to Hologram.reconstruct()
            documentation for details.
        
        Returns
        -------
        out : ReconstructedWave object
            The ReconstructedWave is both stored in the TimeSeries HDF5 file
            and returned to the user. 
        """
        time_point = float(time_point)
        propagation_distance = np.atleast_1d(propagation_distance)

        # TODO: provide an accumulator array for hologram.reconstruct()
        #       so that depths are written to disk on the fly?
        hologram = self.hologram(time_point)
        recon_wave = hologram.reconstruct(propagation_distance, 
                                          fourier_mask = fourier_mask,
                                          **kwargs)

        if propagation_distance.size == 1:
            recon_wave = {float(propagation_distance):recon_wave}
        
        gp = self.reconstructed_group.require_group(str(time_point))
        fg = self.fourier_mask_group.require_group(str(time_point))

        for dist, wave in recon_wave.items():
            gp.create_dataset(str(dist), data = wave.reconstructed_wave, 
                                dtype = np.complex, **self._default_ckwargs)
            fg.create_dataset(str(dist), data = wave.fourier_mask, 
                                dtype = np.bool,**self._default_ckwargs)
        
        # Return the same thins as Hologram.reconstruct() so that the TimeSeries can be passed
        # to anything that expect a reconstruct() method.
        if len(propagation_distance) == 1:
            return recon_wave[float(propagation_distance)]
        return recon_wave
    
    def reconstructed_wave(self, time_point, **kwargs):
        """
        Returns the ReconstructedWave object from archive. 
        Keyword arguments are passed to the ReconstructedWave 
        contructor.

        Parameters
        ----------
        time_point : float
            Time-point in seconds.
        
        Returns
        -------
        out : ReconstructedWave object

        Raises
        ------
        ValueError
            If the reconstruction is unavailable either due to having no
            associated hologram, or reconstruction never having been performed.
        """
        time_point = str(float(time_point))

        gp, fp = self.reconstructed_group, self.fourier_mask_group
        if time_point not in gp:
            raise ValueError('Reconstruction at {} is unavailable or reconstruction \
                              was never performed.'.format(time_point))
        
        recon_wave = dict()
        for dist in gp[time_point].keys():
            wave, mask = np.array(gp[time_point][str(dist)]), np.array(fp[time_point][str(dist)])
            recon_wave[dist] = ReconstructedWave(wave, fourier_mask = mask, wavelength = self.wavelengths)
        
        # We keep in line with the output of Hologram.reconstruct(); for a single
        # propagation distance the returned value is a ReconstructedWave, and
        # a dictionary for multiple propagation distances
        if len(recon_wave) == 1:    # Only a single proapgation distance
            return recon_wave[recon_wave.keys()[0]]
        return recon_wave
    
    def batch_reconstruct(self, propagation_distance, fourier_mask = None,
                          callback = None, **kwargs):
        """ 
        Reconstruct all the holograms stored in the TimeSeries. Keyword 
        arguments are passed to the Hologram.reconstruct() method. 
        
        Parameters
        ----------
        time_point : float
            Time-point in seconds.
        propagation_distance : float
            Propagation distance in meters.
        fourier_mask : ndarray or None, optional
            User-specified Fourier mask. Refer to Hologram.reconstruct()
            documentation for details.
        callback : callable, optional
            Callable that takes an int between 0 and 99. The callback will be
            called after each reconstruction with the proportion of completed
            reconstruction.
        """
        if callback is None:
            callback = lambda i: None 
            
        total = len(self.time_points)
        
        for index, time_point in enumerate(self.time_points):
            self.reconstruct(time_point = time_point, 
                             propagation_distance = propagation_distance,
                             fourier_mask = fourier_mask, **kwargs)
            callback(int(100*index / total))