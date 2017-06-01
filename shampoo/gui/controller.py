from __future__ import absolute_import

import numpy as np
from pyqtgraph import QtCore

from . import error_aware
from .. import Hologram, TimeSeries


class ShampooController(QtCore.QObject):
    """
    Underlying controller to SHAMPOO's Graphical User Interface

    Signals
    -------
    reconstructedto_be_reconstructed
        Emits a reconstructed hologram whenever one is available.
    
    raw_data_signal
        Emits holographic data whenever one is loaded into memory.
    
    camera_connected_signal
        Emits True when a camera has been successfully connected, and False when disconnected.
    
    Slots
    -------
    load_time_series[str]
        Load an HDF5-based TimeSeries object

    reconstruct[object]
        Send raw holographic data, to be reconstructed.
    
    send_snapshot_data
        Send raw holographic data to be reconstructed, from a camera snapshot.
    
    update_propagation_distance
        Change the propagation distance(s) used in the holographic reconstruction process.
    
    update_camera_features
    
    connect_camera
        Connect a camera by ID. Check for available cameras using available_cameras()
    
    Methods
    -------
    choose_camera
        Choose camera from a list of ID. Not implemented.
    """
    raw_data_signal = QtCore.pyqtSignal(object)
    reconstruction_parameters_signal = QtCore.pyqtSignal(dict)
    reconstructed_hologram_signal = QtCore.pyqtSignal(object)
    reconstruction_status_signal = QtCore.pyqtSignal(str)

    to_be_reconstructed = QtCore.pyqtSignal(object, dict)

    # Time series metadata
    time_series_metadata_signal = QtCore.pyqtSignal(dict)

    error_message_signal = QtCore.pyqtSignal(str)

    def __init__(self, **kwargs):
        super(ShampooController, self).__init__(**kwargs)

        self.time_series = None

        self.propagation_distance = list()
        self.fourier_mask = None
        
        self._reconstruction_thread = QtCore.QThread()
        self.reconstructor = QReconstructor()
        self.reconstructor.moveToThread(self._reconstruction_thread)
        self._reconstruction_thread.start()

        # Wire up reconstructor
        self.reconstruction_parameters_signal.connect(
            self.reconstructor.update_reconstruction_parameters)
        self.to_be_reconstructed.connect(self.reconstructor.reconstruct)

        # Propagation of signals across reconstructor and controller
        self.reconstructor.reconstructed_signal.connect(self.reconstructed_hologram_signal)
        self.reconstructor.reconstruction_status.connect(self.reconstruction_status_signal)
    
    def __del__(self):
        self._reconstruction_thread.quit()

    @error_aware('Time series could not be loaded')
    @QtCore.pyqtSlot(str)
    def load_time_series(self, path):
        """
        Load TimeSeries object into the controller

        Parameters
        ----------
        path : str
            Path to the HDF5 file
        """
        if self.time_series is not None:
            self.time_series.close()
        
        self.time_series = TimeSeries(path, mode = 'r+')
        metadata = dict(self.time_series.attrs)
        metadata.update({'filename': path})
        self.time_series_metadata_signal.emit(metadata)

    @QtCore.pyqtSlot(object)
    @QtCore.pyqtSlot(object, dict)
    def reconstruct(self, data, params = dict()):
        """ 
        Send holographic data to the reconstruction reactor.

        Parameters
        ----------
        data : Hologram object or TimeSeries
            Can be any type that can is accepted by the Hologram() constructor.
        """
        self.raw_data_signal.emit(data)
        self.to_be_reconstructed[object, dict].emit(data, params)
    
    @error_aware('Data could not be loaded from time-series')
    @QtCore.pyqtSlot(float)
    def data_from_time_series(self, time_point):
        """ Display raw data and reconstruction from TimeSeries """
        self.reconstruct(self.time_series, {'time_point': time_point})
    
    @error_aware('Fourier mask could not be set.')
    @QtCore.pyqtSlot(object)
    def set_fourier_mask(self, mask):
        self.reconstruction_parameters_signal.emit({'fourer_mask': img_as_bool(mask)})
    
    @error_aware('Propagation distance(s) could not be updated.')
    @QtCore.pyqtSlot(object)
    def update_propagation_distance(self, item):
        """
        Thread-safe PyQt slot API to updating the propagation distance. 

        Parameters
        ----------
        item : array-like
            Propagation distances in meters.
        """
        self.reconstruction_parameters_signal.emit({'propagation_distance': item})

class QReconstructor(QtCore.QObject):
    """ QObject responsible for reconstructing holograms """
    reconstructed_signal = QtCore.pyqtSignal(object)
    reconstruction_status = QtCore.pyqtSignal(str)

    def __init__(self, *args, **kwargs):
        super(QReconstructor, self).__init__(*args, **kwargs)

        self.reconstruction_parameters = dict()
        self.latest_hologram = None
    
    @QtCore.pyqtSlot(object, dict)
    def reconstruct(self, hologram, params = dict()):
        """ Hologram reconstruction. Note that because a TimeSeries has a reconstruct() 
        method that returns a ReconstructedWave, we can pass a TimeSeries as the hologram 
        parameter and include the time_point in params """
        self.latest_hologram = hologram

        self.reconstruction_status.emit('Reconstruction in progress...')
        reconstructed_wave = hologram.reconstruct(**self.reconstruction_parameters, **params)
        self.reconstructed_signal.emit(reconstructed_wave)
        self.reconstruction_status.emit('Reconstruction complete.')
    
    @QtCore.pyqtSlot(dict)
    def update_reconstruction_parameters(self, params):
        """ Update reconstruction parameters with new values, and 
        reconstruct latest with new parameters """
        self.reconstruction_parameters.update(params)
        if self.latest_hologram is not None:
            self.reconstruct(hologram = self.latest_hologram)