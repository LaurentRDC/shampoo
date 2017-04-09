"""
Graphical User Interface to the SHAMPOO API.
"""
from __future__ import absolute_import

from .camera import available_cameras, AlliedVisionCamera
from .debug import DebugCamera
import functools
from .fourier_mask_dialog import FourierMaskDialog
import numpy as np
import os.path
from pyqtgraph import QtGui, QtCore
import pyqtgraph as pg
from .reactor import Reconstructor
from ..reconstruction import Hologram, ReconstructedWave
from skimage.io import imsave
from skimage import img_as_bool
import sys
from .time_series_creator import TimeSeriesCreator
from ..time_series import TimeSeries
import traceback
from .widgets import (RawDataViewer, ReconstructedHologramViewer, 
                      PropagationDistanceSelector, CameraFeatureDialog, ShampooStatusBar)
                    
# Try importing optional dependency PyFFTW for Fourier transforms. If the import
# fails, import scipy's FFT module instead
try:
    from pyfftw.interfaces.scipy_fftpack import fft2, ifft2
except ImportError:
    from scipy.fftpack import fft2, ifft2

def error_aware(message):
    """
    Wrap an instance method with a try/except.
    Instance must have a signal called 'error_message_signal' which
    will be emitted with the message upon error. 
    """
    def wrap(func):
        @functools.wraps(func)
        def aware_func(self, *args, **kwargs):
            try:
                return func(self, *args, **kwargs)
            except: 
                exc = traceback.format_exc()
                self.error_message_signal.emit(message + '\n \n \n' + exc)
        return aware_func
    return wrap

def run(debug = False):
    """
    
    """
    app = QtGui.QApplication(sys.argv)
    app.setStyle(QtGui.QStyleFactory.create('cde'))
    gui = App(debug = debug)
    sys.exit(app.exec_())

class ShampooController(QtCore.QObject):
    """
    Underlying controller to SHAMPOO's Graphical User Interface

    Signals
    -------
    reconstructed_hologram_signal
        Emits a reconstructed hologram whenever one is available.
    
    raw_data_signal
        Emits holographic data whenever one is loaded into memory.
    
    camera_connected_signal
        Emits True when a camera has been successfully connected, and False when disconnected.
    
    Slots
    -------
    send_data
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
    raw_data_signal = QtCore.pyqtSignal(object, name = 'raw_data_signal')
    reconstructed_hologram_signal = QtCore.pyqtSignal(object, name = 'reconstructed_hologram_signal')

    # Status signals
    reconstruction_in_progress_signal = QtCore.pyqtSignal(str, name = 'reconstruction_in_progress_signal')
    reconstruction_complete_signal = QtCore.pyqtSignal(str, name = 'reconstruction_complete_signal')
    camera_connected_signal = QtCore.pyqtSignal(bool, name = 'camera_connected_signal')

    error_message_signal = QtCore.pyqtSignal(str, name = 'error_message_signal')

    def __init__(self, **kwargs):
        super(ShampooController, self).__init__(**kwargs)

        self.time_series = None

        self.propagation_distance = list()
        self.fourier_mask = None
        
        self.camera = None
        self.camera_connected_signal.emit(False)

        # Hologram reconstruction and display
        def display_callback(item):
            self.reconstructed_hologram_signal.emit(item)
            self.reconstruction_complete_signal.emit('Reconstruction complete') 
        
        self.reconstruction_reactor = Reconstructor(callback = display_callback)
        self.reconstruction_reactor.start()

        # Private attributes
        self._latest_hologram = None
    
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

    @QtCore.pyqtSlot()
    def send_snapshot_data(self):
        """
        Send holographic data from the camera to the reconstruction reactor.
        """
        data = self.camera.snapshot()
        self.send_data(data)
    
    @QtCore.pyqtSlot(object)
    def send_data(self, data):
        """ 
        Send holographic data to the reconstruction reactor.

        Parameters
        ----------
        data : ndarray or Hologram object
            Can be any type that can is accepted by the Hologram() constructor.
        """
        if not isinstance(data, Hologram):
            data = Hologram(data)
        
        self._latest_hologram = data
        self.raw_data_signal.emit(data)

        self.reconstruction_reactor.send_item( (self.propagation_distance, data, self.fourier_mask) )
        self.reconstruction_in_progress_signal.emit('Reconstruction in progress...')
    
    @error_aware('Latest hologram could not be saved.')
    @QtCore.pyqtSlot(object)
    def save_latest_hologram(self, path):
        """
        Save latest raw holographic data into a HDF5

        Parameters
        ----------
        path : str or path-like object
        """
        imsave(path, arr = self._latest_hologram.hologram, plugin = 'tifffile')
    
    @error_aware('Fourier mask could not be set.')
    @QtCore.pyqtSlot(object)
    def set_fourier_mask(self, mask):
        self.fourier_mask = img_as_bool(mask)
        # Refresh screen
        if self._latest_hologram:
            self.send_data(self._latest_hologram)
    
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
        self.propagation_distance = item
        # Refresh screen
        if self._latest_hologram:
            self.send_data(self._latest_hologram)
    
    @error_aware('Camera features could not be updated.')
    @QtCore.pyqtSlot(dict)
    def update_camera_features(self, feature_dict):
        """ 
        Update camera features (e.g. exposure, bit depth) according to a dictionary.
        
        Parameters
        ----------
        feature_dict : dict
        """
        if not self.camera:
            return
        
        for feature, value in feature_dict.items():
            setattr(self.camera, feature, value)
    
    @error_aware('Camera could not be connected.')
    @QtCore.pyqtSlot(object)
    def connect_camera(self, ID):
        """ 
        Connect camera by ID. 
        
        Parameters
        ----------
        ID : str
            String identifier to a camera. If 'debug', a dummy DebugCamera
            instance will be connected.
        """
        # TODO: generalize to other manufacturers
        # This method should never fail. available_cameras() must have been called
        # before so that connecting is always successful.
        if ID == 'debug':
            self.camera = DebugCamera()
        else:
            self.camera = AlliedVisionCamera(ID)
        self.camera_connected_signal.emit(True)
    
    def stop(self):
        """ Stop all reactors. """
        self.reconstruction_reactor.stop()

class App(QtGui.QMainWindow):
    """
    GUI shell to the ShampooController object.

    Widgets
    -------
    data_viewer
        View raw holographic data and associated Fourier plane information
    
    reconstructed_viewer
        View reconstructed holographic data

    propagation_distance_selector
        Select the propagation distance(s) with which to reconstruct
        holograms.
    """

    save_latest_hologram_signal = QtCore.pyqtSignal(object, name = 'save_latest_hologram_signal')
    connect_camera_signal = QtCore.pyqtSignal(object, name = 'connect_camera_signal')

    error_message_signal = QtCore.pyqtSignal(str, name = 'error_message_signal')
    
    def __init__(self, debug = False):
        """
        Parameters
        ----------
        debug : bool, optional
            If True, extra options are available as a debug tool. Default is False.
        """
        super(App, self).__init__()

        self.controller = ShampooController()
        self.debug = debug

        self.data_viewer = RawDataViewer(parent = self)
        self.reconstructed_viewer = ReconstructedHologramViewer(parent = self)
        self.propagation_distance_selector = PropagationDistanceSelector(parent = self)

        self.file_dialog = QtGui.QFileDialog(parent = self)
        self.menubar = self.menuBar()

        # Assemble menu from previously-defined actions
        self.file_menu = self.menubar.addMenu('&File')
        self.camera_menu = self.menubar.addMenu('&Camera')

        # Assemble window
        self.main_splitter = QtGui.QSplitter(QtCore.Qt.Horizontal)
        self.right_splitter = QtGui.QSplitter(QtCore.Qt.Vertical)

        self.right_splitter.addWidget(self.reconstructed_viewer)
        self.right_splitter.addWidget(self.propagation_distance_selector)
        self.main_splitter.addWidget(self.data_viewer)
        self.main_splitter.addWidget(self.right_splitter)

        self.status_bar = ShampooStatusBar(parent = self)
        self.setStatusBar(self.status_bar)
        self.status_bar.update_status('Ready')

        self.error_window = QtGui.QErrorMessage(parent = self)
        self.error_message_signal.connect(self.error_window.showMessage)
        self.controller.error_message_signal.connect(self.error_window.showMessage)

        self.layout = QtGui.QVBoxLayout()
        self.layout.addWidget(self.main_splitter)

        self.central_widget = QtGui.QWidget()
        self.central_widget.setLayout(self.layout)
        self.setCentralWidget(self.central_widget)

        self.setGeometry(500, 500, 800, 800)
        self.setWindowTitle('SHAMPOO')
        self._center_window()
        self.showMaximized()

        self.load_data_action = QtGui.QAction('&Load raw data', self)
        self.load_data_action.triggered.connect(self.load_data)
        self.file_menu.addAction(self.load_data_action)

        self.save_data_action = QtGui.QAction('&Save raw data', self)
        self.save_data_action.triggered.connect(self.save_raw_data)
        self.file_menu.addAction(self.save_data_action)
        self.save_data_action.setEnabled(False)

        self.load_fourier_mask_action = QtGui.QAction('&Load Fourier mask', self)
        self.load_fourier_mask_action.triggered.connect(self.load_fourier_mask)
        self.file_menu.addAction(self.load_fourier_mask_action)

        self.time_series_creator_action = QtGui.QAction('&Create hologram time series', self)
        self.time_series_creator_action.triggered.connect(self.launch_time_series_creator)
        self.file_menu.addAction(self.time_series_creator_action)

        self.connect_camera_action = QtGui.QAction('&Connect a camera', self)
        self.connect_camera_action.triggered.connect(self.connect_camera)
        self.camera_menu.addAction(self.connect_camera_action)

        self.camera_snapshot_action = QtGui.QAction('&Take camera snapshot', self)
        self.camera_snapshot_action.triggered.connect(self.controller.send_snapshot_data)
        self.camera_menu.addAction(self.camera_snapshot_action)
        self.camera_snapshot_action.setEnabled(False)

        self.camera_features_action = QtGui.QAction('&Change camera features', self)
        self.camera_features_action.triggered.connect(self.change_camera_features)
        self.camera_menu.addAction(self.camera_features_action)
        self.camera_features_action.setEnabled(False)

        self.export_reconstructed_action = QtGui.QAction('&Export current reconstructed data (placeholder)', self)
        self.file_menu.addAction(self.export_reconstructed_action)
        self.export_reconstructed_action.setEnabled(False)

        self.propagation_distance_selector.propagation_distance_signal.connect(self.controller.update_propagation_distance)
        self.controller.reconstructed_hologram_signal.connect(self.reconstructed_viewer.display)
        self.controller.raw_data_signal.connect(self.data_viewer.display)

        # Save and loads
        self.save_latest_hologram_signal.connect(self.controller.save_latest_hologram)

        # Controller status signals
        self.connect_camera_signal.connect(self.controller.connect_camera)
        self.controller.reconstruction_in_progress_signal.connect(self.status_bar.update_status)
        self.controller.reconstruction_complete_signal.connect(self.status_bar.update_status)

        # What actions are available when a camera is made available
        # These actions will become unavailable when a camera is disconnected.
        self.controller.camera_connected_signal.connect(lambda x: self.status_bar.update_status('Camera connected'))
        self.controller.camera_connected_signal.connect(self.camera_snapshot_action.setEnabled)
        self.controller.camera_connected_signal.connect(self.camera_features_action.setEnabled)
        self.controller.raw_data_signal.connect(lambda x: self.save_data_action.setEnabled(True))

        self.propagation_distance_selector.update_propagation_distance()

    @error_aware('Data could not be loaded.')
    @QtCore.pyqtSlot()
    def load_data(self):
        """ Load a hologram into memory and displays it. """
        path = self.file_dialog.getOpenFileName(self, 'Load holographic data', filter = '*tif')[0]
        hologram = Hologram.from_tif(os.path.abspath(path))
        self.controller.send_data(data = hologram)
    
    @error_aware('Fourier mask could not be loaded')
    @QtCore.pyqtSlot()
    def load_fourier_mask(self):
        """ Load a user-defined reconstruction Fourier mask """
        fourier_mask_dialog = FourierMaskDialog(initial_mask = self.controller.fourier_mask)
        fourier_mask_dialog.fourier_mask_update_signal.connect(self.controller.set_fourier_mask)
        success = fourier_mask_dialog.exec_()
    
    @error_aware('The hologram time series could not be created.')
    @QtCore.pyqtSlot()
    def launch_time_series_creator(self):
        time_series_creator = TimeSeriesCreator(parent = self)
        time_series_creator.time_series_path_signal.connect(self.controller.load_time_series)
        success = time_series_creator.exec_()
    
    @error_aware('Raw data could not be saved.')
    @QtCore.pyqtSlot()
    def save_raw_data(self):
        """ Save a raw hologram from the raw data screen """
        path = self.file_dialog.getSaveFileName(self, 'Save holographic data', filter = '*tif')
        if not path.endswith('.tif'):
            path = path + '.tif'
        self.save_latest_hologram_signal.emit(path)
    
    @error_aware('Camera could not be connected')
    @QtCore.pyqtSlot()
    def connect_camera(self):
        """ Bring up a modal dialog to choose amongst available cameras. """
        cameras = available_cameras()

        if self.debug:
            cameras.append('debug')
        
        if not cameras:
            self.error_message_signal.emit('No cameras available.')
            return
        
        camera_ID, ok = QtGui.QInputDialog.getItem(self, 'Select camera', 'List of cameras', 
                                                   cameras, 0, False)
        
        if ok and camera_ID:
            self.connect_camera_signal.emit(camera_ID)
    
    @error_aware('Camera features could not be updated.')
    @QtCore.pyqtSlot()
    def change_camera_features(self):
        self.camera_features_dialog = CameraFeatureDialog(camera = self.controller.camera, parent = self)
        self.camera_features_dialog.camera_features_update_signal.connect(self.controller.update_camera_features)
        success = self.camera_features_dialog.exec_()
        if not success:
            raise RuntimeError
    
    def closeEvent(self, event):
        reply = QtGui.QMessageBox.question(self, 'SHAMPOO', 'Are you sure you want to quit?', 
                                           QtGui.QMessageBox.Yes, QtGui.QMessageBox.No)

        if reply == QtGui.QMessageBox.Yes:
            event.accept()
            self.controller.stop()
        else:
            event.ignore()
    
    def _center_window(self):
        qr = self.frameGeometry()
        cp = QtGui.QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())