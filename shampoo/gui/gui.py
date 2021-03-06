"""
Graphical User Interface to the SHAMPOO API.
"""
from __future__ import absolute_import

from contextlib import suppress
from functools import wraps
import os.path
import sys
import traceback
from types import FunctionType

import numpy as np
import pyqtgraph as pg
from pyqtgraph import QtCore, QtGui
from skimage import img_as_bool
from skimage.io import imsave

from ..reconstruction import Hologram, ReconstructedWave
from ..time_series import TimeSeries
from .fourier_mask_dialog import FourierMaskDialog
from .hologram_viewer import HologramViewer
from .recon_params_widget import ReconstructionParametersWidget
from .time_series_creator import TimeSeriesCreator
from .time_series_reconstruction_dialog import TimeSeriesReconstructionDialog
from .widgets import ReconstructedHologramViewer, TimeSeriesControls

def run(*args, **kwargs):
    app = QtGui.QApplication(sys.argv)
    app.setStyle(QtGui.QStyleFactory.create('cde'))
    gui = App()
    sys.exit(app.exec_())

def error_aware(func):
    """
    Wrap an instance method with a try/except and emit a message.
    Instance must have a signal called 'error_message_signal' which
    will be emitted with the message upon error. 
    """
    @wraps(func)
    def aware_func(self, *args, **kwargs):
        try:
            return func(self, *args, **kwargs)
        except:
            exc = traceback.format_exc()
            self.error_message_signal.emit(exc)
    return aware_func

class ErrorAware(QtCore.pyqtWrapperType):
    
    def __new__(meta, classname, bases, class_dict):
        new_class_dict = dict()
        for attr_name, attr in class_dict.items():
            if isinstance(attr, FunctionType):
                attr = error_aware(attr)
            new_class_dict[attr_name] = attr
        
        return super().__new__(meta, classname, bases, new_class_dict)

class ShampooController(QtCore.QObject, metaclass = ErrorAware):
    """
    Underlying controller to SHAMPOO's Graphical User Interface
    """
    time_series_loaded = QtCore.pyqtSignal(bool)
    raw_data_signal = QtCore.pyqtSignal(object)
    reconstructed_hologram_signal = QtCore.pyqtSignal(object)
    time_series_metadata_signal = QtCore.pyqtSignal(dict)
    error_message_signal = QtCore.pyqtSignal(str)

    def __init__(self, **kwargs):
        super(ShampooController, self).__init__(**kwargs)
        self.time_series = None

        self.time_series_loaded.emit(False)
    
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

        self.data_from_time_series(metadata['time_points'][0])
        self.time_series_loaded.emit(True)
    
    @QtCore.pyqtSlot()
    def close_time_series(self):
        try: self.time_series.close()
        except: pass
        
        self.time_series_loaded.emit(False)
    
    @QtCore.pyqtSlot(dict)
    def assemble_time_series(self, params):
        """ Assemble a TimeSeries object from parameters """
        wavelengths = params['wavelengths'] 
        callback = params.pop('callback')
        done = params.pop('final_callback')
        total = len(params['hologram_paths'])

        callback(0)
        with TimeSeries(name = params['filename'], mode = 'w') as t:
            for index, path in enumerate(params['hologram_paths']):
                # TODO: choose time-points instead of index
                holo = Hologram.from_tif(path, wavelength = wavelengths)
                t.add_hologram(holo, time_point = index)
                callback(int(100*index / total))
        callback(100); done();
        self.load_time_series(params['filename'])
    
    @QtCore.pyqtSlot(dict)
    def reconstruct_time_series(self, params):
        done = params.pop('final_callback')
        self.time_series.batch_reconstruct(**params)
        done()
        self.data_from_time_series(self.time_series.time_points[0])
    
    @QtCore.pyqtSlot(float)
    def data_from_time_series(self, time_point):
        """ Display raw data and reconstruction from TimeSeries """
        self.raw_data_signal.emit(self.time_series.hologram(time_point))

        with suppress(ValueError):
            reconstructed = self.time_series.reconstructed_wave(time_point)
            self.reconstructed_hologram_signal.emit(reconstructed)

class App(QtGui.QMainWindow, metaclass = ErrorAware):
    """
    GUI shell to the ShampooController object.

    Widgets
    -------
    data_viewer
        View raw holographic data and associated Fourier plane information
    
    reconstructed_viewer
        View reconstructed holographic data

    reconstruction_parameters_widget
        Select the propagation distance(s) with which to reconstruct
        holograms.
    """

    error_message_signal = QtCore.pyqtSignal(str, name = 'error_message_signal')
    
    def __init__(self, **kwargs):
        """
        Parameters
        ----------
        debug : bool, optional
            If True, extra options are available as a debug tool. Default is False.
        """
        super(App, self).__init__(**kwargs)

        self._controller_thread = QtCore.QThread()
        self.controller = ShampooController()
        self.controller.moveToThread(self._controller_thread)
        self._controller_thread.start()

        self.data_viewer = HologramViewer(parent = self)
        self.reconstructed_viewer = ReconstructedHologramViewer(parent = self)

        self.time_series_controls = TimeSeriesControls(parent = self)
        self.time_series_controls.hide()
        self.controller.time_series_metadata_signal.connect(lambda d: self.time_series_controls.show())
        self.controller.time_series_metadata_signal.connect(self.time_series_controls.update_metadata)
        self.time_series_controls.time_point_request_signal.connect(self.controller.data_from_time_series)

        self.controller.reconstructed_hologram_signal.connect(self.reconstructed_viewer.display)
        self.controller.raw_data_signal.connect(self.data_viewer.display)

        self.file_dialog = QtGui.QFileDialog(parent = self)
        self.menubar = self.menuBar()

        # Assemble menu from previously-defined actions
        self.file_menu = self.menubar.addMenu('&File')

        raw_data_layout = QtGui.QVBoxLayout()
        raw_data_layout.addWidget(self.data_viewer)
        raw_data_layout.addWidget(self.time_series_controls)

        reconstructed_layout = QtGui.QVBoxLayout()
        reconstructed_layout.addWidget(self.reconstructed_viewer)

        self.error_message_signal.connect(self.show_error_message)
        self.controller.error_message_signal.connect(self.show_error_message)

        self.layout = QtGui.QHBoxLayout()
        self.layout.addLayout(raw_data_layout)
        self.layout.addLayout(reconstructed_layout)

        self.central_widget = QtGui.QWidget()
        self.central_widget.setLayout(self.layout)
        self.setCentralWidget(self.central_widget)

        self.setGeometry(500, 500, 800, 800)
        self.setWindowTitle('SHAMPOO')
        self._center_window()
        self.showMaximized()

        self.load_time_series_action = QtGui.QAction('&Load time-series', self)
        self.load_time_series_action.triggered.connect(self.load_time_series)
        self.file_menu.addAction(self.load_time_series_action)

        self.close_time_series_action = QtGui.QAction('&Close time-series', self)
        self.close_time_series_action.triggered.connect(self.controller.close_time_series)
        self.controller.time_series_loaded.connect(self.close_time_series_action.setEnabled)
        self.file_menu.addAction(self.close_time_series_action)

        self.file_menu.addSeparator()

        self.time_series_creator_action = QtGui.QAction('&Create hologram times-series', self)
        self.time_series_creator_action.triggered.connect(self.launch_time_series_creator)
        self.file_menu.addAction(self.time_series_creator_action)

        self.time_series_reconstruct_action = QtGui.QAction('&Reconstruct a hologram time-series', self)
        self.time_series_reconstruct_action.triggered.connect(self.launch_time_series_reconstruction)
        self.controller.time_series_loaded.connect(self.time_series_reconstruct_action.setEnabled)
        self.file_menu.addAction(self.time_series_reconstruct_action)
    
    @QtCore.pyqtSlot()
    def load_time_series(self):
        path = self.file_dialog.getOpenFileName(self, 'Load time-series', filter = '*hdf5 *.h5')[0]
        if path:
            self.controller.load_time_series(path)
    
    @QtCore.pyqtSlot()
    def launch_time_series_creator(self):
        time_series_creator = TimeSeriesCreator(parent = self)
        time_series_creator.time_series_assembly.connect(self.controller.assemble_time_series)
        success = time_series_creator.exec_()
    
    @QtCore.pyqtSlot(str)
    def show_error_message(self, message):
        error_window = QtGui.QErrorMessage(parent = self)
        error_window.showMessage(message)
    
    @QtCore.pyqtSlot()
    def launch_time_series_reconstruction(self):
        try:
            nwavelengths = len(self.controller.time_series.wavelengths)
        except:
            nwavelengths = 1
        
        holo = self.controller.time_series.hologram(time_point = self.controller.time_series.time_points[0])
        time_series_reconstruction = TimeSeriesReconstructionDialog(parent = self, nwavelengths = nwavelengths, 
                                                                    hologram = np.array(holo.hologram, copy = True))
        time_series_reconstruction.reconstruction_parameters.connect(self.controller.reconstruct_time_series)
        success = time_series_reconstruction.exec_()
        
    def closeEvent(self, event):
        reply = QtGui.QMessageBox.question(self, 'SHAMPOO', 'Are you sure you want to quit?', 
                                           QtGui.QMessageBox.Yes, QtGui.QMessageBox.No)
        if reply == QtGui.QMessageBox.Yes:
            event.accept()
            self._controller_thread.quit()
        else:
            event.ignore()
    
    def _center_window(self):
        qr = self.frameGeometry()
        cp = QtGui.QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())
