from __future__ import absolute_import

import os
from multiprocessing import Queue

import numpy as np
import pyqtgraph as pg
from pyqtgraph import QtCore, QtGui

from ..reconstruction import Hologram, ReconstructedWave

ICONS_FOLDER = os.path.join(os.path.dirname(__file__), 'icons')
DEFAULT_PROPAGATION_DISTANCE = 0.03658

class TimeSeriesControls(QtGui.QWidget):
    """ Control of TimeSeries, as well as some metadata """
    
    time_point_request_signal = QtCore.pyqtSignal(float)

    def __init__(self, *args, **kwargs):
        super(TimeSeriesControls, self).__init__(*args, **kwargs)
        self.time_points = None

        self.filename_label = QtGui.QLabel('', self)
        self.filename_label.setAlignment(QtCore.Qt.AlignCenter)

        self.time_point_label = QtGui.QLabel('', self)
        self.time_point_label.setAlignment(QtCore.Qt.AlignLeft)

        self.time_point_slider = QtGui.QSlider(parent = self)
        self.time_point_slider.setTracking(True)
        self.time_point_slider.setOrientation(QtCore.Qt.Horizontal)

        # Update label with time_point
        self.time_point_slider.valueChanged.connect(self.time_point_label.setNum)
        self.time_point_slider.valueChanged.connect(self._update)

        time_label = QtGui.QLabel('Time point: ', self)
        time_label.setAlignment(QtCore.Qt.AlignLeft)
        controls = QtGui.QHBoxLayout()
        controls.addWidget(time_label)
        controls.addWidget(self.time_point_label)
        controls.addWidget(self.time_point_slider)

        title = QtGui.QLabel('<h3>Time-series controls</h3>')
        title.setAlignment(QtCore.Qt.AlignCenter)
        layout = QtGui.QVBoxLayout()
        layout.addWidget(title)
        layout.addWidget(self.filename_label)
        layout.addLayout(controls)
        self.setLayout(layout)
    
    @QtCore.pyqtSlot(dict)
    def update_metadata(self, metadata):
        """ Update TimeSeries metadata """
        if not metadata:
            self.hide()
            return
        
        self.time_points = metadata['time_points']
        self.time_point_slider.setRange(0, len(metadata['time_points']) - 1)
        self.time_point_slider.setValue(0)
        self.filename_label.setText('File: {}'.format(metadata['filename']))
        self.show()
    
    @QtCore.pyqtSlot(int)
    def _update(self, index):
        self.time_point_request_signal.emit(self.time_points[index])

class ReconstructedHologramViewer(QtGui.QWidget):
    """
    QWidget displaying the reconstructed wave holographic data, in two plots: phase and amplitude.

    Slots
    -----
    display
        Display the phase and amplitude information of a reconstructed hologram.
    
    clear
        Clear view.
    """

    def __init__(self, *args, **kwargs):
        """
        Parameters
        ----------
        parent : QObject
        """
        super(ReconstructedHologramViewer, self).__init__(*args, **kwargs)

        self.amplitude_viewer = pg.ImageView(parent = self, name = 'Reconstructed amplitude')
        self.phase_viewer = pg.ImageView(parent = self, name = 'Reconstructed phase')
        self.fourier_mask_viewer = pg.ImageView(parent = self, name = 'Reconstruction Fourier mask')

        self.tabs = QtGui.QTabWidget()
        self.tabs.addTab(self.amplitude_viewer, 'Amplitude')
        self.tabs.addTab(self.phase_viewer, 'Phase')
        self.tabs.addTab(self.fourier_mask_viewer, 'Fourier mask')

        self.layout = QtGui.QVBoxLayout()
        self.layout.addWidget(self.tabs)
        self.setLayout(self.layout)

        # Set to maximal size, since this is the star of the show
        self.resize(self.maximumSize())
    
    @QtCore.pyqtSlot(object)
    def display(self, reconstructed):
        """
        Dsplays the amplitude and phase information of a reconstructed hologram.

        Parameters
        ----------
        reconstructed : ReconstructedWave
        """
        # By swapping axes, we don't have to specify the axis order
        #axes = {0: 'x', 1:'y', 2:'t', 3:'c'}

        fourier_mask, depths = reconstructed.fourier_mask, reconstructed.depths

        self.amplitude_viewer.setImage(img = np.swapaxes(reconstructed.intensity, 0, 2), xvals = reconstructed.depths)
        self.phase_viewer.setImage(img = np.swapaxes(np.nan_to_num(reconstructed.phase), 0, 2), xvals = reconstructed.depths)
        #self.fourier_mask_viewer.setImage(img = fourier_mask)    #TODO: depths?
        
    @QtCore.pyqtSlot()
    def clear(self):
        self.amplitude_viewer.clear()
        self.phase_viewer.clear()
